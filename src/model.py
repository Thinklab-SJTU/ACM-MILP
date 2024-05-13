import json
from typing import List, Union
from collections import defaultdict

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor, LongTensor
from torch_geometric.data import Batch
from torch_geometric.utils import sort_edge_index

import src.tb_writter as tb_writter
from src.data import BipartiteGraph
from src.modules import (Bias_Predictor, Degree_Predictor, Logits_Predictor, Edge_Selector,
                         ReSample, Weights_Predictor)
from src.nn import BipartiteGraphEmbedding, BipartiteGraphGNN


class ACMMILP(nn.Module):

    def __init__(self,
                 config: DictConfig,
                 data_stats: dict
                 ):
        """
        ACMMILP model, the main model.

        Args:
            config: model config
            data_stats: dataset statistics, used to configure the model
        """
        super().__init__()

        self.config = config

        self.embedding_layer = BipartiteGraphEmbedding(config.graph_embedding)
        self.encoder_layer_1 = BipartiteGraphGNN(config.gnn)
        self.encoder_layer_2 = BipartiteGraphGNN(config.gnn)
        self.decoder_layer = BipartiteGraphGNN(config.gnn)
        self.resample_layer = ReSample(config.resample)

        self.bias_predictor = Bias_Predictor(
            config.bias_predictor,
            data_stats,
        )
        self.degree_predictor = Degree_Predictor(
            config.degree_predictor,
            data_stats,
        )
        self.logits_predictor = Logits_Predictor(
            config.logits_predictor,
            data_stats,
        )
        self.edge_selector = Edge_Selector(
            config.edge_selector,
            data_stats,
        )
        self.weights_predictor = Weights_Predictor(
            config.weights_predictor,
            data_stats,
        )

    def train_forward(self, data: Union[Batch, BipartiteGraph],
                          community_idx: LongTensor,
                          beta_cons: float,
                          beta_var: float) -> Tensor:
        """
        Forward pass of the model (for training).
        Inputs:
        data: batch of bipartite graphs
        community_idx: constraint node idx in each community to be substituted
        beta_cons: coefficient of the KL loss for constraints
        beta_var: coefficient of the KL loss for variables

        Output:
        loss: loss of the model

        """
        orig_graph = self.embedding_layer.embed_graph(data)
        z_conss, z_vars = self.encoder_layer_1.forward(orig_graph)

        z_conss, z_vars, cons_kl_loss, var_kl_loss = self.resample_layer.forward(
            z_conss, z_vars)

        h_conss, h_vars = self.encoder_layer_2.forward(orig_graph)

        resampled_graph = orig_graph.clone()
        resampled_graph = self.update_labels(data, resampled_graph, community_idx)

        h_conss[community_idx] = z_conss[community_idx]
        h_vars[resampled_graph.connected_vars_idx] = z_vars[resampled_graph.connected_vars_idx]
        resampled_graph.x_constraints = h_conss
        resampled_graph.x_variables = h_vars

        p_conss, p_vars = self.decoder_layer.forward(resampled_graph)

        p_edge_selection = []
        for i, c_list in enumerate(resampled_graph.involved_constraints):
            num_variables = len(resampled_graph.involved_variables[i])
            for c_idx in c_list:
                p_cv = torch.cat((p_vars[resampled_graph.involved_variables[i]], p_conss[c_idx].repeat(num_variables, 1)), dim=-1)
                p_edge_selection.append(p_cv)

        p_edge_selection = torch.cat(p_edge_selection, dim=0)

        p_weight = []
        for i, c_list in enumerate(resampled_graph.involved_constraints):
            for c_idx in c_list:
                num_logits = len(resampled_graph.connected_vars[c_idx.item()])
                p_cv = torch.cat((p_vars[resampled_graph.connected_vars[c_idx.item()]], p_conss[c_idx].repeat(num_logits, 1)), dim=-1)
                p_weight.append(p_cv)

        p_weight = torch.cat(p_weight, dim=0)

        cons_loss, _ = self.bias_predictor.forward(
            p_conss, community_idx, resampled_graph.bias_label)

        degree_loss, _ = self.degree_predictor.forward(
            p_conss, community_idx, resampled_graph.degree_label)

        logits_loss, _ = self.logits_predictor.forward(
            p_vars, resampled_graph.logits_label)

        # add an edge selection loss
        edge_selection_loss, _ = self.edge_selector.forward(
            p_edge_selection, resampled_graph.edge_selection_label)

        weights_loss, _ = self.weights_predictor.forward(
            p_weight, resampled_graph.weights_label)

        cons_loss = cons_loss * self.config.loss_weights.cons_loss
        degree_loss = degree_loss * self.config.loss_weights.degree_loss
        logits_loss = logits_loss * self.config.loss_weights.logits_loss
        edge_selection_loss = edge_selection_loss * self.config.loss_weights.edge_selection_loss
        weights_loss = weights_loss * self.config.loss_weights.weights_loss
        loss = (beta_cons * cons_kl_loss + beta_var * var_kl_loss + cons_loss +
                degree_loss + logits_loss + edge_selection_loss + weights_loss) / data.num_graphs

        tb = tb_writter.tb_writter
        if self.training:
            tb.add_histogram("Embeddings/h_vars", h_vars, tb_writter.step)
            tb.add_histogram("Embeddings/h_conss", h_conss, tb_writter.step)
            tb.add_histogram("Embeddings/z_vars", z_vars, tb_writter.step)
            tb.add_histogram("Embeddings/z_conss", z_conss, tb_writter.step)
            tb.add_histogram("Embeddings/p_vars", p_vars, tb_writter.step)
            tb.add_histogram("Embeddings/p_conss", p_conss, tb_writter.step)
        else:
            tb.add_histogram("Embeddings/h_vars_val", h_vars, tb_writter.step)
            tb.add_histogram("Embeddings/h_conss_val",
                             h_conss, tb_writter.step)
            tb.add_histogram("Embeddings/z_vars_val", z_vars, tb_writter.step)
            tb.add_histogram("Embeddings/z_conss_val", z_conss, tb_writter.step)
            tb.add_histogram("Embeddings/p_vars_val", p_vars, tb_writter.step)
            tb.add_histogram("Embeddings/p_conss_val", p_conss, tb_writter.step)

        return loss

    def sort_community(self, data, retain_num: int):
        """
        Sort the constraint node idx in each community to be substituted.
        """
        if isinstance(data, list):
            batched_data = Batch.from_data_list(data, follow_batch=["x_constraints", "x_variables"])
        else:
            batched_data = data
        orig_graph = self.embedding_layer.embed_graph(batched_data)
        z_conss, z_vars = self.encoder_layer_1.forward(orig_graph)
        community_rank = self.resample_layer.resample_and_rank(z_conss, batched_data.community_info,
                                                               batched_data.x_constraints_ptr, retain_num=retain_num)

        return community_rank

    def decode(self, graphs: List[BipartiteGraph], num_iters: int, community_idx: list, modified_num_constraints: int):
        for b, graph in enumerate(graphs):
            modified = 0
            for num in range(num_iters):
                if modified >= modified_num_constraints:
                    break
                modified += len(community_idx[b][num])

                batched_graph = Batch.from_data_list([graph], follow_batch=["x_constraints", "x_variables"])
                orig_graph = self.embedding_layer.embed_graph(batched_graph)

                # z_conss0, z_vars0 = self.encoder_layer_1.forward(orig_graph)

                # z_conss, z_vars, _, _ = self.resample_layer.forward(
                #     z_conss, z_vars)

                h_conss, h_vars = self.encoder_layer_2.forward(orig_graph)

                involved_edges_idx = []
                edge_idx_cpu = graph.edge_index[0].cpu()
                community_idx_cpu = community_idx[b][num].cpu()
                indicator_edge = torch.zeros(len(edge_idx_cpu), dtype=int)

                for i in range(len(community_idx_cpu)):
                    involved_edges_idx_i = torch.where(edge_idx_cpu == community_idx_cpu[i])[0]
                    for j in involved_edges_idx_i:
                        involved_edges_idx.append(j.item())
                        indicator_edge[j] = 1

                seen_edges_idx = torch.where(indicator_edge == 0)[0]

                connected_vars_idx = graph.edge_index[1][involved_edges_idx]

                z_conss = torch.randn(
                    (len(community_idx[b][num]), self.config.common.embd_size)).cuda()
                z_vars = torch.randn(
                    (len(connected_vars_idx), self.config.common.embd_size)).cuda()

                resampled_graph = orig_graph.clone()
                h_conss[community_idx[b][num]] = z_conss
                h_vars[connected_vars_idx] = z_vars
                resampled_graph.x_constraints = h_conss
                resampled_graph.x_variables = h_vars

                p_conss, p_vars = self.decoder_layer.forward(resampled_graph)

                conss = self.bias_predictor.decode(p_conss, community_idx[b][num])

                degree = self.degree_predictor.decode(p_conss, community_idx[b][num])
                sum_degree = torch.sum(degree, dim=-1).long()
                degree = degree.long()

                logits_indices = self.logits_predictor.decode(p_vars, sum_degree)

                p_edge_selection = []
                for c_idx in community_idx[b][num]:
                    p_cv = torch.cat(
                        (p_vars[logits_indices],
                         p_conss[c_idx].repeat(logits_indices.shape[0], 1)), dim=-1)
                    p_edge_selection.append(p_cv)

                p_edge_selection = torch.stack(p_edge_selection, dim=0)

                edge_selection = self.edge_selector.decode(p_edge_selection, degree, logits_indices)

                p_weight = []
                for i, c_idx in enumerate(community_idx[b][num]):
                    p_cv = torch.cat(
                        (p_vars[edge_selection[i]],
                         p_conss[c_idx].repeat(p_vars[edge_selection[i]].shape[0], 1)), dim=-1)
                    p_weight.append(p_cv)

                p_weight_ = torch.cat(p_weight, dim=0)

                weights = self.weights_predictor.decode(p_weight_)

                # update the graph
                weights_ptr = 0
                seen_edges = seen_edges_idx.cuda()
                cur_edge_index = graph.edge_index[:, seen_edges]
                cur_edge_attr = graph.edge_attr[seen_edges]
                for ii, masked_cons_idx in enumerate(community_idx[b][num]):
                    graph.x_constraints[masked_cons_idx][0] = conss[ii].view(-1, 1)

                    new_edge_index = torch.LongTensor(
                        [[masked_cons_idx, i] for i in edge_selection[ii]]).T.cuda()
                    new_edge_attr = weights[weights_ptr: weights_ptr + p_weight[ii].shape[0]]
                    weights_ptr += p_weight[ii].shape[0]
                    cur_edge_index = torch.cat(
                        (cur_edge_index, new_edge_index), dim=-1)
                    cur_edge_attr = torch.cat(
                        (cur_edge_attr, new_edge_attr), dim=0)

                graph.edge_index, graph.edge_attr = sort_edge_index(
                    cur_edge_index, cur_edge_attr)

                graphs[b] = graph

        results = []
        for graph in graphs:
            x_constraints = graph.x_constraints[:, 0].unsqueeze(1).detach().cpu().numpy()
            edge_index = graph.edge_index.detach().cpu().numpy()
            edge_attr = graph.edge_attr.detach().cpu().numpy()
            x_variables = graph.x_variables.detach().cpu().numpy()
            results.append([x_constraints, edge_index, edge_attr, x_variables])
        return results

    def update_labels(self, graph, r_graph, community_idx):
        """
        Update labels of the resampled graph.
        """
        # get the involved edges by detecting whether the edge index is in the community
        involved_edges_idx = []
        connected_edge_attr_dict = defaultdict(list)
        connected_vars = defaultdict(list)

        edge_idx_cpu = graph.edge_index[0].cpu()
        community_idx_cpu = community_idx.cpu()

        for i in range(len(community_idx_cpu)):
            involved_edges_idx_i = torch.where(edge_idx_cpu == community_idx_cpu[i])[0]
            for j in involved_edges_idx_i:
                involved_edges_idx.append(j)
                connected_edge_attr_dict[graph.edge_index[0][j].item()].append(graph.edge_attr[j].item())
                connected_vars[graph.edge_index[0][j].item()].append(graph.edge_index[1][j].item())

        connected_vars_idx = graph.edge_index[1, involved_edges_idx]
        logits_label = torch.zeros(graph.num_variables, dtype=int, device=r_graph.x_constraints.device)
        logits_label[connected_vars_idx] = 1
        # degree_label = torch.sum(logits_label)
        # compute degree_label for each node in community_idx
        degree_label = torch.zeros(len(community_idx), dtype=int, device=r_graph.x_constraints.device)
        for i, node_idx in enumerate(community_idx):
            degree_label[i] = torch.sum(graph.edge_index[0] == node_idx)

        # get logits_label for each node in community_idx
        involved_node_idx = torch.where(logits_label == 1)[0]
        batched_invloved_variable_idx = []
        batched_invloved_constraint_idx = []
        for i in range(len(graph.community_info)):
            batch_involved_variable_idx = involved_node_idx[involved_node_idx < graph.x_variables_ptr[i + 1]]
            batch_involved_variable_idx = batch_involved_variable_idx[batch_involved_variable_idx >= graph.x_variables_ptr[i]]
            batched_invloved_variable_idx.append(batch_involved_variable_idx)

            batch_invloved_constraint_idx = community_idx[community_idx < graph.x_constraints_ptr[i + 1]]
            batch_invloved_constraint_idx = batch_invloved_constraint_idx[batch_invloved_constraint_idx >= graph.x_constraints_ptr[i]]
            batched_invloved_constraint_idx.append(batch_invloved_constraint_idx)

        constraint_logits_dict = defaultdict(list)
        for edge_idx in involved_edges_idx:
            edge = graph.edge_index[:, edge_idx]
            constraint_logits_dict[edge[0].item()].append(edge[1])

        edge_selection_label = []
        for i, c_list in enumerate(batched_invloved_constraint_idx):
            v_dict = dict()
            for iii, v_idx in enumerate(batched_invloved_variable_idx[i]):
                v_dict[v_idx.item()] = iii
            for c_idx in c_list:
                label = torch.zeros(len(batched_invloved_variable_idx[i]), dtype=int, device=r_graph.x_constraints.device)
                logits = [v_dict[v_idx.item()] for v_idx in constraint_logits_dict[c_idx.item()]]
                label[logits] = 1
                edge_selection_label.append(label)

        edge_selection_label = torch.cat(edge_selection_label, dim=0)

        weight_label = []
        for i, c_list in enumerate(batched_invloved_constraint_idx):
            for c_idx in c_list:
                weight_label.append(torch.tensor(connected_edge_attr_dict[c_idx.item()],
                                                 device=r_graph.x_constraints.device))

        weight_label = torch.cat(weight_label, dim=0)

        device = r_graph.x_constraints.device
        r_graph.bias_label = graph.x_constraints[community_idx, 0].to(device)
        r_graph.degree_label = degree_label.to(device)
        r_graph.logits_label = logits_label.to(device)
        r_graph.edge_selection_label = edge_selection_label.to(device)
        r_graph.connected_vars_idx = connected_vars_idx.to(device)
        r_graph.weights_label = weight_label.to(device)
        r_graph.involved_variables = batched_invloved_variable_idx
        r_graph.involved_constraints = batched_invloved_constraint_idx
        r_graph.connected_vars = connected_vars

        return r_graph

    @staticmethod
    def load_model(config: DictConfig, load_model_path: str = None) -> "ACMMILP":
        """
        Loads the model.
        """
        data_stats = json.load(open(config.paths.dataset_stats_path, 'r'))
        model = ACMMILP(config.model, data_stats).cuda()

        if load_model_path:
            load_ckpt = torch.load(load_model_path)
            model.load_state_dict(load_ckpt, strict=False)

        return model