import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import src.tb_writter as tb_writter
from sklearn.metrics import roc_auc_score
from src.utils import downsample
from torch import Tensor, LongTensor


class ReSample(nn.Module):
    def __init__(self, config) -> None:
        """
        The resample layer in VAE.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embd_size = config.embd_size

        self.z_conss_mean = nn.Sequential(
            nn.Linear(self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.embd_size),
        )
        self.z_vars_mean = nn.Sequential(
            nn.Linear(self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.embd_size),
        )
      
        self.z_conss_logstd = nn.Sequential(
            nn.Linear(self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.embd_size),
        )
        self.z_vars_logstd = nn.Sequential(
            nn.Linear(self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.embd_size),
        )
    
    def forward(self,
            z_conss: Tensor,
            z_vars: Tensor
        ):
        """
        Input:
            z_conss: (num_cons, embd_size), constraint embeddings
            z_vars: (num_vars, embd_size), variable embeddings

        Output:
            z_conss: (num_cons, embd_size), 
            z_vars: (num_vars, embd_size)
            conss_kl_loss: scalar
            vars_kl_loss: scalar
        """
        z_conss_mean = self.z_conss_mean(z_conss)
        z_conss_logstd = torch.clamp_max(self.z_conss_logstd(z_conss), max=10)
        conss_kl_loss = - 0.5 * torch.sum(1.0 + z_conss_logstd - z_conss_mean * z_conss_mean - torch.exp(z_conss_logstd))
        if self.training:
            conss_epsilon = torch.randn_like(z_conss_mean, device=z_conss.device)
            z_conss = z_conss_mean + torch.exp(z_conss_logstd / 2) * conss_epsilon
        else:
            z_conss = z_conss_mean

        z_vars_mean = self.z_vars_mean(z_vars)
        z_vars_logstd = torch.clamp_max(self.z_vars_logstd(z_vars), max=10)
        vars_kl_loss = - 0.5 * torch.sum(1.0 + z_vars_logstd - z_vars_mean * z_vars_mean - torch.exp(z_vars_logstd))
        if self.training:
            vars_epsilon = torch.randn_like(z_vars_mean, device=z_vars.device)
            z_vars = z_vars_mean + torch.exp(z_vars_logstd / 2) * vars_epsilon
        else:
            z_vars = z_vars_mean

        tb = tb_writter.tb_writter
        if self.training:
            tb.add_scalar("resample/conss_kl_loss", conss_kl_loss, tb_writter.step)
            tb.add_scalar("resample/vars_kl_loss", vars_kl_loss, tb_writter.step)
        else:
            tb.add_scalar("resample/conss_kl_loss_val", conss_kl_loss, tb_writter.step)
            tb.add_scalar("resample/vars_kl_loss_val", vars_kl_loss, tb_writter.step)

        return z_conss, z_vars, conss_kl_loss, vars_kl_loss

    def resample_and_rank(self, z_conss: Tensor, community: list, ptrs: Tensor,
                          instance_dependent: bool = True,
                          retain_num: int = 10):
        z_conss_mean = self.z_conss_mean(z_conss)
        z_conss_logstd = torch.clamp_max(self.z_conss_logstd(z_conss), max=10)

        # compute the mean of z_conss_mean w.r.t. ptr
        batch_mean_list = []
        for b in range(len(ptrs) - 1):
            batch_mean_list.append(torch.mean(z_conss_mean[ptrs[b]:ptrs[b+1]], dim=0))

        # compute sampling probability of each z_conss_mean
        # when taking a gaussian distribution with mean of batch_mean_list[b] and std of 1
        batch_prob_list = []
        for b in range(len(ptrs) - 1):
            if instance_dependent:
                batch_prob_list.append(
                    torch.exp(-0.5 * torch.mean((z_conss_mean[ptrs[b]:ptrs[b+1]] - batch_mean_list[b]) ** 2, dim=1)))
            else:
                batch_prob_list.append(torch.exp(-0.5 * torch.mean((z_conss_mean[ptrs[b]:ptrs[b + 1]]) ** 2, dim=1)))

        community_prob_list = []
        for b in range(len(ptrs) - 1):
            community_prob_list.append(torch.zeros(len(community[b]), device=z_conss.device))
            for i in range(len(community[b])):
                community_prob_list[b][i] = torch.mean(batch_prob_list[b][community[b][i]])

        # rank the z_conss_mean in each community
        community_rank_list = []
        for b in range(len(ptrs) - 1):
            community_rank_list.append(torch.argsort(community_prob_list[b], descending=False))

        # construct a list of node index in community w.r.t. community_rank in a batched form
        batch_rank_list = []
        retain_num = min(min([community_rank_list[b].shape[0] for b in range(len(ptrs) - 1)]), retain_num)
        for i in range(retain_num):
            rank_list = []
            for b in range(len(ptrs) - 1):
                community_idx = community_rank_list[b][i]
                node_idx = community[b][community_idx]
                rank_list.extend(node_idx)
            batch_rank_list.append(LongTensor(rank_list).to(z_conss.device))

        return batch_rank_list


class Bias_Predictor(nn.Module):
    def __init__(self, config, data_stats):
        """
        Bias predictor.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embd_size = config.embd_size

        self.rhs_type = data_stats["rhs_type"]

        self.rhs_min = int(data_stats["rhs_min"])
        self.rhs_max = int(data_stats["rhs_max"])

        self.cons_predictor = nn.Sequential(
            nn.Linear(self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )

        self.cons_loss = MSELoss(reduction="sum")
    
    def convert(self, cons: torch.Tensor):
        if self.rhs_type == "int":
            return torch.round(cons * (self.rhs_max - self.rhs_min) + self.rhs_min)
        elif self.rhs_type == "float":
            return cons * (self.rhs_max - self.rhs_min) + self.rhs_min

    def forward(self,
            p_conss: torch.Tensor,
            masked_cons_idx: torch.LongTensor,
            cons_label: torch.Tensor
        ):
        inputs = p_conss[masked_cons_idx]
        cons_pred = self.cons_predictor(inputs).view(-1)
        if abs(self.rhs_max - self.rhs_min) < 1e-6:
            cons_loss = torch.tensor(0.0, device=inputs.device)
        else:
            cons_label_ = (cons_label.view(-1) - self.rhs_min) / (self.rhs_max - self.rhs_min)
            cons_loss = self.cons_loss(cons_pred, cons_label_)

        tb = tb_writter.tb_writter
        if self.training:
            tb.add_scalar("Constraint_predictor/cons_loss", cons_loss, tb_writter.step)
            tb.add_histogram("Constraint_predictor/cons", cons_label, tb_writter.step)
            tb.add_histogram("Constraint_predictor/cons_pred", self.convert(cons_pred), tb_writter.step)
        else:
            tb.add_scalar("Constraint_predictor/cons_loss_val", cons_loss, tb_writter.step)
            tb.add_histogram("Constraint_predictor/cons_val", cons_label, tb_writter.step)
            tb.add_histogram("Constraint_predictor/cons_pred_val", self.convert(cons_pred), tb_writter.step)

        return cons_loss, cons_pred

    def decode(self, p_conss: torch.Tensor, masked_cons_idx: torch.LongTensor):
        inputs = p_conss[masked_cons_idx]
        cons_pred = self.cons_predictor(inputs).view(-1)
        return self.convert(cons_pred)


class Degree_Predictor(nn.Module):

    def __init__(self, config, data_stats):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embd_size = config.embd_size

        self.degree_min = data_stats["cons_degree_min"]
        self.degree_max = data_stats["cons_degree_max"]

        self.degree_predictor = nn.Sequential(
            nn.Linear(self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        self.degree_loss = MSELoss(reduction="sum")
        
    def convert(self, degree: torch.Tensor):
        return torch.round(degree * (self.degree_max - self.degree_min) + self.degree_min)
    
    def forward(self, p_conss: torch.Tensor, masked_cons_idx: torch.LongTensor, degree_label: torch.Tensor):
        inputs = p_conss[masked_cons_idx]
        degree_pred = self.degree_predictor(inputs).view(-1)
        degree_label_ = (degree_label - self.degree_min) / (self.degree_max - self.degree_min)
        degree_loss = self.degree_loss(degree_pred, degree_label_)

        tb = tb_writter.tb_writter
        if self.training:
            tb.add_scalar("Degree_predictor/degree_loss", degree_loss, tb_writter.step)
            tb.add_histogram("Degree_predictor/degree", degree_label, tb_writter.step)
            tb.add_histogram("Degree_predictor/degree_pred", self.convert(degree_pred), tb_writter.step)
            tb.add_histogram("Embeddings/h_masked_cons", p_conss[masked_cons_idx], tb_writter.step)
        else:
            tb.add_scalar("Degree_predictor/degree_loss_val", degree_loss, tb_writter.step)
            tb.add_histogram("Degree_predictor/degree_val", degree_label, tb_writter.step)
            tb.add_histogram("Degree_predictor/degree_pred_val", self.convert(degree_pred), tb_writter.step)
            tb.add_histogram("Embeddings/h_masked_cons_val", p_conss[masked_cons_idx], tb_writter.step)

        return degree_loss, degree_pred
    
    def decode(self, p_conss: torch.Tensor, masked_cons_idx: torch.LongTensor):
        inputs = p_conss[masked_cons_idx]
        degree_pred = self.degree_predictor(inputs).view(-1)
        return self.convert(degree_pred)


class Logits_Predictor(nn.Module):

    def __init__(self, config, data_stats):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embd_size = config.embd_size

        self.logits_predictor = nn.Sequential(
            nn.Linear(self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

        self.logits_loss = BCEWithLogitsLoss(reduction='sum')

    def forward(self, p_vars: torch.Tensor, logits_label: torch.LongTensor):
       
        logits_input = p_vars
        logits_input, logits_label = downsample(logits_input, logits_label)
        logits_pred = self.logits_predictor(logits_input).view(-1)
        logits_loss = self.logits_loss(logits_pred, logits_label.float())

        tb = tb_writter.tb_writter
        if self.training:
            tb.add_scalar("Logits_predictor/logits_loss", logits_loss, tb_writter.step)

            logits_label = logits_label.cpu().numpy()
            logits_pred = logits_pred.detach().cpu().numpy()
            auc = roc_auc_score(logits_label, logits_pred)
            tb.add_scalar("Logits_predictor/logits_auc", auc, tb_writter.step)
            logits_pred = (logits_pred > 0)
            tb.add_histogram("Logits_predictor/logits", logits_label, tb_writter.step)
            tb.add_histogram("Logits_predictor/logits_pred", logits_pred, tb_writter.step)

        else:
            tb.add_scalar("Logits_predictor/logits_loss_val", logits_loss, tb_writter.step)

            logits_label = logits_label.cpu().numpy()
            logits_pred = logits_pred.detach().cpu().numpy()
            auc = roc_auc_score(logits_label, logits_pred)
            tb.add_scalar("Logits_predictor/logits_auc_val", auc, tb_writter.step)
            logits_pred = (logits_pred > 0)
            tb.add_histogram("Logits_predictor/logits_val", logits_label, tb_writter.step)
            tb.add_histogram("Logits_predictor/logits_pred_val", logits_pred, tb_writter.step)

        return logits_loss, logits_pred
        
    
    def decode(self, p_vars: torch.Tensor, sum_degree: torch.LongTensor):
        logits_input = p_vars
        logits_pred = self.logits_predictor(logits_input).view(-1)

        degree_ = torch.minimum(sum_degree, torch.tensor(logits_pred.shape[0]))
        _, indices = torch.topk(logits_pred, int(degree_.item()))
            
        return indices


class Edge_Selector(nn.Module):

    def __init__(self, config, data_stats):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embd_size = config.embd_size

        self.edge_selector = nn.Sequential(
            nn.Linear(2 * self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

        self.edge_selector_loss = BCEWithLogitsLoss(reduction='sum')

    def forward(self, p_vars: torch.Tensor, edge_label: torch.LongTensor):

        edge_input = p_vars
        edge_input, edge_label = downsample(edge_input, edge_label)
        edge_pred = self.edge_selector(edge_input).view(-1)
        edge_selector_loss = self.edge_selector_loss(edge_pred, edge_label.float())

        tb = tb_writter.tb_writter
        if self.training:
            tb.add_scalar("Edge_selector/edge_selector_loss", edge_selector_loss, tb_writter.step)

            edge_label = edge_label.cpu().numpy()
            edge_pred = edge_pred.detach().cpu().numpy()
            edge_pred = (edge_pred > 0)
            tb.add_histogram("Edge_selector/edge_label", edge_label, tb_writter.step)
            tb.add_histogram("Edge_selector/edge_pred", edge_pred, tb_writter.step)

        else:
            tb.add_scalar("Edge_selector/edge_selector_loss_val", edge_selector_loss, tb_writter.step)

            edge_label = edge_label.cpu().numpy()
            edge_pred = edge_pred.detach().cpu().numpy()
            edge_pred = (edge_pred > 0)
            tb.add_histogram("Edge_selector/edge_label_val", edge_label, tb_writter.step)
            tb.add_histogram("Edge_selector/edge_pred_val", edge_pred, tb_writter.step)

        return edge_selector_loss, edge_pred

    def decode(self, p_vars: torch.Tensor, degree: torch.LongTensor, logits_indices: torch.Tensor):
        logits_input = p_vars
        logits_pred = self.edge_selector(logits_input)

        node_logits = []
        tuple_logits = []
        for i in range(len(degree)):
            degree_ = torch.minimum(degree[i], torch.tensor(logits_pred[i].shape[0]))
            # compute softmax of logits_pred[i]
            logits_p = torch.softmax(logits_pred[i], dim=0).view(-1)
            # sample degree_ indices from logits_p
            indices = torch.multinomial(logits_p, int(degree_.item()), replacement=False)

            # _, indices = torch.topk(logits_pred[i].view(-1), int(degree_.item()))
            indices = indices.sort()[0]
            i_tuple = tuple(indices)
            num_while = 0
            while i_tuple in tuple_logits and num_while < 100 * degree_.item():
                num_while += 1
                indices = torch.multinomial(logits_p, int(degree_.item()), replacement=False)
                i_tuple = tuple(indices)
            node_logits.append(logits_indices[indices])
            tuple_logits.append(i_tuple)

        return node_logits


class Weights_Predictor(nn.Module):

    def __init__(self, config, data_stats):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.embd_size = config.embd_size

        self.weights_type = data_stats["coef_type"]
        self.weights_min = int(data_stats["coef_min"])
        self.weights_max = int(data_stats["coef_max"])

        self.weights_predictor = nn.Sequential(
            nn.Linear(2 * self.embd_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        )

        self.weights_loss = MSELoss(reduction="sum")

    def convert(self, weights: torch.Tensor):
        if self.weights_type == "int":
            return torch.round(weights * (self.weights_max - self.weights_min) + self.weights_min)
        else:
            return weights * (self.weights_max - self.weights_min) + self.weights_min
        
    def forward(self, p_vars, weights_label):
        weights_input = p_vars
        weights_pred = self.weights_predictor(weights_input).view(-1)
        if abs(self.weights_max - self.weights_min) < 1e-6:
            weights_loss = torch.tensor(0.0).to(weights_pred.device)
        else:
            weights_label_ = (weights_label.view(-1) - self.weights_min) / (self.weights_max - self.weights_min)
            weights_loss = self.weights_loss(weights_pred, weights_label_)

        tb = tb_writter.tb_writter
        if self.training:
            tb.add_scalar("Weights_predictor/weights_loss", weights_loss, tb_writter.step)
            tb.add_histogram("Weights_predictor/weights", weights_label, tb_writter.step)
            tb.add_histogram("Weights_predictor/weights_pred", self.convert(weights_pred), tb_writter.step)
        else:
            tb.add_scalar("Weights_predictor/weights_loss_val", weights_loss, tb_writter.step)
            tb.add_histogram("Weights_predictor/weights_val", weights_label, tb_writter.step)
            tb.add_histogram("Weights_predictor/weights_pred_val", self.convert(weights_pred), tb_writter.step)
        return weights_loss, weights_pred

    def decode(self, p_vars):
        weight_input = p_vars
        weight_pred = self.weights_predictor(weight_input)

        return self.convert(weight_pred)
