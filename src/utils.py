import os
import random

import community.community_louvain as community
import ecole
import gurobipy as gp
import networkx as nx
import numpy as np
import pyscipopt as scip
import torch
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.data import BipartiteGraph


COEF_TYPES = {
    'debug': 'int',
    'setcover': 'int',
    'mis': 'int',
}

RHS_TYPES = {
    'debug': 'int',
    'setcover': 'int',
    'mis': 'int',
}

VAR_FEATURES = ["objective", "is_type_binary", "is_type_integer", "is_type_implicit_integer", "is_type_continuous",
                "has_lower_bound", "has_upper_bound", "lower_bound", "upper_bound"]

VAR_TYPE_FEATURES = ["B", "I", "M", "C"]


def instance2graph(path: str, compute_features: bool = False, comm_detec: bool = True, resolution: float = 1):
    """
    Extract the bipartite graph from the instance file.
    compute_features: whether to compute the features of the instance
    """
    model = ecole.scip.Model.from_file(path)
    obs = ecole.observation.MilpBipartite().extract(model, True)
    constraint_features = obs.constraint_features
    edge_indices = np.array(obs.edge_features.indices, dtype=int)
    edge_features = obs.edge_features.values.reshape((-1, 1))
    variable_features = obs.variable_features
    graph = [constraint_features, edge_indices, edge_features, variable_features]
    if not compute_features:
        raise NotImplementedError
        # return graph, None, None
    else:
        n_conss = len(constraint_features)
        n_vars = len(variable_features)
        n_cont_vars = np.sum(variable_features, axis=0)[1]

        lhs = coo_matrix((edge_features.reshape(-1), edge_indices), shape=(n_conss, n_vars)).toarray()
        rhs = constraint_features.flatten()
        obj = variable_features[:, 0].flatten()

        nonzeros = (lhs != 0)
        n_nonzeros = np.sum(nonzeros)
        lhs_coefs = lhs[np.where(nonzeros)]
        var_degree, cons_degree = nonzeros.sum(axis=0), nonzeros.sum(axis=1)

        nx_edge_indices = edge_indices.copy()
        nx_edge_indices[1] += edge_indices[0].max() + 1

        pyg_graph = Data(
            x_s=constraint_features,
            x_t=variable_features,
            edge_index=torch.LongTensor(nx_edge_indices),
            node_attribute="bipartite",
            num_nodes=len(constraint_features) + len(variable_features)
        )

        nx_graph = to_networkx(pyg_graph, to_undirected=True)

        if comm_detec:
            community_partition = community.best_partition(nx_graph, resolution=resolution)

            result_dict = {}

            for key, value in community_partition.items():
                if key > edge_indices[0].max():
                    break
                if value in result_dict:
                    result_dict[value].append(key)
                else:
                    result_dict[value] = [key]

            lll = len(list(result_dict.keys()))
            v_l = 0
            min_vl = 10000
            max_vl = -1
            for key, value in result_dict.items():
                v_l += len(value) / lll
                if len(value) > max_vl:
                    max_vl = len(value)
                if len(value) < min_vl:
                    min_vl = len(value)
            print("community_num, avg_community_size, max_community_size, min_community_size:", lll, v_l, max_vl, min_vl)
            print("-" * 30)

            communities = [result_dict[k] for k in result_dict.keys()]

        features = {
            "instance": path,

            "n_conss": n_conss,
            "n_vars": n_vars,
            "n_cont_vars": n_cont_vars,
            "ratio_cont_vars": float(n_cont_vars / n_vars),

            "n_nonzeros": n_nonzeros,
            "coef_dens": float(len(edge_features) / (n_vars * n_conss)),

            "var_degree_mean": float(var_degree.mean()),
            "var_degree_std": float(var_degree.std()),
            "var_degree_min": float(var_degree.min()),
            "var_degree_max": float(var_degree.max()),

            "cons_degree_mean": float(cons_degree.mean()),
            "cons_degree_std": float(cons_degree.std()),
            "cons_degree_min": int(cons_degree.min()),
            "cons_degree_max": int(cons_degree.max()),

            "lhs_mean": float(lhs_coefs.mean()),
            "lhs_std": float(lhs_coefs.std()),
            "lhs_min": float(lhs_coefs.min()),
            "lhs_max": float(lhs_coefs.max()),

            "rhs_mean": float(rhs.mean()),
            "rhs_std": float(rhs.std()),
            "rhs_min": float(rhs.min()),
            "rhs_max": float(rhs.max()),

            "obj_mean": float(obj.mean()),
            "obj_std": float(obj.std()),
            "obj_min": float(obj.min()),
            "obj_max": float(obj.max()),

            "clustering": float(nx.average_clustering(nx_graph)),
            "modularity": float(community.modularity(community.best_partition(nx_graph), nx_graph)),
        }

        if comm_detec:
            return graph, features, communities
        else:
            return graph, features


def solve_instance(path: str):
    """
    Solve the instance using Gurobi.
    """
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.setParam("TimeLimit", 60)
    env.setParam("Threads", 1)
    env.start()
    model = gp.read(path, env=env)
    model.optimize()

    results = {
        "status": model.status,
        "obj": model.objVal,
        "num_nodes": model.NodeCount,
        "num_sols": model.SolCount,
        "solving_time": model.Runtime,
    }

    return results


def model_add_var(model: scip.Model, var_idx: int, var_features: np.ndarray):
    """
    Add a variable to the SCIP model.
    model: the SCIP model.
    var_idx: the index of the added variable.
    var_features: the features of the variable. Form: [objective, is_type_binary, is_type_integer, is_type_implicit_integer, is_type_continuous, has_lower_bound, has_upper_bound, lower_bound, upper_bound].
    """
    return model.addVar(
        name=f"x{var_idx}",
        obj=var_features[0],
        vtype=VAR_TYPE_FEATURES[np.argmax(var_features[1:5])],
        lb=var_features[7] if bool(var_features[5]) else None,
        ub=var_features[8] if bool(var_features[6]) else None,
    )


def graph2instance(graph):
    [constraint_features, edge_indices, edge_features, variable_features] = graph

    model = scip.Model()
    model.setIntParam('display/verblevel', 0)

    vars = [model_add_var(model, i, var_features) for i, var_features in enumerate(variable_features)]
    for cons_idx, cons in enumerate(constraint_features):
        edge_idx = np.where(edge_indices[0] == cons_idx)[0]
        cons_expr = sum([edge_features[i, 0] * vars[edge_indices[1, i]] for i in edge_idx])
        cons_expr.normalize()
        model.addCons(cons_expr <= cons[0])

    return model


def downsample(X: torch.Tensor, y: torch.LongTensor):
    """
    Downsample the input data and target data so that the negative samples are balanced with the positive samples.
    """
    pos_idx = torch.where(y == 1)[0]
    neg_idx = torch.where(y == 0)[0]
    neg_idx = neg_idx[torch.randperm(neg_idx.shape[0])[:pos_idx.shape[0]]]
    idx = torch.cat([pos_idx, neg_idx])
    idx = idx[torch.randperm(idx.shape[0])]
    return X[idx], y[idx]


def set_seed(seed):
    """
    Set the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Set CuDNN to be deterministic. Notice that this may slow down the training.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def set_cpu_num(cpu_num):
    """
    Set the number of used cpu kernals.
    """
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
