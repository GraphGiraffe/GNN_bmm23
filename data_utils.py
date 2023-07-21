import os
import tqdm

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch
from torch_geometric.data import Data
from torch_geometric.utils import (
    remove_self_loops,
    to_edge_index,
    to_torch_csr_tensor,
)

from utils import generate_edges_dir


def calc_norm_params(data_dir, file_list, num_fields=4,
                     geom_dim=2):  # norm parameters for flow fields, coordinates and boundary cond. in nodes

    min_value = torch.zeros(num_fields)
    max_value = torch.zeros(num_fields)

    domain_boundaries_min = torch.zeros(geom_dim)
    domain_boundaries_max = torch.zeros(geom_dim)

    for file_name in file_list:
        nodes = torch.tensor(pd.read_csv(os.path.join(data_dir, 'nodes', file_name), header=None).values)
        flow = torch.tensor(pd.read_csv(os.path.join(data_dir, 'flow', file_name), header=None).values)
        xy_min, _ = torch.min(nodes[:, :2], 0)
        xy_max, _ = torch.max(nodes[:, :2], 0)
        uvpt_min, _ = torch.min(flow, 0)
        uvpt_max, _ = torch.max(flow, 0)

        # print(xy_min)
        domain_boundaries_min = torch.minimum(domain_boundaries_min, xy_min)
        domain_boundaries_max = torch.maximum(domain_boundaries_max, xy_max)
        min_value = torch.minimum(min_value, uvpt_min)
        max_value = torch.maximum(max_value, uvpt_max)

    return domain_boundaries_min, domain_boundaries_max, min_value, max_value


def calc_norm_params_avstd(data_dir, file_list, num_fields=3,
                           geom_dim=2):  # norm parameters for flow fields, coordinates and boundary cond. in nodes

    av_val = torch.tensor([])
    std_val = torch.tensor([])

    domain_boundaries_min = torch.zeros(geom_dim)
    domain_boundaries_max = torch.zeros(geom_dim)

    for file_name in file_list:
        nodes = torch.tensor(pd.read_csv(os.path.join(data_dir, 'nodes', file_name), header=None).values)
        flow = torch.tensor(pd.read_csv(os.path.join(data_dir, 'flow', file_name), header=None).values)
        xy_min, _ = torch.min(nodes[:, :2], 0)
        xy_max, _ = torch.max(nodes[:, :2], 0)
        uvp_av = torch.mean(flow, 0)
        uvp_std = torch.std(flow, 0)

        # print(xy_min)
        domain_boundaries_min = torch.minimum(domain_boundaries_min, xy_min)
        domain_boundaries_max = torch.maximum(domain_boundaries_max, xy_max)
        av_val = torch.cat([av_val, uvp_av[None, :]], 0)
        std_val = torch.cat([std_val, uvp_std[None, :]], 0)
    av_val = torch.mean(av_val, 0)
    std_val = torch.mean(std_val, 0)

    return domain_boundaries_min, domain_boundaries_max, av_val, std_val


def get_xyz_and_uvp_mins(norm_par_fp):
    import re
    mask = '.*\[(.*),(.*)\]'
    pattern = re.compile(mask)

    lines = list()
    with open(norm_par_fp, 'r') as f:
        for idx, line in enumerate(f):
            if idx != 0:
                lines.append(line)
    lines = lines[:-1]

    xy_mins, xy_maxs = list(), list()
    for line in lines[:2]:
        res = pattern.match(line)
        xy_mins.append(float(res.group(1)))
        xy_maxs.append(float(res.group(2)))

    uvp_mins, uvp_maxs = list(), list()
    for line in lines[2:]:
        res = pattern.match(line)
        uvp_mins.append(float(res.group(1)))
        uvp_maxs.append(float(res.group(2)))
    return np.array(xy_mins), np.array(xy_maxs), np.array(uvp_mins), np.array(uvp_maxs)


def save_norm_params(save_path, geom_in_dim, out_dim, domain_boundaries_min, domain_boundaries_max, min_value,
                     max_value):
    with open(save_path, 'w') as f:
        out_str = f'The [min, max] values for:\n'
        out_str += f'\tx-coordinates: [{domain_boundaries_min[0]},{domain_boundaries_max[0]}]\n'
        out_str += f'\ty-coordinates: [{domain_boundaries_min[1]},{domain_boundaries_max[1]}]\n'

        if geom_in_dim == 3:
            out_str += f'\tz-coordinates: [{domain_boundaries_min[2]},{domain_boundaries_max[2]}]\n'
        out_str += f'\tu            : [{min_value[0]},{max_value[0]}]\n'
        out_str += f'\tv            : [{min_value[1]},{max_value[1]}]\n'
        out_str += f'\tp            : [{min_value[2]},{max_value[2]}]\n'
        if out_dim > 3:
            out_str += f'\tt            : [{min_value[3]},{max_value[3]}]\n'
        print(out_str, file=f)


def make_dist_map(nodes):
    buf_arr = nodes[:, 3:].numpy()
    buf_arr = np.equal(buf_arr, -5)
    row_sum = np.sum(buf_arr, axis=1)
    bc_indx = np.not_equal(row_sum, 4)[:, None]  # True means boundary point
    upd_nodes = np.concatenate([nodes, bc_indx], axis=1)
    bc_points_xy = upd_nodes[upd_nodes[:, -1] == 1][:, :2]
    nodes_dist = np.zeros(upd_nodes.shape[0])

    for node_idx in range(upd_nodes.shape[0]):

        if upd_nodes[node_idx][7] == 1:
            nodes_dist[node_idx] = 0

        else:
            point_xy = upd_nodes[node_idx][:2]
            min_dist = np.min(cdist(point_xy[None, :], bc_points_xy, 'chebyshev'))
            nodes_dist[node_idx] = min_dist

    upd_nodes = np.concatenate([upd_nodes, nodes_dist[:, None]], axis=1)
    return torch.tensor(upd_nodes)


def add_2hop_edges(edge_index, N):
    k_hop_edges_index = []
    k_hop_edges_index.append(edge_index)
    adj = to_torch_csr_tensor(edge_index, size=(N, N))
    k_hop_edges, _ = to_edge_index(torch.matrix_power(adj, 2))
    k_hop_edges, _ = remove_self_loops(k_hop_edges)

    return k_hop_edges


def read_dataset(data_dir, num_files, dataset_name=None):
    if dataset_name is None:
        dataset_name = os.path.split((data_dir)[1] + '.pt')
    data_list = torch.load(os.path.join(data_dir, dataset_name))[:num_files]
    print(len(data_list))  # снеси потом
    return data_list


# TODO: fix "flownorm1, flownorm2"
def make_dataset(data_dir, num_files, dataset_name=None, data_source=None, with_bc=False,
                 norm_coord=False, norm_flow=False, bc_in_nodes_norm=False, save=False,
                 num_fields=4, avstd=True, add_dist_func=False, hop2=False, nodes_dim=None):
    data_list = []

    if data_source is not None:
        # print(dataset_source)
        with open(data_source, 'r') as f:
            # print(num_files)
            file_list = f.read().splitlines()
            file_list = file_list[:num_files]
    else:
        file_list = os.listdir(os.path.join(data_dir, 'nodes'))[:num_files]

    if 'edges' not in os.listdir(data_dir):
        generate_edges_dir(data_dir)

    if os.path.join(data_dir, dataset_name, '_norm_prms.txt') in os.listdir(os.path.join(data_dir)):
        dataset_name_n, _ = os.path.splitext(dataset_name)
        norm_params_pth = os.path.join(data_dir, dataset_name_n + '_norm_prms.txt')
        dom_bound_min, dom_bound_max, flownorm1, flownorm2 = get_xyz_and_uvp_mins(norm_params_pth)

    else:

        if avstd:
            dom_bound_min, dom_bound_max, flownorm1, flownorm2 = calc_norm_params_avstd(data_dir, file_list, num_fields)
            print(f'flownorm1: {flownorm1}')
            print(f'flownorm2: {flownorm2}')

        else:
            dom_bound_min, dom_bound_max, flownorm1, flownorm2 = calc_norm_params(data_dir, file_list, num_fields)
        dataset_name_n, _ = os.path.splitext(dataset_name)
        save_path = os.path.join(data_dir, dataset_name_n + '_norm_prms.txt')
        save_norm_params(save_path, 2, num_fields, dom_bound_min, dom_bound_max, flownorm1, flownorm2)

    range_xy = torch.sub(dom_bound_max, dom_bound_min)
    range_flow = torch.sub(flownorm2, flownorm1)
    if avstd:
        range_flow = flownorm2

    print(f'range_flow: {range_flow}')
    hop2_nodes = None
    for file in tqdm.tqdm(file_list):

        # nodes = torch.tensor(pd.read_csv(os.path.join(data_dir,'nodes',file), header=None).values).to(torch.float32)
        nodes = pd.read_csv(os.path.join(data_dir, 'nodes', file), header=None).replace(-50, -5)
        nodes = nodes.iloc[:, :nodes_dim]
        nodes = torch.tensor(nodes.values).to(torch.float32)

        if add_dist_func:
            nodes = make_dist_map(nodes)

        edges = torch.tensor(pd.read_csv(os.path.join(data_dir, 'edges', file), header=None).values)  # .to(torch.long)
        edges_transp = torch.transpose(edges, 0, 1)

        if hop2:
            hop2_nodes = add_2hop_edges(edges_transp, nodes.shape[0])

        flow = torch.tensor(pd.read_csv(os.path.join(data_dir, 'flow', file), header=None).values).to(torch.float32)
        elements = torch.tensor(pd.read_csv(os.path.join(data_dir, 'elements', file), header=None).values)
        edge_feats = torch.mean(nodes[edges], 1)

        bc = None
        if with_bc:
            bc = torch.tensor((pd.read_csv(os.path.join(data_dir, 'sv', file), header=None)).values).to(
                torch.float32).swapaxes(0, 1).reshape(1, 1, 100)
            # bc = bc.repeat(nodes.shape[0], 1)
            # bc = torch.FloatTensor(bc).reshape(380164, 1, -1)

        if norm_coord:
            xy_norm = torch.div(torch.sub(nodes[:, :2], dom_bound_min), range_xy)

            if bc_in_nodes_norm:

                bc_in_nodes = torch.div(torch.sub(nodes[:, 3:7], flownorm1), range_flow).to(torch.float32)

                nodes = torch.cat([xy_norm, nodes[:, 2].reshape(-1, 1), bc_in_nodes, nodes[:, 7:]], 1).to(torch.float32)
            else:
                nodes = torch.cat([xy_norm], 1).to(torch.float32)

            edge_feats = torch.mean(nodes[edges], 1)

        if norm_flow:
            flow = torch.div(torch.sub(flow, flownorm1), range_flow)

        data_list.append(
            Data(x=nodes, edge_index=edges_transp, edge_attr=edge_feats, flow=flow, bc=bc, hop2_nodes=hop2_nodes,
                 cells=elements))

    if save:
        print(f'Saving : {dataset_name}')
        print(f'Data: {data_list[0]}')
        torch.save(data_list, os.path.join(data_dir, dataset_name))

    return data_list
