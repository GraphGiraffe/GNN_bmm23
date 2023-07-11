import os
import numpy as np
import pandas as pd
import torch


def generate_edges_dir(data_dir):
    elem_dir = os.path.join(data_dir, 'elements')

    if 'edges' not in os.listdir(data_dir):
        os.mkdir(os.path.join(data_dir, 'edges'))

    for file in sorted(os.listdir(elem_dir)):
        file_path = os.path.join(data_dir, 'elements', file)
        elements = pd.read_csv(file_path, header=None).to_numpy()
        edges = np.unique(np.vstack([elements[:, :2], elements[:, 1:], elements[:, [0, 2]]]), axis=0)
        receivers = np.amin(edges, axis=1).reshape(-1, 1)
        senders = np.amax(edges, axis=1).reshape(-1, 1)

        edges = np.hstack([senders, receivers])
        # print(edges.shape)
        edges = np.unique(edges, axis=0)
        edges_inv = edges[:, [1, 0]]
        edges = np.vstack([edges, edges_inv])
        np.savetxt(os.path.join(data_dir, 'edges', file), edges, delimiter=',', fmt="%i")


def lift_drag(data, flow, viscos):
    nodes = data.x[:, :3].numpy()
    edges = torch.transpose(data.edge_index, 0, 1).numpy()
    elements = data.cells.numpy()
    obstacle = np.where(nodes[:, 2] == 0)[0]
    coord_obstacle = nodes[obstacle, :2]
    min_index = np.argmin(coord_obstacle[:, 0])
    node_left = obstacle[min_index]

    edges_obstacle = np.asarray([edges[i, 0] in obstacle for i in range(edges.shape[0])]) * np.asarray(
        [edges[i, 1] in obstacle for i in range(edges.shape[0])])
    print(edges_obstacle)
    edges_obstacle = edges[edges_obstacle, :]
    print(edges_obstacle)

    sorted_nodes = [node_left]
    print(sorted_nodes)
    while len(sorted_nodes) < edges_obstacle.shape[0]:
        three_nodes = np.unique(edges_obstacle[np.logical_or(edges_obstacle[:, 0] == sorted_nodes[-1],
                                                             edges_obstacle[:, 1] == sorted_nodes[-1])])
        print(three_nodes)
        coord_three_nodes = [nodes[i, :2] for i in three_nodes]
        neighbours = np.setdiff1d(three_nodes, sorted_nodes)
        print(neighbours)
        if len(neighbours) == 1:
            sorted_nodes.append(neighbours[0])
        else:
            k0 = (nodes[neighbours[0], 1] - nodes[sorted_nodes[-1], 1])
            k1 = (nodes[neighbours[1], 1] - nodes[sorted_nodes[-1], 1])
            unclockwise = np.asarray(neighbours)[~np.asarray([k0 > 0, k1 > 0])][0]
            sorted_nodes.append(unclockwise)

            sorted_edges = np.zeros((len(sorted_nodes), 2), 'int32')
    for i in range(len(sorted_nodes) - 1):
        sorted_edges[i, :] = sorted_nodes[i:i + 2]
    sorted_edges[-1, :] = [sorted_nodes[-1], sorted_nodes[0]]

    drag_lift = 0

    arclen = 0
    for edge in sorted_edges:
        node0 = nodes[edge[0], :2]
        u0 = flow[edge[0], 0]
        v0 = flow[edge[0], 1]
        p0 = flow[edge[0], 2]
        node1 = nodes[edge[1], :2]
        u1 = flow[edge[1], 0]
        v1 = flow[edge[1], 1]
        p1 = flow[edge[1], 2]

        tangent = node1 - node0
        normal = np.asarray([-tangent[1], tangent[0]])

        # calculate pressure force
        force_p = 0.5 * (p0 + p1) * normal
        arclen += np.linalg.norm(tangent)

        associated_element = elements[[len(np.setdiff1d(edge, elements[i, :])) == 0 for i in range(elements.shape[0])]][
            0]
        associated_node = np.setdiff1d(associated_element, edge)[0]
        # turn inverse clockwise edge diretion into inverse upwind
        variable = np.dot(np.asarray(flow[associated_node, :2]), tangent)
        if variable < 0.0:
            tangent = -tangent
            variable = -variable
        length = np.linalg.norm(tangent)

        local_matrix = np.hstack([np.asarray([nodes[i, :2] for i in associated_element]), np.ones((3, 1))])
        force_nu = viscos * tangent * variable / np.linalg.det(local_matrix)

        drag_lift = drag_lift + force_nu + force_p

    print(drag_lift)
    return drag_lift

# код чтоб конвертировать буквы в числа (в гран условиях)

# dir_path = '/home/vlad/gnn_diplom/shapeopt-fields_predict-gnn_laminar_flow/GNN_PytorchGeo/datasets/dataset_IrregMesh_rndshap_rndbc_fixobj_0.65_0.07/bcs_copy'
# dest_file = '/home/vlad/gnn_diplom/shapeopt-fields_predict-gnn_laminar_flow/GNN_PytorchGeo/test_csv.csv'
# for file in os.listdir(dir_path):
#     file_path = os.path.join(dir_path,file)

#     f = open(file_path, 'r')
#     txt = f.read().split('\n')
#     nums = txt[4:-1]
#     norm_nums = [eval(str(txt.replace('Pi', str(np.pi))).strip('"')) for txt in nums]

#     file_path = os.path.join(dir_path, file)

#     print(norm_nums, nums[4:], file)
#     res = list(map(float, txt[:4])) + norm_nums
#     np.savetxt(file_path,res)
