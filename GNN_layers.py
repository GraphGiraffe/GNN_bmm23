import torch
from torch import nn

import torch_scatter

from torch_geometric.nn.conv import MessagePassing


class ProcessorLayer(MessagePassing):

    def __init__(self, edge_feats, node_feats, hidden_state, idx=0, selu=False):
        super(ProcessorLayer, self).__init__()

        self.name = 'processor'
        self.idx = idx
        activation = nn.ReLU()

        if selu:
            activation = nn.SELU()

        self.edge_mlp = nn.Sequential(nn.LazyLinear(hidden_state),
                                      activation,
                                      # nn.AlphaDropout(p=0.05),
                                      nn.LazyLinear(edge_feats)
                                      # activation,
                                      )

        self.node_mlp = nn.Sequential(nn.LazyLinear(hidden_state),
                                      activation,
                                      # nn.AlphaDropout(p=0.05),
                                      nn.LazyLinear(node_feats),
                                      activation,
                                      )

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        out, updated_edges = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        print(x.shape, out.shape, updated_edges.shape, edge_index.shape, edge_attr.shape)
        if x.shape[0] != out.shape[0]:
          out = torch.cat([out[:0], out], dim=1)
        updated_nodes = torch.cat([x, out], dim=1)
        updated_nodes = self.node_mlp(updated_nodes)

        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        # print(x_i.shape, x_j.shape,edge_attr.shape)

        updated_edges = torch.cat([torch.div(x_i + x_j, 2), torch.abs(x_i - x_j) / 2, edge_attr], 1)
        # print(updated_edges.shape, 'hui message')
        updated_edges = self.edge_mlp(updated_edges)
        return updated_edges

    def aggregate(self, updated_edges, edge_index):
        node_dim = 0
        out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce='sum')
        return out, updated_edges


class SmoothingLayer(MessagePassing):

    def __init__(self, idx=0):
        super(SmoothingLayer, self).__init__()

        self.name = 'smoothing'
        self.idx = idx

    def forward(self, x, edge_index, edge_attr):
        out_nodes, out_edges = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out_nodes, out_edges

    def message(self, x_i, x_j):
        updated_edges = (x_i + x_j) / 2
        return updated_edges

    def aggregate(self, updated_edges, edge_index):
        node_dim = 0
        out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce='mean')
        return out, updated_edges
