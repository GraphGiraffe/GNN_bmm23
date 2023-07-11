import os
import os.path as osp
import json
import shutil

import torch
from torch_geometric.loader import DataLoader

from GNN_models import FlowGNN
from data_utils import read_dataset
from training_utils import train_model, loss_mae, loss_mse

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg_path = 'datasets/dataset_rndshapSM_rndbc_fixobj_0.65_0.07/gnn_skipBCFC_1500smpls_GP_coordnorm_flownorm_sc036912_8l_64N_v1.json'

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    dataset_cfg = cfg['dataset_cfg']
    nn_cfg = cfg['nn_cfg']
    train_cfg = cfg['train_cfg']

    node_filters = nn_cfg['node_filters']
    edge_filters = nn_cfg['edge_filters']
    fc_in_dim = nn_cfg['fc_in_dim']
    fc_out_dim = nn_cfg['fc_out_dim']
    out_dim = nn_cfg['out_dim']
    fc_con_list = nn_cfg['fc_con_list']
    fc_hidden_layers = nn_cfg["fc_hidden_layers"]

    loss = train_cfg["loss"]
    train_ratio = train_cfg['train_ratio']
    valid_ratio = train_cfg['valid_ratio']
    num_epochs = train_cfg['num_epochs']
    decay_factor = train_cfg['decay_factor']
    decay_step = train_cfg['decay_step']
    batch_size = train_cfg['batch_size']

    if loss == 'mae':
        loss_fn = loss_mae
    elif loss == 'mse':
        loss_fn = loss_mse
    else:
        raise RuntimeError('loss must be in ["mae", "mse"]')

    out_dir = cfg['out_dir']
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(cfg_path, osp.join(out_dir, 'cfg.json'))

    train_dataset = read_dataset(dataset_cfg['train_dataset_dir'],
                                 dataset_cfg['train_dataset_source_len'],
                                 dataset_cfg['train_dataset_name'])

    val_dataset = read_dataset(dataset_cfg['val_dataset_dir'],
                               dataset_cfg['val_dataset_source_len'],
                               dataset_cfg['val_dataset_name'])

    train_size = int(dataset_cfg['train_dataset_source_len'] * train_ratio)
    val_size = int(dataset_cfg['val_dataset_source_len'] * valid_ratio)
    train_dataset = train_dataset[:train_size]
    val_dataset = val_dataset[:val_size]
    test_dataset = val_dataset[val_size:]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(train_dataset[0])

    gnn_model = FlowGNN(edge_filters, node_filters, fc_in_dim, fc_out_dim,
                        fc_con_list=fc_con_list, fc_hidden_layers=fc_hidden_layers)

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_factor)
    train_loss_hist, val_loss_hist, _ = train_model(gnn_model, train_dataloader, val_dataloader,
                                                    device, num_epochs, optimizer, scheduler, loss_fn,
                                                    out_dir, collect_grads=False)
