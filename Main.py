import argparse
import numpy as np
import pickle
import random
import itertools
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

import DGPP.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader
from DGPP.Models import DGPP
from tqdm import tqdm

def read_graph(file_name):
    edges = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            s, t = line.split()
            edges.append([int(s), int(t)])
    max_id = np.max(edges) + 2
    graph = np.zeros((max_id, max_id))
    for edge in edges:
        graph[edge[0] + 1, edge[1] + 1] = 1
        graph[edge[1] + 1, edge[0] + 1] = 1
    return graph

def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            feature = data['node']
            data = data[dict_name]
            return data, int(num_types), feature

    print('[Info] Loading train data...')
    train_data, num_types, feature = load_data(opt.data + 'train.pkl', 'train')
    print('[Info] Loading test data...')
    test_data, _, _ = load_data(opt.data + 'test.pkl', 'test')

    data = train_data + test_data
    policy = [elem['policy'] for inst in data for elem in inst]
    policy.sort()
    policy = list(k for k, _ in itertools.groupby(policy))
    policy = torch.tensor(policy, dtype=torch.float32).to(opt.device)
    
    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, testloader, num_types, policy, feature


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type, policy = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time, policy)

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = Utils.time_loss(prediction[1], event_time)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 10
        loss = pred_loss + se / scale_time_loss #+ event_loss
        loss.backward()

        """ update parameters """
        optimizer.step()

        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    return total_event_rate / total_num_pred


def eval_epoch(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type, policy = map(lambda x: x.to(opt.device), batch)

            """ forward """
            enc_out, prediction = model(event_type, event_time, policy)

            _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)
            se = Utils.time_loss(prediction[1], event_time)

            """ note keeping """
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    return total_event_rate / total_num_pred


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt, policy):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = 0  # validation event type prediction accuracy
    valid_rmse = 100  # validation event time prediction RMSE
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_type = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
        print('  - (Training)     '
              'accuracy: {type: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(type=train_type, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_type = eval_epoch(model, validation_data, pred_loss_func, opt)
        print('  - (Testing)     '
              'accuracy: {type: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(type=valid_type, elapse=(time.time() - start) / 60))

        if valid_pred_losses < valid_type:
            valid_pred_losses = valid_type
            
        print('  - [Info] Maximum ll: '
              'Maximum accuracy: {pred: 8.5f}'
              .format(pred=valid_pred_losses))

        scheduler.step()


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0)

    opt = parser.parse_args()

    # default device is CUDA
    opt.device = torch.device('cpu')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """

    trainloader, testloader, num_types, policy, feature = prepare_dataloader(opt)

    g = read_graph("graph.txt")
    edge_index = torch.LongTensor(g.nonzero()).to(opt.device)
    feature = torch.FloatTensor(feature).to(opt.device)

    """ prepare model """
    model = DGPP(edge_index, feature,
        num_types=num_types,
        d_model=opt.d_model,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ train the model """
    train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt, policy)


if __name__ == '__main__':
    seed = 1024 
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    main()
