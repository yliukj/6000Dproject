import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv, GATConv

import DGPP.Constants as Constants
from DGPP.Layers import EncoderLayer
from torch_geometric.nn.conv import GCNConv

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask

class GCN(nn.Module):
    def __init__(self, d_model, n_layers, edge_index):
        super(GCN, self).__init__()
        self.edge_index = edge_index
        self.layer_stack = nn.ModuleList([
            GATConv(101, d_model)
            for _ in range(n_layers)])

    def forward(self, x):
        for enc_layer in self.layer_stack:
            x = enc_layer(x, self.edge_index)
        return x


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, edge_index, feature,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model

        self.GNN = GCN(d_model, 1, edge_index)
        self.linear = nn.Linear(18, d_model)
        self.flinear = nn.Linear(101, d_model)

        self.gate = nn.Linear(d_model, 3)

        # event type embedding
        self.event_emb = feature
        self.time_emb = nn.Embedding(num_types + 1, d_model, padding_idx=Constants.PAD)
        self.type_weight = nn.Parameter(torch.FloatTensor([0, 3]).view(2, 1, 1, 1))
        self.time_weight = nn.Parameter(torch.FloatTensor([3, 0]).view(2, 1, 1, 1))

        self.time_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

        self.type_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def forward(self, event_type, event_time, policy, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        event_emb = torch.relu(self.GNN(self.event_emb))

        for enc_layer in self.time_stack:
            time_output, _ = enc_layer(
                self.time_emb(event_time.long()), 
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        for enc_layer in self.type_stack:
            type_output, _ = enc_layer(
                event_emb[event_type], 
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
       
        output = torch.stack((time_output, type_output))
        type_output = torch.sum(output * F.softmax(self.type_weight, 0), 0)
        time_output = torch.sum(output * F.softmax(self.time_weight, 0), 0)
        
        return type_output, time_output


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        self.linear_ = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.xavier_normal_(self.linear_.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out

class Time_Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.uniform_(self.linear.weight, -1, 0)

    def forward(self, data, non_pad_mask):
        out = torch.matmul(data, F.softplus(self.linear.weight).t())
        out = torch.relu(out)
        out = out * non_pad_mask
        return out

class Type_Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.uniform_(self.linear.weight, -1, 0)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out

class DGPP(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self, edge_index, feature,
            num_types, d_model=256, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.encoder = Encoder(edge_index, feature,
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)
        self.type_predictor = Predictor(d_model, num_types)
        self.time_predictor = Predictor(d_model, 1)
        self.policy_embed = nn.Linear(18, d_model)

        self.type_policy = Type_Predictor(18, num_types)
        self.time_policy = Time_Predictor(18, 1)


    def forward(self, event_type, event_time, policy):
        non_pad_mask = get_non_pad_mask(event_type)

        type, time = self.encoder(event_type, event_time, policy, non_pad_mask)

        type_prediction = self.type_predictor(type, non_pad_mask) 
        time_prediction = self.time_predictor(time, non_pad_mask) 


        return type, (type_prediction, time_prediction)
