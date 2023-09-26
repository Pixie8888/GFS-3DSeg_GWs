"""DGCNN as Backbone to extract point-level features
   Adapted from https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
   Author: Zhao Na, 2020
"""

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention import SelfAttention


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x) #(B,N,N)
    xx = torch.sum(x ** 2, dim=1, keepdim=True) #(B,1,N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1) #(B,N,N)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B,N,k)
    return idx


def get_edge_feature(x, K=20, idx=None):
    """Construct edge feature for each point
      Args:
        x: point clouds (B, C, N)
        K: int
        idx: knn index, if not None, the shape is (B, N, K)
      Returns:
        edge feat: (B, 2C, N, K)
    """
    B, C, N = x.size()
    if idx is None:
        idx = knn(x, k=K)  # (batch_size, num_points, k)
    central_feat = x.unsqueeze(-1).expand(-1,-1,-1,K)
    idx = idx.unsqueeze(1).expand(-1, C, -1, -1).contiguous().view(B,C,N*K)
    knn_feat = torch.gather(x, dim=2, index=idx).contiguous().view(B,C,N,K)
    edge_feat = torch.cat((knn_feat-central_feat, central_feat), dim=1)
    return edge_feat


class conv2d(nn.Module):
    def __init__(self, in_feat, layer_dims, batch_norm=True, relu=True, bias=False):
        super().__init__()
        self.layer_dims = layer_dims
        layers = []
        for i in range(len(layer_dims)):
            in_dim = in_feat if i==0 else layer_dims[i-1]
            out_dim = layer_dims[i]
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_dim))
            if relu:
                layers.append(nn.LeakyReLU(0.2))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class conv1d(nn.Module):
    def __init__(self, in_feat, layer_dims, batch_norm=True, relu=True, bias=False):
        super().__init__()
        self.layer_dims = layer_dims
        layers = []
        for i in range(len(layer_dims)):
            in_dim = in_feat if i==0 else layer_dims[i-1]
            out_dim = layer_dims[i]
            layers.append(nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=bias))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            if relu:
                layers.append(nn.LeakyReLU(0.2))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DGCNN(nn.Module):
    """
    DGCNN with only stacked EdgeConv, return intermediate features if use attention
    Parameters:
      edgeconv_widths: list of layer widths of edgeconv blocks [[],[],...] [[64,64], [64, 64], [64, 64]]
      mlp_widths: list of layer widths of mlps following Edgeconv blocks [512, 256]
      nfeat: number of input features 9
      k: number of neighbors
      conv_aggr: neighbor information aggregation method, Option:['add', 'mean', 'max', None]
    """
    def __init__(self, edgeconv_widths, mlp_widths, nfeat, k=20, return_edgeconvs=False):
        super(DGCNN, self).__init__()
        self.n_edgeconv = len(edgeconv_widths) # 3
        self.k = k
        self.return_edgeconvs = return_edgeconvs # true

        self.edge_convs = nn.ModuleList()
        for i in range(self.n_edgeconv):
            if i==0:
                in_feat = nfeat*2 # 9*2=18
            else:
                in_feat = edgeconv_widths[i-1][-1]*2

            self.edge_convs.append(conv2d(in_feat, edgeconv_widths[i]))

        in_dim = 0
        for edgeconv_width in edgeconv_widths:
            in_dim += edgeconv_width[-1]
        self.conv = conv1d(in_dim, mlp_widths)

    def forward(self, x):
        edgeconv_outputs = []
        for i in range(self.n_edgeconv):
            x = get_edge_feature(x, K=self.k) # (b, 2c, N, k)
            x = self.edge_convs[i](x)
            x = x.max(dim=-1, keepdim=False)[0]
            edgeconv_outputs.append(x) # [(b, 64, n), (b, 64, n), (b, 64, n)]

        out = torch.cat(edgeconv_outputs, dim=1)
        out = self.conv(out) # (b, 256, N)

        if self.return_edgeconvs:
            return edgeconv_outputs, out
        else:
            return edgeconv_outputs[0], out


class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, in_channels, params):
        super(BaseLearner, self).__init__()

        self.num_convs = len(params)
        self.convs = nn.ModuleList()

        for i in range(self.num_convs):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = params[i-1]
            self.convs.append(nn.Sequential(
                              nn.Conv1d(in_dim, params[i], 1),
                              nn.BatchNorm1d(params[i])))

    def forward(self, x):
        for i in range(self.num_convs):
            x = self.convs[i](x)
            if i != self.num_convs-1:
                x = F.relu(x)
        return x


class DGCNNSeg_att(nn.Module):
    def __init__(self, args, num_classes):
        super(DGCNNSeg_att, self).__init__()
        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k)

        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)
        self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)

        self.feat_dim = args.edgeconv_widths[0][-1] + args.output_dim + args.base_widths[-1]


        # in_dim = args.dgcnn_mlp_widths[-1]
        # for edgeconv_width in args.edgeconv_widths:
        #     in_dim += edgeconv_width[-1]

        in_dim = self.feat_dim
        self.segmenter = nn.Sequential(
                            nn.Conv1d(in_dim, 256, 1, bias=False),
                            nn.BatchNorm1d(256),
                            nn.LeakyReLU(0.2),
                            nn.Conv1d(256, 128, 1),
                            nn.BatchNorm1d(128),
                            nn.LeakyReLU(0.2),
                            nn.Dropout(0.3),
                            nn.Conv1d(128, num_classes, 1)
                         )

    def forward(self, pc, return_feat=False):
        num_points = pc.shape[2]

        # get feature
        feat_level1, feat_level2 = self.encoder(pc)
        feat_level3 = self.base_learner(feat_level2)
        att_feat = self.att_learner(feat_level2)
        pc_feat = torch.cat((feat_level1, att_feat, feat_level3), dim=1)


        # edgeconv_feats, point_feat = self.encoder(pc)
        # global_feat = point_feat.max(dim=-1, keepdim=True)[0] # (B, 256, 1)
        # edgeconv_feats.append(global_feat.expand(-1,-1,num_points)) # [(b, 64, n), (b, 64, n), (b, 64, n), (b, 256, n)]
        # pc_feat = torch.cat(edgeconv_feats, dim=1)

        logits = self.segmenter(pc_feat) # (b, 13, n)

        if return_feat:
            return logits, feat_level1
        else:
            return logits


