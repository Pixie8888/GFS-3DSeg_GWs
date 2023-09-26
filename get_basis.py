#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21/12/22 1:50 PM
# @Author  : Yating
# @File    : get_basis.py

import ast
import argparse
import pickle
import time
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


from dataloaders.loader import MyPretrainDataset
from model.dgcnn import DGCNN, DGCNNSeg_att
from util.checkpoint_util import save_pretrain_checkpoint, load_pretrain_checkpoint


def Kmean2Proto(cls_feat, kmean_label, num_cnt):
    ''' array.
    :param cls_feat: (N, D)
    :param cls_size: (N). normalize to [0, 1]
    :param kmean_label: N
    :return: global proto: (num_cnt, d)
    '''
    proto = []
    for i in range(num_cnt):
        center_mask = kmean_label == i
        assert np.sum(center_mask) != 0
        # center_feat = cls_feat[center_mask] * (cls_size[center_mask][:,np.newaxis] / np.max(cls_size[center_mask]))
        # center_feat = cls_feat[center_mask] * softmax(cls_size[center_mask])[:, np.newaxis]
        # center_feat = np.sum(center_feat, axis=0)  #(d)
        center_feat = np.mean(cls_feat[center_mask], axis=0)
        proto.append(center_feat)
    proto = np.stack(proto, axis=0)
    return proto





def compute_svd(proto_list):
    '''
    Args:
        proto_list: (n, d)
    Returns: orthogonal basis

    '''
    u, s, vh = np.linalg.svd(proto_list.transpose(1,0), full_matrices=False)
    # print(s)
    # take u as basis (d, n)
    basis = u.transpose(1,0) # (n,d). numpy
    # only 95%
    for i in range(len(s)):
        if np.sum(s[:i+1]) > 0.95 * np.sum(s):
            break
    # basis = basis[:i+1] # (n,d)
    # print('basis shape: {}'.format(basis.shape))
    print(i)
    basis = u[:, :i+1] @ np.diag(s[:i+1]) @ vh[:i+1, :]
    basis = basis.transpose(1,0)
    print(basis.shape)
    return basis





class DGCNNSeg(nn.Module):
    def __init__(self, args, num_classes):
        super(DGCNNSeg, self).__init__()
        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        in_dim = args.dgcnn_mlp_widths[-1]
        for edgeconv_width in args.edgeconv_widths:
            in_dim += edgeconv_width[-1]
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
        edgeconv_feats, point_feat = self.encoder(pc)
        global_feat = point_feat.max(dim=-1, keepdim=True)[0] # (B, 256, 1)
        edgeconv_feats.append(global_feat.expand(-1,-1,num_points)) # [(b, 64, n), (b, 64, n), (b, 64, n), (b, 256, n)]
        pc_feat = torch.cat(edgeconv_feats, dim=1) # (b, d, n)

        logits = self.segmenter(pc_feat) # (b, 13, n)
        if return_feat == True:
            return logits, torch.cat(edgeconv_feats[:3], dim=1)
        else:
            return logits





def Get_GlobalProto_GlobalKmeans(args, num_cnt=100, save_dir=None):
    ''' perform kmeans on all the base class features!
    Args:
        args:
        num_cnt: kmeans. number of center per category
        save_dir: path to save basis

    Returns:

    '''


    # redefine training dataset !!! the data path is defferent
    if args.dataset == 's3dis':
        from dataloaders.s3dis import S3DISDataset
        DATASET = S3DISDataset(args.cvfold, args.data_path)  # class2scan is defferent from testing dataset!!!
    elif args.dataset == 'scannet':
        from dataloaders.scannet import ScanNetDataset
        DATASET = ScanNetDataset(args.cvfold, args.data_path)
    else:
        raise NotImplementedError('Unknown dataset %s!' % args.dataset)

    CLASSES = sorted(DATASET.train_classes)
    NUM_CLASSES = len(CLASSES) + 1 # include 'bg'
    print('base classes : {}'.format(CLASSES))
    train_CLASS2SCANS = {c: DATASET.class2scans[c] for c in CLASSES}  # only use the train classes in class2scan

    train_data = MyPretrainDataset(args.data_path, CLASSES, train_CLASS2SCANS, mode='train',
                                   num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                   pc_augm=False)
    TRAIN_LOADER = torch.utils.data.DataLoader(train_data, batch_size=1, num_workers=args.n_workers,
                                               shuffle=False, drop_last=False)

    # WRITER = SummaryWriter(log_dir=args.log_dir)

    # Init model and optimizer
    model = DGCNNSeg(args, num_classes=NUM_CLASSES)

    # load weight
    model = load_pretrain_checkpoint(model, args.pretrain_checkpoint_path) # only load encoder weight.
    model = model.cuda()
    model.eval()
    print('model done!')

    collect_feat_size = {k: {'feat': []} for k in range(NUM_CLASSES) if k != 0} # ignore bg. use class_idx.learning order +1
    print(collect_feat_size)
    # clean_record = torch.zeros(NUM_CLASSES)
    # initalize global proto dic, use absolute class id. bg class id = -1.


    with torch.no_grad():
        # collect class-wise feature
        for i, (ptclouds, labels, _) in enumerate(TRAIN_LOADER):
            ptclouds = ptclouds.cuda()  # (1, 9, N)
            # print(ptclouds.shape)
            labels = labels.cuda()  # given label noisy (1, N). int64
            labels = labels[0].float()  # (N,)

            _, feat = model(ptclouds, return_feat=True)  # (1, d, N)
            feat = feat.squeeze(0)  # (d, N)

            for cls in range(NUM_CLASSES):
                if cls == 0:
                    continue

                # get mask
                final_mask = labels == cls
                if torch.sum(final_mask) == 0:
                    continue

                cls_feat = feat[:, final_mask]
                collect_feat_size[cls]['feat'].append(cls_feat.cpu())


        # perform k-means to intialize each class prototype. cpu operation.

        # collect point feature
        MAX_NUM = 300000
        point_feat = []
        for i in range(NUM_CLASSES):
            if i == 0:
                continue
            cls_feat = torch.cat(collect_feat_size[i]['feat'], dim=1).transpose(0, 1).numpy()  # (N, d)
            print('cls : {} has {} features'.format(i, cls_feat.shape[0]))
            if cls_feat.shape[0] > MAX_NUM:
                sampled_point_inds = np.random.choice(np.arange(cls_feat.shape[0]), MAX_NUM, replace=False)
                cls_feat = cls_feat[sampled_point_inds, :]  # (n, d)
            else:
                cls_feat = cls_feat

            point_feat.append(cls_feat)
            del collect_feat_size[i]

        point_feat = np.concatenate(point_feat, axis=0) # (n, d)



        begin = time.time()
        kmean = KMeans(n_clusters=num_cnt, init='k-means++').fit(point_feat)
        print('kmean : {}'.format(time.time() - begin))
        kmean_label = kmean.labels_
        cls_proto = Kmean2Proto(point_feat, kmean_label, num_cnt)  # numpy array (num_cnt, d)
        proto_list = cls_proto # (num_cnt, d)

        basis = compute_svd(proto_list)

        # save
        proto_file = os.path.join(save_dir, 'GlobalKmeans_EdgeConv123_cnt={}_energy=095_SVDReconstruct.pkl'.format(args.num_cnt))  # log_dir == pretrain_checkpoint_path
        print(proto_file)
        with open(proto_file, 'wb') as f:
            pickle.dump(basis, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--dataset', type=str, default='s3dis', help='Dataset name: s3dis|scannet')
    parser.add_argument('--cvfold', type=int, default=0, help='Fold left-out for testing in leave-one-out setting '
                                                              'Options:{0,1}')
    parser.add_argument('--data_path', type=str, default='/home/yating/Documents/3d_segmentation/GFS_pcd_seg/datasets/S3DIS/blocks_bs1_s1',
                        help='Directory to the source data')
    parser.add_argument('--pretrain_checkpoint_path', type=str, default='/home/yating/Documents/3d_segmentation/GFS_pcd_seg/mpti/log_s3dis/log_pretrain_s3dis_S0_LongTail',
                        help='pretrain weight')

    # optimization
    parser.add_argument('--batch_size', type=int, default=16, help='Number of samples/tasks in one batch')
    parser.add_argument('--n_workers', type=int, default=16, help='number of workers to load data')
    parser.add_argument('--n_iters', type=int, default=100, help='number of iterations/epochs to train')


    parser.add_argument('--pretrain_lr', type=float, default=0.001, help='pretrain learning rate [default: 0.001]')
    parser.add_argument('--pretrain_weight_decay', type=float, default=0.0001, help='weight decay for regularization')
    parser.add_argument('--pretrain_step_size', type=int, default=50, help='Period of learning rate decay')
    parser.add_argument('--pretrain_gamma', type=float, default=0.5,
                        help='Multiplicative factor of learning rate decay')

    # few-shot episode setting
    parser.add_argument('--n_way', type=int, default=2, help='Number of classes for each episode: 1|3')
    parser.add_argument('--k_shot', type=int, default=5, help='Number of samples/shots for each class: 1|5')
    parser.add_argument('--n_queries', type=int, default=1, help='Number of queries for each class')
    parser.add_argument('--n_episode_test', type=int, default=100,
                        help='Number of episode per configuration during testing')

    # Point cloud processing
    parser.add_argument('--pc_npts', type=int, default=2048, help='Number of input points for PointNet.')
    parser.add_argument('--pc_attribs', default='xyzrgbXYZ',
                        help='Point attributes fed to PointNets, if empty then all possible. '
                             'xyz = coordinates, rgb = color, XYZ = normalized xyz')
    parser.add_argument('--pc_augm', action='store_true', help='Training augmentation for points in each superpoint')
    parser.add_argument('--pc_augm_scale', type=float, default=0,
                        help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', type=int, default=1,
                        help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', type=float, default=0,
                        help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', type=int, default=1,
                        help='Training augmentation: Bool, Gaussian jittering of all attributes')

    # feature extraction network configuration
    parser.add_argument('--dgcnn_k', type=int, default=20, help='Number of nearest neighbors in Edgeconv')
    parser.add_argument('--edgeconv_widths', default='[[64,64], [64, 64], [64, 64]]', help='DGCNN Edgeconv widths')
    parser.add_argument('--dgcnn_mlp_widths', default='[512, 256]',
                        help='DGCNN MLP (following stacked Edgeconv) widths')
    parser.add_argument('--base_widths', default='[128, 64]', help='BaseLearner widths')  # didn't use in pre-train
    parser.add_argument('--output_dim', type=int, default=64,
                        help='The dimension of the final output of attention learner or linear mapper')  # didn't use in pre-train
    parser.add_argument('--use_attention', action='store_false',
                        help='it incorporate attention learner')  # use attention

    # protoNet configuration
    parser.add_argument('--dist_method', default='gaussian',
                        help='Method to compute distance between query feature maps and prototypes.[Option: cosine|euclidean]')

    # MPTI configuration
    parser.add_argument('--n_subprototypes', type=int, default=100,
                        help='Number of prototypes for each class in support set')
    parser.add_argument('--k_connect', type=int, default=200,
                        help='Number of nearest neighbors to construct local-constrained affinity matrix')
    parser.add_argument('--sigma', type=float, default=1., help='hyeprparameter in gaussian similarity function')


    # global prototype
    parser.add_argument('--num_cnt', type=int, default=10, help='number of global prototype per class')
    parser.add_argument('--seed', default=123, type=int, help='seed')
    parser.add_argument('--save_path', type=str, default='log_s3dis/S0_K5', help='path to save the basis')
    args = parser.parse_args()


    args.edgeconv_widths = ast.literal_eval(args.edgeconv_widths)
    args.dgcnn_mlp_widths = ast.literal_eval(args.dgcnn_mlp_widths)
    args.base_widths = ast.literal_eval(args.base_widths)
    args.pc_in_dim = len(args.pc_attribs)

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    # random.seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    Get_GlobalProto_GlobalKmeans(args, save_dir=args.save_path, num_cnt=args.num_cnt)
