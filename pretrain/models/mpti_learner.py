""" MPTI with/without attention Learner for Few-shot 3D Point Cloud Semantic Segmentation

Author: Zhao Na, 2020
"""
import os
import torch
from torch import optim
from torch.nn import functional as F

from models.mpti import MultiPrototypeTransductiveInference
from utils.checkpoint_util import load_pretrain_checkpoint, load_model_checkpoint
import numpy as np

import pickle


class MPTILearner(object):
    def __init__(self, args, mode='train'):

        # init model and optimizer
        self.model = MultiPrototypeTransductiveInference(args)
        # print(self.model)
        if torch.cuda.is_available():
            self.model.cuda()

        if mode=='train':
            if args.use_attention:
                self.optimizer = torch.optim.Adam(
                    [{'params': self.model.encoder.parameters(), 'lr': 0.0001},
                     {'params': self.model.base_learner.parameters()},
                     {'params': self.model.att_learner.parameters()}], lr=args.lr)
            else:
                self.optimizer = torch.optim.Adam(
                    [{'params': self.model.encoder.parameters(), 'lr': 0.0001},
                     {'params': self.model.base_learner.parameters()},
                     {'params': self.model.linear_mapper.parameters()}], lr=args.lr)
            #set learning rate scheduler
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size,
                                                          gamma=args.gamma)
            if args.model_checkpoint_path is None:
                # load pretrained model for point cloud encoding
                self.model = load_pretrain_checkpoint(self.model, args.pretrain_checkpoint_path)
            else:
                # resume from model checkpoint
                self.model, self.optimizer = load_model_checkpoint(self.model, args.model_checkpoint_path,
                                                                   optimizer=self.optimizer, mode='train')
        elif mode=='test':
            # Load model checkpoint
            self.model = load_model_checkpoint(self.model, args.model_checkpoint_path, mode='test')
        else:
            raise ValueError('Wrong GraphLearner mode (%s)! Option:train/test' %mode)

    def train(self, data):
        """
        Args:
            data: a list of torch tensors wit the following entries.
            - support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            - support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            - query_x: query point clouds with shape (n_queries, in_channels, num_points)
            - query_y: query labels with shape (n_queries, num_points)
        """

        [support_x, support_y, query_x, query_y] = data

        # # onlu use one query. align the test setting
        # x = np.random.randint(low=0, high=query_x.shape[0])
        # query_x = query_x[x:x+1]
        # query_y = query_y[x:x+1]

        self.model.train()

        query_logits, loss= self.model(support_x, support_y, query_x, query_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.lr_scheduler.step()

        query_pred = F.softmax(query_logits, dim=1).argmax(dim=1)
        correct = torch.eq(query_pred, query_y).sum().item()  # including background class
        accuracy = correct / (query_y.shape[0]*query_y.shape[1])

        return loss, accuracy


    def test(self, data):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points), each point \in {0,1}.
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
        """
        [support_x, support_y, query_x, query_y] = data
        self.model.eval()

        with torch.no_grad():
            logits, loss= self.model(support_x, support_y, query_x, query_y)
            pred = F.softmax(logits, dim=1).argmax(dim=1)
            correct = torch.eq(pred, query_y).sum().item()
            accuracy = correct / (query_y.shape[0]*query_y.shape[1])

        return pred, loss, accuracy


    def test_gfs(self, base_dataloader, val_supp_loader, VALID_LOADER, base_classes, novel_classes, all_classes,
                 k_connect, log_dir, iter):
        '''
        Args:
            base_dataloader: base dataloader
            base_classes: base class names
            novel_classes: novel class names
            all_classes: all class names
            k_connect: knn
            log_dir: save base_proto

        Returns:

        '''

        self.model.eval()

        # 1. get prototype of base classes
        with torch.no_grad():
            # 0. get base proto
            collect_feat_size = {k: [] for k in range(len(base_classes))}
            for _, (ptclouds, labels) in enumerate(base_dataloader):
                ptclouds = ptclouds.cuda()  # (1, 9, N)
                # print(ptclouds.shape)
                labels = labels.cuda()  # given label noisy (1, N). int64
                labels = labels[0].float()  # (N,)
                # gt_labels = gt_labels[0] # (N, )
                # cluster_label = cluster_label.cuda()
                # cluster_label = cluster_label[0].long() # (N). index need to be long

                feat = self.model.getFeatures(ptclouds)  # (1, d, N)
                feat = feat.squeeze(0)  # (d, N)

                # collect base class: [0,1,2,3,4,5,6]
                for cls in range(len(base_classes)):

                    mask = labels == cls + 1
                    if torch.sum(mask) > 0:
                        # cls_feat = torch.mean(feat[:, mask], dim=1, keepdim=True) # (d, n)
                        cls_feat = feat[:, mask]
                        collect_feat_size[cls].append(cls_feat.cpu())

            # get prototypees per class
            max_points = 200000
            cls_proto_dict = {}
            for cls in range(len(base_classes)):
                # get feat:
                cls_feat = torch.cat(collect_feat_size[cls], dim=1)  # (d, n)
                if cls_feat.shape[1] > max_points:
                    sampled_point_inds = np.random.choice(np.arange(cls_feat.shape[1]), max_points, replace=False)
                    cls_feat = cls_feat[:, sampled_point_inds].transpose(1, 0).contiguous()  # (n, d)
                else:
                    cls_feat = cls_feat.transpose(1, 0).contiguous()  # (n, d) # cpu tensor
                print(cls_feat.shape)
                # get proto
                cls_proto = self.model.getMutiplePrototypes(cls_feat, k=100)  # (100,d) # cpu tneosr?
                # get absolute cls id. class_name
                cls_id = base_classes[cls]
                cls_proto_dict[cls_id] = cls_proto.numpy()  # numpy

            # with open(os.path.join(log_dir, 'base_proto_{}.pkl'.format(iter)), 'wb') as f:
            #     pickle.dump(cls_proto_dict, f)

            # load base proto
            base_protoes = []
            base_labels = []
            for cls, proto in cls_proto_dict.items():
                print('base class: {}'.format(cls))
                # proto
                tmp_proto = torch.from_numpy(proto)  # cpu tensor (n, d)
                base_protoes.append(tmp_proto)
                # label
                cls_label = torch.zeros((tmp_proto.shape[0], len(all_classes)))  # (n, 13)
                cls_label[:, cls] = 1
                base_labels.append(cls_label)

            base_protoes = torch.cat(base_protoes, dim=0)  # (n, d)
            base_labels = torch.cat(base_labels, dim=0)  # (n, 13)
            # put on cuda
            base_protoes = base_protoes.cuda()  # (n, d)
            base_labels = base_labels.cuda()  # (n, 13)

            # 2. get novel class prototypes:
            new_cls_feat_dict = {cls: [] for cls in novel_classes}

            for _, (pcd, mask, cls_id) in enumerate(val_supp_loader):  # cls_id is the class_name
                pcd = pcd.cuda()  # (b, 9, 2048)
                mask = mask.cuda()[0]  # (2048)
                # get fg feat:
                pcd_feat = self.model.getFeatures(pcd)[0]  # (d, 2048)
                fg_feat = pcd_feat[:, mask == 1]  # (d, n)
                new_cls_feat_dict[cls_id[0].item()].append(fg_feat)

            # get proto
            novel_protoes = []
            novel_labels = []

            for cls in novel_classes:
                print('processing prototype of novel class {}'.format(cls))
                tmp_feat = torch.cat(new_cls_feat_dict[cls], dim=1).transpose(1, 0).contiguous()  # (n, d)
                proto = self.model.getMutiplePrototypes(tmp_feat, k=100)  # (100, d)
                novel_protoes.append(proto)

                tmp_label = torch.zeros((proto.shape[0], len(all_classes)), device=pcd.device)  # (n, 13)
                tmp_label[:, cls] = 1
                novel_labels.append(tmp_label)

            novel_protoes = torch.cat(novel_protoes, dim=0)  # (n, d)
            novel_labels = torch.cat(novel_labels, dim=0)  # (n, 13)
            print('done processing novel prototype')

            # 3. infer query label
            pred_labels_list = []
            gt_labels_list = []
            for i, (pcd, label) in enumerate(VALID_LOADER):

                if (i + 1) % 100 == 0:
                    print('processing {} query pcd'.format(i + 1))

                pcd = pcd.cuda()  # (b, d, 2048)
                label = label.cuda()  # (1, 2048)

                # get query feat
                query_feat = self.model.getFeatures(pcd)[0].transpose(1, 0)  # (2048, d)

                # label propagation
                node_feat = torch.cat([base_protoes, novel_protoes, query_feat], dim=0)  # (n, d)
                Y = torch.zeros((query_feat.shape[0], len(all_classes)), device=pcd.device)  # (n, 13)
                Y = torch.cat([base_labels, novel_labels, Y], dim=0)  # (n, 13)
                num_prototypes = base_protoes.shape[0] + novel_protoes.shape[0]

                # label propagation
                A = self.model.calculateLocalConstrainedAffinity(node_feat, k=k_connect)
                Z = self.model.label_propagate(A, Y)  # (num_nodes, 13)

                query_pred = torch.argmax(torch.softmax(Z[num_prototypes:, :], dim=1), dim=1, keepdim=False).unsqueeze(0)  # (1, 2048)
                pred_labels_list.append(query_pred)
                gt_labels_list.append(label)

        return pred_labels_list, gt_labels_list, cls_proto_dict