import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


from model.dgcnn import DGCNN
from model.attention import SelfAttention


manual_seed=321
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
random.seed(manual_seed)



class mpti_net_Point_GeoAsWeight_v2(nn.Module):
    def __init__(self, classes=13, criterion=nn.CrossEntropyLoss(), args=None, base_num=7, gp=None, energy=None):
        ''' use geometric primiytives as visual words to describe any point cloud. Only use three layer of edge conv'''
        ''' hard coding + No atten
        classes: number of total classes in the whole dataset
        gp: precomputed geometric primitives. (k, d). cpu tensor. no grad.
        '''
        super(mpti_net_Point_GeoAsWeight_v2, self).__init__()
        # assert layers in [50, 101, 152]
        # assert 2048 % len(bins) == 0
        assert classes > 1
        # assert zoom_factor in [1, 2, 4, 8]
        # self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.classes = classes  # 13

        # define model
        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k, return_edgeconvs=True)
        # # freeze edge convs:
        # for param in self.encoder.edge_convs.parameters():
        #     param.requires_grad = False

        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)
        self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)

        self.feat_dim = args.edgeconv_widths[0][-1] + args.output_dim + args.base_widths[-1]

        # load gp
        self.gp = gp
        self.gp.requires_grad = False

        # define classifier
        # main_dim = self.gp.shape[0] + self.feat_dim
        main_dim = 128
        # aux_dim = 256
        self.main_proto = nn.Parameter(torch.randn((classes, main_dim)))  # (13, 192) this is for the testing classes
        self.bg_proto = nn.Parameter(torch.randn((1, main_dim))) # during base training, set all the novel classes as bg. no use in the test.

        self.args = args
        self.base_num = base_num

        # fusion of geometric feature with semantic feature
        self.fusion = nn.Sequential(nn.Conv1d(in_channels=self.feat_dim+self.gp.shape[0], out_channels=main_dim, kernel_size=1),
                                    nn.BatchNorm1d(main_dim),
                                    nn.LeakyReLU(0.2))


        self.energy = energy
        print('model using energy: {}'.format(self.energy))

    def Get_Fg_Feat(self, x, y):
        '''
        Args:
            x: (1, d, 2048)
            y: (1, 2048) binary mask. support point cloud
        Returns: fg feature (n, d)
        '''
        y = y[0] # (2048)
        # get feature of x
        point_feat, _, gp_feat = self.getFeatures(x)  # (1, c, 2048)
        point_feat = point_feat[0] # (d, 2048)
        fg_feat = point_feat[:,y==1] # (d, n)

        # add gp feat
        gp_feat = gp_feat[0] # (k, 2048)
        fg_gp_feat = gp_feat[:, y==1] # (k,n)

        return fg_feat.transpose(1,0), fg_gp_feat.transpose(1,0) # (n,k)



    def get_gp_weight(self, gp_classifier, gp_feat, use_bg_weight=False, gt_label=None, th=None):
        '''
        Args: k is the feature dimention
            gp_classifier: (n_cls, k)
            gp_feat: (b, k, n) is only for debugging
            gt_label: (b, n)

        Returns: (b, cls, n)
        '''

        score = torch.matmul(gp_classifier.unsqueeze(0).repeat(gp_feat.shape[0], 1, 1), gp_feat) # (b, cls, n) score is 0 / 1
        # check score accuracy
        if gt_label != None and use_bg_weight == False: # test
            gt_one_hot = F.one_hot(gt_label, num_classes=score.shape[1]).transpose(2,1) # (b, cls, n)
            acc = torch.sum(gt_one_hot * score, dim=1) # (b, n)
            acc = torch.mean(acc)
            # novel class acc:
            novel_mask = gt_label > self.base_num-1
            if torch.sum(novel_mask) > 0:
                novel_acc = torch.mean(torch.sum(gt_one_hot * score, dim=1)[novel_mask])
            else:
                novel_acc = torch.zeros_like(acc)

        elif gt_label != None and use_bg_weight == True: # train
            gt_one_hot = F.one_hot(gt_label, num_classes=score.shape[1]+1).transpose(2, 1)  # (b, bg+cls, n)
            gt_one_hot = gt_one_hot[:,1:,:] # (b, cls, n)
            acc = torch.sum(gt_one_hot * score, dim=1)  # (b, n)
            acc = torch.mean(acc)
            novel_acc = 0.

        else:
            acc = 0.
            novel_acc = 0.

        # get weight
        weight = torch.ones_like(score)
        weight[score == 1] = torch.tensor(th, device=score.device, dtype=score.dtype) # (b, cls, n)


        # for training: bg weight
        if use_bg_weight == True:
            bg_weight = torch.ones((weight.shape[0], 1, weight.shape[2]), dtype=weight.dtype, device=weight.device) # (b, 1, n). value=1
            weight = torch.cat([bg_weight, weight], dim=1) # (b, bg+n_cls, n)
            # set weight[target] = 1
            gt_mask = F.one_hot(gt_label, num_classes=score.shape[1] + 1).transpose(2, 1)  # (b, bg+cls, n)
            weight[gt_mask==1] = torch.tensor(1, device=score.device, dtype=score.dtype)




        return weight, acc, novel_acc

    def forward(self, x, y=None, gened_proto=None,\
                gen_proto=False, eval_model=False, target_cls=None, segment_label=None,
                geo2sem_proto=None, base_class_coding=None, novel_class_coding=None, bg_class_coding=None):
        '''
        Args: point-level prediction. No use of segment
            x: (b, d, 2048)
            y: (b, 2048)
            gened_proto:
            base_num:
            novel_num:
            epoch: define as epoch
            gen_proto:
            eval_model:
            visualize:
            target_cls: in the novel class stage, the novel class idx in the whole testing classes. also is the oder in the main_proto.
            base_class_coding: (n_base, k)
            novel_class_coding: (n_new, k)
            bg_class_coding: (k,)
        Returns:

        '''
        base_num = self.base_num

        point_feat, semantic_feat, gp_feat = self.getFeatures(x, segment_label=segment_label) # (k+d, m1+m2..) . (b, k_d, n)


        if eval_model:

            #### evaluation
            if len(gened_proto.size()[:]) == 3:
                gened_proto = gened_proto[0] # p_orig

            refine_proto = self.post_refine_proto_v2(proto=self.main_proto, x=point_feat, point_feat=point_feat) # (b, classes, c)


            refine_proto[:, :base_num] = refine_proto[:, :base_num] + gened_proto[:base_num].unsqueeze(0) # refine proto is not l2 norm, but gened_proto is l2 norm. mismatch?
            refine_proto[:, base_num:] = refine_proto[:, base_num:] * 0 + gened_proto[base_num:].unsqueeze(0)
            x_pre = self.get_pred(point_feat, refine_proto) # (b, cls, n)

            # gp weight
            gp_coding = torch.cat([base_class_coding, novel_class_coding], dim=0) # (n_cls, k)
            # get gp weight
            gp_weight, gp_acc, gp_novel_acc = self.get_gp_weight(gp_coding, gp_feat, gt_label=y, th=self.args.eval_weight)  # (b, n_base+n_novel, n)
            # pred
            x_pre = x_pre * gp_weight



            return x_pre, gp_acc, gp_novel_acc

        else:
            ##### training

            # fake novel + fake base
            fake_num = x.size(0) // 2  # B/2
            ori_proto, fake_novel = self.generate_fake_proto(x=point_feat[fake_num:], y=y[fake_num:], main_proto=self.main_proto.clone())  # ori_new_proto is eqn.8

            # # gp coding
            # gp_coding, _ = self.generate_fake_proto(x=gp_feat[fake_num:], y=y[fake_num:],
            #                                 main_proto=base_class_coding, fake_novel=fake_novel, post_processing=True)  # (n_base, k)
            #
            # # get gp weight
            # # gp_coding = torch.cat([bg_class_coding.unsqueeze(0), gp_coding], dim=0) # (bg+n_base, k)
            # gp_weight, gp_acc, _ = self.get_gp_weight(gp_coding, gp_feat, use_bg_weight=True, gt_label=y, th=1.) # (b, n_base, n). for training, weight=1.



            # # enhance with geo2sem proto
            # ori_new_proto_new = self.enhance_with_geo2sem_proto(ori_new_proto, gp_feat, y, fake_novel=fake_novel, fake_num=fake_num)

            x_pre_1 = self.get_pred(x=point_feat, proto=ori_proto, use_bg_proto=True)  # logits. pred via eqn.8. (b, bg+cls, n). use the whole batch as query... and the first half batch is support...
            # add weight
            weight = torch.ones_like(x_pre_1) # (b, bg+n_cls, n)
            # # weight[:, 1:gp_weight.shape[1]+1, :] = gp_weight # set other classes weight as 1
            # weight[:, 0:gp_weight.shape[1], :] = gp_weight  # set other classes weight as 1

            loss_ce_1 = self.criterion(x_pre_1, y)


            # query pred: should be same as testing
            refine_proto = self.post_refine_proto_v2(proto=self.main_proto.clone(), x=point_feat, point_feat=point_feat, use_bg_proto=True)  # eqn.6 only update base proto. (b, classes, c)
            post_refine_proto = refine_proto.clone()



            post_refine_proto[:, :base_num] = post_refine_proto[:, :base_num] + ori_proto[:base_num].unsqueeze(0)
            post_refine_proto[:, base_num:] = post_refine_proto[:, base_num:] * 0 + ori_proto[base_num:].unsqueeze(0) # (b, cls, d)
            x_pre_2 = self.get_pred(x=point_feat, proto=post_refine_proto, use_bg_proto=True)  # (b, cls, n)

            loss_ce_2 = self.criterion(x_pre_2, y)
            # print(torch.isnan(x).any(), torch.isnan(post_refine_proto).any(), torch.isnan(segment_label).any(), torch.isnan(segment_feat).any())
            # loss
            ce_loss = 0.5 * loss_ce_2 + 0.5 * loss_ce_1
            # ce_loss = (loss_ce_1 + loss_ce_2 + loss_ce_4) / 3.




            return x_pre_2.max(1)[1], ce_loss


    def post_refine_proto_v2(self, proto, x, point_feat, use_bg_proto=False):
        ''' refine the base proto via query prediction. eqn. 6. use segment_feat(x) to predict label. Then aggregate feature using point_feat.
        Args: n: number of point. c: feature dim.

            proto: (13, 192)
            x: point feature of this batch (b, d, n)
            point_feat: (b, d, n)
            segment_label: (b,n)
        Returns: eqn.6 (b, classes, c)

        '''
        if use_bg_proto == False:

            b, c, n = point_feat.shape[:]

            pred = self.get_pred(x, proto).view(b, proto.shape[0], n)  # (b, 13, n)
            pred = F.softmax(pred, 2)  # (b, 21, h*w)

            pred_proto = pred @ point_feat.view(b, c, n).permute(0, 2, 1) # (b, classes, c)
            pred_proto_norm = F.normalize(pred_proto, 2, -1)  # (b, classes, c)
            proto_norm = F.normalize(proto, 2, -1).unsqueeze(0)  # (1, classes, c)
            pred_weight = (pred_proto_norm * proto_norm).sum(-1).unsqueeze(-1)  # (b, classes, 1)
            pred_weight = pred_weight * (pred_weight > 0).float()
            pred_proto = pred_weight * pred_proto + (1 - pred_weight) * proto.unsqueeze(0)  # b, cls, c
        else:
            # base training
            # raw_x = x.clone()
            # b, c, n = raw_x.shape[:]
            b, c, n = point_feat.shape[:]

            pred = self.get_pred(x, proto, use_bg_proto).view(b, proto.shape[0]+1, n)  # (b, bg+13, n)
            pred = F.softmax(pred, 2)  # (b, bg+13, n)

            pred_proto = pred @ point_feat.view(b, c, n).permute(0, 2, 1)
            pred_proto = pred_proto[:,1:,:]  # (b, classes, c). exclude 'bg proto'
            pred_proto_norm = F.normalize(pred_proto, 2, -1)  # (b, classes, c)
            proto_norm = F.normalize(proto, 2, -1).unsqueeze(0)  # (1, classes, c)
            pred_weight = (pred_proto_norm * proto_norm).sum(-1).unsqueeze(-1)  # (b, classes, 1)
            pred_weight = pred_weight * (pred_weight > 0).float()
            pred_proto = pred_weight * pred_proto + (1 - pred_weight) * proto.unsqueeze(0)  # b, cls, c


        return pred_proto


    def get_pred(self, x, proto, use_bg_proto=False):
        ''' cosine similairty between x and proto
        Args: n: number of point. c: feature dim. cls: number of classes.
            x: is the whole batch feature. (b, c, n)
            proto: prototype. (13, 192)
            use_bg_proto: in the base stage, we treat the novel classes as bg class. and use self.bg_proto to classify. but no use in the final test.
        Returns: prediction of x. (b, cls, n)

        '''
        b, c, n = x.size()[:]

        if len(proto.shape[:]) == 3:
            # x: [b, c, n]
            # proto: [b, cls, c]
            if use_bg_proto:
                proto = torch.cat([self.bg_proto.unsqueeze(0).repeat(proto.shape[0],1,1), proto], dim=1) # (b, bg+cls, c)

            cls_num = proto.size(1)
            x = F.normalize(x, p=2, dim=1)
            proto = F.normalize(proto, p=2, dim=-1)  # b, cls, c
            x = x.contiguous().view(b, c, n)  # b, c, n
            pred = proto @ x  # b, cls, n
        elif len(proto.shape[:]) == 2:
            if use_bg_proto:
                proto = torch.cat([self.bg_proto, proto], dim=0) # (bg+cls, c)
            cls_num = proto.size(0)
            x = F.normalize(x, p=2, dim=1)  # l2 norm
            proto = F.normalize(proto, p=2, dim=1)  # l2 norm
            x = x.contiguous().view(b, c, n)  # b, c, n
            proto = proto.unsqueeze(0)  # 1, cls, c
            pred = proto @ x  # b, cls, n
        pred = pred.contiguous().view(b, cls_num, n)  # (b, cls, n)
        return pred * 10 # scaling

    def getFeatures(self, x, segment_label=None):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return:
        segment_feature: features with shape (k+d, m1+m2...). m1 is the number of segment for query 1.
        point_feature: (b, d, n)
        """
        edge_convs, feat_level2 = self.encoder(x)
        feat_level3 = self.base_learner(feat_level2)
        att_feat = self.att_learner(feat_level2)
        feat_level1 = edge_convs[0] # (b, d, n)

        edge_convs = torch.cat(edge_convs, dim=1) # (b, d, n) # to get projection
        semantic_feat = torch.cat((feat_level1, att_feat, feat_level3), dim=1) # (b, d, n)

        # 1. projection via bag of words
        # # add small projection head
        # edge_convs_proj = self.proj(edge_convs) # (b, d, n)

        edge_convs_l2 = F.normalize(edge_convs, p=2, dim=1)  # (b, d, n)
        gp_l2 = F.normalize(self.gp, dim=1, p=2).unsqueeze(0)  # (1, k, d)
        cosine_feat = torch.matmul(gp_l2, edge_convs_l2)  # (b, k, n) feat is bounded within (-1, 1).
        # sharpen
        # print('before sharpen: max: {}, min: {}'.format(cosine_feat.max(), cosine_feat.min()))
        cosine_feat = torch.softmax(10*cosine_feat, dim=1) # (b, k, n). feat_dim=k

        # one-hot
        assignment = torch.argmax(cosine_feat, dim=1)  # (b, n)
        one_hot_feat = F.one_hot(assignment, num_classes=self.gp.shape[0]).transpose(2, 1).float()

        # combine two feature to get point_feat
        point_feat = torch.cat([cosine_feat, semantic_feat], dim=1)  # (b, k+d, n)

        # fusion
        point_feat = self.fusion(point_feat) # (b, d, n)


        return point_feat, semantic_feat, one_hot_feat

    def generate_fake_proto(self, x, y, main_proto, fake_novel=None, post_processing=False):
        ''' only used during training!
        Args: during training
            x: x is the feature of the support set. (b, d, n)
            y: label of x. (b, n). {0,1,2,...} 0 is 'bg' in base stage. but main_proto doesn't count 'bg'!
            main_proto: (n, d)
            fake_novel: fake_novel class ids.
        Returns: l2 normed proto.

        '''
        b, c, n = x.size()[:]  # x is feature
        # get fake novel idx


        tmp_y = y.unsqueeze(1)  # (b, 1, n) # the label set of the support set
        unique_y = list(tmp_y.unique())  # classes exist in the x

        # get fake_novel and fake_context classes. exclude 'bg' in fake_novel and fake_context.
        if fake_novel == None:
            if 0 in unique_y:
                unique_y.remove(0)
            novel_num = len(unique_y) // 2
            fake_novel = random.sample(unique_y, novel_num)  # fake novel classes in the support set.
            for fn in fake_novel:
                unique_y.remove(fn)
            fake_context = unique_y  # fake context classes in this support set

        new_proto = main_proto  # (n_base, d)
        # l2 norm. need l2 norm here!!
        new_proto = new_proto / (torch.norm(new_proto, 2, 1, True) + 1e-12)  # l2 norm
        # input feat l2 norm
        x = x / (torch.norm(x, 2, 1, True) + 1e-12)  # l2 norm (b, d, n)

        # for fake_novel classes, we use the feature average as the classifier.
        for fn in fake_novel:  # if it is fake novel, then its classifier is the prototype of the support set. Otherwise, use the main_proto.
            tmp_mask = (tmp_y == fn).float() # (b, 1, n)
            tmp_feat = (x * tmp_mask).sum(0).sum(-1) / (tmp_mask.sum(0).sum(-1) + 1e-12)  # (d,). proto

            # post processing for gp coding
            if post_processing ==  True:
                tmp_feat = self.post_processing_hard_coding(tmp_feat) # (k,)

            fake_vec = torch.zeros(new_proto.size(0), 1).cuda()  # (n_base, 1)
            fake_vec[fn.long() - 1] = 1
            new_proto = new_proto * (1 - fake_vec) + tmp_feat.unsqueeze(0) * fake_vec


        return new_proto, fake_novel

    def post_processing_hard_coding(self, coding):
        '''
        Args:
            coding: (k,). probability vector
        Returns: (k,) multi-hot vector
        '''
        id_list = torch.argsort(coding, descending=True)
        total_sum = torch.sum(coding)
        acc_sum = 0.
        mask = torch.zeros_like(coding)
        # only keep 0.8 energy
        for id in id_list:
            acc_sum += coding[id]
            mask[id] = torch.tensor(1., dtype=coding.dtype, device=coding.device)
            if acc_sum > self.energy * total_sum:
                break
        # multi-hot label
        coding[mask == 1] = 1
        coding[mask == 0] = 0

        return coding

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






