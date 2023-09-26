import os
import random
import time
# import cv2
import numpy as np
import logging
import argparse
import random
import pickle


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data


from torch.utils.tensorboard import SummaryWriter

from model.capl import mpti_net_Point_GeoAsWeight_v2
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU
from dataloaders.loader import MyPretrainDataset, Testing_Dataset, ValSupp_Dataset
from runs.eval import evaluate_metric_GFS
from util.checkpoint_util import load_pretrain_checkpoint, load_model_checkpoint
import ast
from util.logger import init_logger
from model.dgcnn import DGCNN








def get_class_wise_iou(mean_iou_list, logger):
    '''
    Args:
        mean_iou_list: [[iou_list seed 10], [iou_list seed 20], ...]
        logger:
    Returns:
    '''
    stack_iou = np.stack(mean_iou_list, axis=0) # (5, num_class)
    stack_iou = np.mean(stack_iou, axis=0) # (num_class, )
    num_class = len(mean_iou_list[0])
    for i in range(num_class):
        logger.cprint('class {}, iou over multiple runs: {}'.format(i, stack_iou[i]))




def load_base_class_gp_coding(base_class_gp_coding_path):
    '''
    Args:
        base_class_gp_coding_path:
    Returns: (num_base_cls, k). stack in the learning order
    '''
    with open(base_class_gp_coding_path, 'rb') as f:
        BaseClass_GP_Coding = pickle.load(f)

    base_class_gp_coding = []
    base_classes = sorted(BaseClass_GP_Coding.keys()) # base class name in order
    for cls in base_classes:
        base_class_gp_coding.append(BaseClass_GP_Coding[cls])

    base_class_gp_coding = np.stack(base_class_gp_coding, axis=0) # (cls, k) # stack according to learning order
    base_class_gp_coding = torch.from_numpy(base_class_gp_coding).cuda()
    base_class_gp_coding.requires_grad = False
    print('load base_class_gp_coding done!')

    return base_class_gp_coding



def get_new_proto(val_supp_loader, model, base_num=16, novel_num=5, novel_class_list=None):
    ''' get base and novel protoes. base proto is via eqn.3. novel proto is done by eqn.1
        Args:
            val_supp_loader:
            model:
            base_num:
            novel_num:
            novel_class_list: learning order idx of new classes. [7,8,9,10,11]
            proto_dim: feature dimension of the prototypes
        Returns: proto of all classes (cls, 192)

    '''

    logger.cprint('>>>>>>>>>>>>>>>> Start New Proto Generation >>>>>>>>>>>>>>>>')

    model.eval()
    new_proto_num_epoch = 1  # 1
    with torch.no_grad():
        proto_dim = model.main_proto.shape[1]
        gened_proto_bed = torch.zeros(args.total_classes, proto_dim).cuda()  # empty proto. (21, 512).
        for epoch in range(new_proto_num_epoch):

            new_cls_feat_dict = {cls: [] for cls in novel_class_list}

            for i, (input, target, cls_id) in enumerate(val_supp_loader): # cls_id is idx!!
                input = input.cuda()
                target = target.cuda()
                # get feature
                cls_feat = model.Get_Fg_Feat(x=input, y=target)  # collect class feature. (n, d)
                cls_feat = torch.mean(cls_feat, dim=0, keepdim=True) # add this (1, d)

                new_cls_feat_dict[cls_id[0].item()].append(cls_feat)


            # initialize final proto
            gened_proto = torch.zeros_like(model.main_proto, requires_grad=False) # (cls, d)
            assert gened_proto.device == model.main_proto.device
            # copy base proto
            gened_proto[:base_num, :] = model.main_proto[:base_num, :].detach().clone()
            # get novel proto
            for cls in novel_class_list:
                gened_proto[cls,:] = torch.mean(torch.cat(new_cls_feat_dict[cls], dim=0), dim=0, keepdim=False) # (d,)

            # cnocat with base
            # base_proto = gened_proto[:base_num, :] / (torch.norm(gened_proto[:base_num, :], 2, 1, True) + 1e-12)
            # base_proto = F.normalize(gened_proto[:base_num, :], p=2, dim=1)
            # gened_proto = torch.cat([base_proto, gened_proto[base_num:, :]], 0)  # all the class proto. (base + novel, c)
            gened_proto = F.normalize(gened_proto, p=2, dim=1)  # l2 norm.

            gened_proto_bed = gened_proto_bed + gened_proto
        gened_proto = gened_proto_bed / new_proto_num_epoch

    return gened_proto  # (cls, 192)


def post_processing_hard_coding(coding, energy):
    ''' minor frequency pruning.
    Args:
        coding: (k,). probability vector
        energy: 0.9 / 0.85
    Returns: (k,) multi-hot label
    '''
    id_list = torch.argsort(coding, descending=True)
    total_sum = torch.sum(coding)
    acc_sum = 0.
    mask = torch.zeros_like(coding)
    # only keep 0.8 energy
    for id in id_list:
        acc_sum += coding[id]
        mask[id] = torch.tensor(1., dtype=coding.dtype, device=coding.device)
        if acc_sum > energy * total_sum:
            break
    # multi-hot label
    coding[mask==1] = 1
    coding[mask==0] = 0
    return coding



def collect_base_class_gp_coding_sum(model, train_loader, train_class, energy):
    '''
    Args:
        model:
        train_loader: bs=1, no pc_aug
        train_class:
    Returns: base_class_gp_coding: (num_base, k) cuda
    '''
    model.eval()
    gp_feat_dict = {cls: [] for cls in train_class} # learning order idx. (0,1,2,...)
    gp_feat_num = {cls: [] for cls in train_class}
    base_class_gp_coding = []
    # add bg_class coding
    bg_class_coding = []
    max_len = 2000

    with torch.no_grad():
        for i, (input, target, segment_label) in enumerate(train_loader):
            input = input.cuda() # (1, d, n)
            target = target[0].cuda() # (n,)
            exist_label = torch.unique(target)

            # get gp featufre
            _, _, gp_feat = model.getFeatures(input)
            gp_feat = gp_feat[0] # (k,n)
            # # one-hot
            # assignment = torch.argmax(gp_feat, dim=0) # (n,)
            # gp_feat = F.one_hot(assignment, num_classes=gp_feat.shape[0]).transpose(1,0).float() # (k,n)

            # collect class-wise gp feat:
            for cls in exist_label:
                if cls == 0:
                    # continue
                    mask = target == cls  # (n,)
                    if torch.sum(mask) > 0:
                        tmp_feat = torch.mean(gp_feat[:, mask], dim=1)  # (k,)
                        bg_class_coding.append(tmp_feat)
                    continue


                # get mask
                mask = target == cls #(n,)
                if torch.sum(mask) > 0:
                    tmp_feat = torch.sum(gp_feat[:,mask], dim=1) # (k,)
                    gp_feat_dict[cls.item()-1].append(tmp_feat)
                    gp_feat_num[cls.item()-1].append(torch.sum(mask)) # record counts

        for cls in train_class:
            print('processing {}'.format(cls))
            tmp_feat = torch.sum(torch.stack(gp_feat_dict[cls], dim=0), dim=0) # (k,)
            counts = sum(gp_feat_num[cls])
            tmp_feat = tmp_feat / counts
            tmp_feat = post_processing_hard_coding(tmp_feat, energy=energy) # (k, )
            base_class_gp_coding.append(tmp_feat)

        base_class_gp_coding = torch.stack(base_class_gp_coding, dim=0) # (num_base, k)

        # add bg_coding
        if len(bg_class_coding) > max_len:
            bg_class_coding = random.sample(bg_class_coding, max_len)
        bg_class_coding = torch.mean(torch.stack(bg_class_coding, dim=0), dim=0) # (k, )

    return base_class_gp_coding, bg_class_coding


def collect_new_clsss_gp_coding_sum(new_cls_gp_feat_dict, energy):
    '''
    Args:
        new_cls_gp_feat_dict: {7: [], 8: [], } learning order idx of new class
    Returns:
    '''
    new_classes = sorted(new_cls_gp_feat_dict.keys())
    new_class_gp_coding = []
    for cls in new_classes:
        print('processing {}'.format(cls))
        tmp_feat = torch.sum(torch.cat(new_cls_gp_feat_dict[cls], dim=0), dim=0)
        tmp_feat = tmp_feat / torch.sum(tmp_feat) # probability vector
        tmp_feat = post_processing_hard_coding(tmp_feat, energy=energy) # multi-hot vector

        new_class_gp_coding.append(tmp_feat) # (k,)

    new_class_gp_coding = torch.stack(new_class_gp_coding, dim=0) # (num_new, k)
    return new_class_gp_coding

def get_new_proto_Geo2SemProto(val_supp_loader, model, base_num=16, novel_num=5, novel_class_list=None,
    train_loader_NoAug=None, base_class_coding=None, energy=None):
    ''' get base and novel protoes. base proto is via eqn.3. novel proto is done by eqn.1
        Args:
            val_supp_loader:
            model:
            base_num:
            novel_num:
            novel_class_list: learning order idx of new classes. [7,8,9,10,11]
            proto_dim: feature dimension of the prototypes
            train_loader_NoAug: train_loader_noaug to get gp_coding for base classes
            base_class_coding: (n_base, k)
        Returns: final_proto: fused with geo2sem_proto. (n_cls, d). l2 norm
    '''
    num_all_class = base_num + novel_num
    logger.cprint('>>>>>>>>>>>>>>>> Start New Proto Generation for {} classes >>>>>>>>>>>>>>>>'.format(num_all_class))

    model.eval()
    new_proto_num_epoch = 1  # 1
    with torch.no_grad():
        proto_dim = model.main_proto.shape[1]
        gened_proto_bed = torch.zeros(args.total_classes, proto_dim).cuda()  # empty proto. (21, 512).

        new_cls_feat_dict = {cls: [] for cls in novel_class_list}
        new_cls_gp_feat_dict = {cls: [] for cls in novel_class_list}

        for i, (input, target, cls_id) in enumerate(val_supp_loader): # cls_id is idx!!
            input = input.cuda()
            target = target.cuda()
            # get feature
            cls_feat, cls_gp_feat = model.Get_Fg_Feat(x=input, y=target)  # collect class feature. (n, d)
            cls_feat = torch.mean(cls_feat, dim=0, keepdim=True) # add this (1, d)

            # cls_gp_feat = torch.mean(cls_gp_feat, dim=0, keepdim=True) # (1, k)
            cls_gp_feat = torch.sum(cls_gp_feat, dim=0, keepdim=True)  # (1, k)

            new_cls_feat_dict[cls_id[0].item()].append(cls_feat)
            new_cls_gp_feat_dict[cls_id[0].item()].append(cls_gp_feat)

        # initialize orig proto
        # gened_proto = torch.zeros((num_all_class, proto_dim), device=input.device, requires_grad=False) # (cls, d)
        gened_proto = torch.zeros_like(model.main_proto, requires_grad=False)  # (cls, d)
        assert gened_proto.device == model.main_proto.device
        # copy base proto
        gened_proto[:base_num, :] = model.main_proto[:base_num, :].detach().clone()
        # get novel proto
        for cls in novel_class_list:
            gened_proto[cls,:] = torch.mean(torch.cat(new_cls_feat_dict[cls], dim=0), dim=0, keepdim=False) # (d,)

        gened_proto = F.normalize(gened_proto, dim=1, p=2) # (n_cls, d)


        # collect new-class gp_coding
        novel_class_coding = collect_new_clsss_gp_coding_sum(new_cls_gp_feat_dict, energy=energy) # (n_new, k)

        # # get geo2sem proto
        # geo2sem_proto = model.Geo2SemProto(torch.cat([base_class_coding, novel_class_coding], dim=0))  # (n, d)
        # # fusion
        # ori_new_proto_new = model.Fusion_With_Geo2SemProto(geo2sem_proto, gened_proto)  # (n, d)

        # # get geo2sem proto and fusion with ori_proto
        # geo2sem_proto = model.Geo2SemProto(torch.cat([gp_coding_base, gp_coding_new], dim=0)) # (n, d)
        # final_proto = model.Fusion_With_Geo2SemProto(geo2sem_proto=geo2sem_proto, orig_proto=gened_proto, base_num=base_num) # (n, d)
        # final_proto = F.normalize(final_proto, p=2, dim=1) # l2 norm

    return gened_proto, novel_class_coding  # (cls, 192)



def main(argss, basis_path):
    global args
    args = argss
    global logger, writer
    logger = init_logger(args.save_path, args)
    writer = SummaryWriter(args.save_path)


    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # load basis:
    with open(basis_path, 'rb') as f:
        basis = pickle.load(f) # (n, d)
    basis = torch.from_numpy(basis).cuda()
    basis.requires_grad = False
    print(basis.shape)
    print('load basis done!')


    # ------------------- define validation dataloader --------------------.
    if args.dataset == 's3dis':
        from dataloaders.s3dis import S3DISDataset
        DATASET = S3DISDataset(args.cvfold, args.testing_data_path)
    elif args.dataset == 'scannet':
        from dataloaders.scannet import ScanNetDataset
        DATASET = ScanNetDataset(args.cvfold, args.testing_data_path)
    else:
        raise NotImplementedError('Unknown dataset %s!' % args.dataset)

    train_class_names = sorted(DATASET.train_classes)  # they are sorted by the class name order.
    test_class_names = sorted(DATASET.test_classes)  # sorted by class_name order.

    all_learning_order = sorted(DATASET.train_classes)
    all_learning_order.extend(test_class_names)  # learning order of all classes
    all_class_names = sorted(all_learning_order)  # all classes sorted by class name order.

    # get learning order idx of testing classes
    test_learning_order_idx = []
    for i in test_class_names:
        test_learning_order_idx.append(all_learning_order.index(i))

    print('testing classes : {}'.format(all_class_names))  # make sure CLASSES are in order before sending them into dataset.

    test_CLASS2SCANS = {c: DATASET.class2scans[c] for c in all_class_names}  # only use the train classes in class2scan

    VALID_DATASET = Testing_Dataset(args.testing_data_path, all_class_names, all_learning_order, test_CLASS2SCANS,
                                    mode='test',
                                    num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                    pc_augm=False)

    val_loader = torch.utils.data.DataLoader(VALID_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False,
                                             drop_last=False)

    # get novel class dataset: support set. label is binary only.
    seed_list = [10, 20, 30, 40, 50]
    val_supp_loader_list = []
    for seed in seed_list:
        val_supp_data = ValSupp_Dataset(data_path=args.data_path, dataset_name=args.dataset, cvfold=args.cvfold,
                                        k_shot=args.k_shot, mode='test',
                                        num_point=args.pc_npts, pc_attribs=args.pc_attribs, pc_augm=False,
                                        pc_augm_config=None, seed=seed, learning_order=all_learning_order)

        val_supp_loader = torch.utils.data.DataLoader(val_supp_data, batch_size=1, shuffle=False,
                                                      num_workers=args.n_workers, drop_last=False)
        val_supp_loader_list.append(val_supp_loader)




    # --------------------------- TRAINING ----------------------------------- Init datasets, dataloaders, and writer
    PC_AUGMENT_CONFIG = {'scale': args.pc_augm_scale,
                         'rot': args.pc_augm_rot,
                         'mirror_prob': args.pc_augm_mirror_prob,
                         'jitter': args.pc_augm_jitter
                         }

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
    print('training classes : {}'.format(CLASSES))
    train_CLASS2SCANS = {c: DATASET.class2scans[c] for c in CLASSES}  # only use the train classes in class2scan

    train_data = MyPretrainDataset(args.data_path, train_class_names, train_CLASS2SCANS, mode='train',
                                   num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                   pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG)

    logger.cprint('=== Pre-train Dataset (classes: {0}) | Train: {1} blocks | Valid: {2} blocks ==='.format(
        train_class_names, len(train_data), 0))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=args.n_workers,
                                               shuffle=True,
                                               drop_last=True)

    ## add train loader without augmentation
    train_data_NoAug = MyPretrainDataset(args.data_path, train_class_names, train_CLASS2SCANS, mode='train',
                                         num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                         pc_augm=False, pc_augm_config=None)
    train_loader_NoAug = torch.utils.data.DataLoader(train_data_NoAug, batch_size=1, num_workers=args.n_workers,
                                                     shuffle=True, drop_last=False)





    model = mpti_net_Point_GeoAsWeight_v2(classes=len(all_class_names), criterion=criterion, args=args, base_num=len(train_class_names),
                                       gp=basis, energy=args.energy)
    model = model.cuda()

    # define optimizer
    optimizer = torch.optim.Adam(
                    [
                     {'params': model.encoder.parameters(), 'lr':0.1*args.base_lr},
                     {'params': model.base_learner.parameters()},
                     {'params': model.att_learner.parameters()},
                     {'params': model.main_proto},
                     {'params': model.bg_proto},
                        {'params': model.fusion.parameters()},
                     ],
                        lr=args.base_lr)

    # set learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size,
                                                  gamma=args.gamma)




    # load pretrain weight of feature extractor:
    if args.use_pretrain_weight:
        logger.cprint('----- loading pretrain weight of feature extractor --------')
        model = load_pretrain_checkpoint(model, args.pretrain_checkpoint_path)
        # model = load_model_checkpoint(model, args.model_checkpoint_path, mode='test')

    # load weight of the whole model. for evaluation
    if args.only_evaluate:
        logger.cprint('--------- loading weight for evaluation -----------')
        model = load_model_checkpoint(model, args.model_checkpoint_path, mode='test')





    if args.only_evaluate:
        mean_mean_mIoU = 0
        mean_base_mIoU = 0
        mean_novel_mIoU = 0
        mean_hm_mIoU = 0
        mean_iou_list = []
        for val_supp_loader in val_supp_loader_list:
            if os.path.exists(os.path.join(args.save_path, 'base_class_gp_coding_energy={}.pth'.format(args.energy))):
                base_class_coding = torch.load(os.path.join(args.save_path, 'base_class_gp_coding_energy={}.pth'.format(args.energy)))
            else:
                # recompute
                logger.cprint('-------------------- recompute base_class_coding, energy={} --------------------'.format(args.energy))
                base_class_coding, _ = collect_base_class_gp_coding_sum(model, train_loader_NoAug, train_class=np.arange(len(train_class_names)), energy=args.energy)
                torch.save(base_class_coding, os.path.join(args.save_path, 'base_class_gp_coding_energy={}.pth'.format(args.energy)))



            gened_proto, novel_class_coding = get_new_proto_Geo2SemProto(val_supp_loader, model, novel_num=len(test_class_names), base_num=len(train_class_names), \
                                        novel_class_list=test_learning_order_idx, base_class_coding=base_class_coding, energy=args.energy)
            mean_iou, base_iou, novel_iou, hm_iou, iou_list = validate(val_loader, model, novel_num=len(test_class_names),
                                                    base_num=len(train_class_names),
                                                    gened_proto=gened_proto.clone(),
                                                     all_classes=all_class_names, novel_classes=test_class_names,
                                                     all_learning_order=all_learning_order, basis=basis, geo2sem_proto=None,
                                                     base_class_coding=base_class_coding, novel_class_coding=novel_class_coding)

            mean_mean_mIoU += mean_iou
            mean_base_mIoU += base_iou
            mean_novel_mIoU += novel_iou
            mean_hm_mIoU += hm_iou
            mean_iou_list.append(iou_list)
        mIoU_val = mean_mean_mIoU / len(val_supp_loader_list)
        base_mIoU = mean_base_mIoU / len(val_supp_loader_list)
        novel_mIoU = mean_novel_mIoU / len(val_supp_loader_list)
        hm_mIoU = mean_hm_mIoU / len(val_supp_loader_list)
        logger.cprint('Eval result: Final mIoU: {}, BASE: {}, NOVEL: {}, hm_mIoU: {}'.format(mIoU_val, base_mIoU, novel_mIoU, hm_mIoU))
        # print class-wise mean iou:
        get_class_wise_iou(mean_iou_list, logger)


        exit(0)



    max_iou = 0.
    filename = 'capl.pth'
    filename_after100 = 'capl.pth'
    max_hm =0.
    hm_filename = 'hm.pth'

    for epoch in range(args.start_epoch, args.epochs):
        # # epoch_log = epoch + 1
        # every 5 epochs, re-estimate base_class_coding
        if epoch % 5 == 0 or epoch == 0:
            base_class_coding, bg_class_coding = collect_base_class_gp_coding_sum(model, train_loader_NoAug, train_class=np.arange(len(train_class_names)), energy=args.energy)  # (n_class, k) cuda

        train(train_loader, model, optimizer, epoch, lr_scheduler, base_class_coding=base_class_coding, bg_class_coding=bg_class_coding)


        if args.evaluate and (epoch+1) % 5 == 0:
            mean_mean_mIoU = 0
            mean_base_mIoU = 0
            mean_novel_mIoU = 0
            mean_hm_mIoU = 0
            val_supp_loader_list = val_supp_loader_list[0:1]
            for val_supp_loader in val_supp_loader_list:

                gened_proto, novel_class_coding = get_new_proto_Geo2SemProto(val_supp_loader, model, novel_num=len(test_class_names), base_num=len(train_class_names),
                                            novel_class_list=test_learning_order_idx, train_loader_NoAug=train_loader_NoAug, base_class_coding=base_class_coding,
                                                                             energy=args.energy)  # get base + novel protoes. skip eqn.3. because we don't have base annotation is novel stage! (13, 192)
                mean_iou, base_iou, novel_iou, hm_iou, _ = validate(val_loader, model,
                                                         novel_num=len(test_class_names),
                                                         base_num=len(train_class_names),
                                                         gened_proto=gened_proto.clone(),
                                                         all_classes=all_class_names, novel_classes=test_class_names,
                                                         all_learning_order=all_learning_order, basis=basis, geo2sem_proto=None,
                                                         base_class_coding=base_class_coding, novel_class_coding=novel_class_coding)
                mean_mean_mIoU += mean_iou
                mean_base_mIoU += base_iou
                mean_novel_mIoU += novel_iou
                mean_hm_mIoU += hm_iou
            mIoU_val = mean_mean_mIoU / len(val_supp_loader_list)
            base_mIoU = mean_base_mIoU / len(val_supp_loader_list)
            novel_mIoU = mean_novel_mIoU / len(val_supp_loader_list)
            hm_mIoU = mean_hm_mIoU / len(val_supp_loader_list)
            logger.cprint('Epoch: {}, Final mIoU: {}, BASE: {}, NOVEL: {}, hm: {}'.format(epoch, mIoU_val, base_mIoU, novel_mIoU, hm_mIoU))

            # writer.add_scalar('loss_val', loss_val, epoch_log)
            writer.add_scalar('Val/mIoU_val', mIoU_val, epoch)
            writer.add_scalar('Val/base_mIoU', base_mIoU, epoch)
            writer.add_scalar('Val/novel_mIoU', novel_mIoU, epoch)
            writer.add_scalar('Val/hm_mIoU', hm_mIoU, epoch)




            if mIoU_val > max_iou and epoch < 100: # debug. not used for inference
                max_iou = mIoU_val # mean iou.
                if os.path.exists(filename):
                    os.remove(filename)
                filename = args.save_path + '/train_epoch_' + str(epoch) + '_'+ str(max_iou)+'_Base_'+str(base_mIoU)+'_Novel_'+str(novel_mIoU)+'.pth'
                logger.cprint('Saving best checkpoint to: ' + filename)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'max_iou': max_iou}, filename)
                # save base class gp coding
                torch.save(base_class_coding, os.path.join(args.save_path, 'base_class_gp_coding_energy={}.pth'.format(args.energy)))

            if mIoU_val > max_iou and epoch >= 100:
                max_iou = mIoU_val # mean iou.
                if os.path.exists(filename_after100):
                    os.remove(filename_after100)
                filename_after100 = args.save_path + '/train_epoch_' + str(epoch) + '_'+ str(max_iou)+'_Base_'+str(base_mIoU)+'_Novel_'+str(novel_mIoU)+'_hm_'+str(hm_mIoU)+'.pth'
                logger.cprint('Saving best checkpoint to: ' + filename_after100)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'max_iou': max_iou}, filename_after100)
                # save base class gp_coding
                torch.save(base_class_coding, os.path.join(args.save_path, 'base_class_gp_coding_energy={}.pth'.format(args.energy)))

            # hm evaluation
            if hm_mIoU > max_hm:
                max_hm = hm_mIoU # mean iou.
                if os.path.exists(hm_filename):
                    os.remove(hm_filename)
                hm_filename = args.save_path + '/train_hm_epoch_' + str(epoch) + '_'+ str(max_iou)+'_Base_'+str(base_mIoU)+'_Novel_'+str(novel_mIoU)+'_hm_'+str(hm_mIoU)+'.pth'
                logger.cprint('Saving best checkpoint to: ' + hm_filename)
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'max_iou': max_hm}, hm_filename)
                # save base class gp_coding
                torch.save(base_class_coding, os.path.join(args.save_path, 'hm_base_class_gp_coding_energy={}.pth'.format(args.energy)))





def train(train_loader, model, optimizer, epoch, lr_scheduler, base_class_coding=None, bg_class_coding=None):
    '''
    Args:
        train_loader:
        model:
        optimizer:
        epoch: current epoch
        lr_scheduler:
        base_class_coding: (n_base, k)

    Returns:

    '''
    torch.cuda.empty_cache()

    accuracy_meter = AverageMeter()
    loss_meter = AverageMeter()
    loss_ce_meter = AverageMeter()
    loss_consistency_meter = AverageMeter()
    loss_kd_meter = AverageMeter()
    loss_contrast_meter = AverageMeter()
    loss_ce_geo2sem_meter = AverageMeter()
    gp_acc_meter = AverageMeter()

    model.train()

    for i, (input, target, segment_label) in enumerate(train_loader):

        current_iter = epoch * len(train_loader) + i + 1

        input = input.cuda() # (b, d, 2048)
        target = target.cuda() # (b, 2048)
        segment_label = segment_label.cuda()

        output, loss_ce = model(x=input, y=target, segment_label=segment_label,
                                base_class_coding=base_class_coding, bg_class_coding=bg_class_coding) # output: pred_label. (b, 2048)

        loss = loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = torch.eq(output, target).sum().item()  # including background
        accuracy = correct / (input.shape[0] * input.shape[-1])


        accuracy_meter.update(accuracy)
        loss_meter.update(loss.item(), 1)
        loss_ce_meter.update(loss_ce.item(), 1)


        if (i + 1) % args.print_freq == 0:
            logger.cprint('Epoch: [{}/{}][{}/{}] '                       
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Loss_ce {loss_ce_meter.val:.4f} ({loss_ce_meter.avg:.4f})'
                        'loss_geosem {loss_ce_geo2sem_meter.val:.4f} ({loss_ce_geo2sem_meter.avg:.4f})'
                        'Accuracy {accuracy:.4f} ({accuracy_meter.avg:.4f}).'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                          loss_meter=loss_meter, loss_ce_meter=loss_ce_meter, loss_ce_geo2sem_meter=loss_ce_geo2sem_meter,
                                                          accuracy=accuracy, accuracy_meter=accuracy_meter))

    lr_scheduler.step()

    acc = accuracy_meter.avg
    logger.cprint('Train result at epoch [{}/{}]: acc {:.4f}.'.format(epoch, args.epochs, acc))

    # tensorboard
    writer.add_scalar('Train/loss', loss_meter.avg, epoch)
    writer.add_scalar('Train/loss_ce', loss_ce_meter.avg, epoch)
    writer.add_scalar('Train/loss_kd', loss_kd_meter.avg, epoch)
    writer.add_scalar('Train/accuracy', acc, epoch)
    writer.add_scalar('Train/loss_contrast', loss_contrast_meter.avg, epoch)
    writer.add_scalar('Train/gp_acc', gp_acc_meter.avg, epoch)



def validate(val_loader, model, novel_num, base_num, gened_proto, all_classes, novel_classes, all_learning_order, basis=None,
             geo2sem_proto=None, base_class_coding=None, novel_class_coding=None):
    '''
    Args:
        val_supp_loader:
        val_loader:
        model:
        criterion:
        novel_num:
        base_num:
        gened_proto:
        all_classes: [0,1,2,3,4,5,7,8,9,....]
        novel_classes: novel class names
        all_learning_order: learning order of all classes
        basis: basis (n, d). gpu tensor. no grad.
    Returns:

    '''
    torch.cuda.empty_cache() 

    if len(all_learning_order) > 13:
        use_scannet = True
        print('use scannet, skip class 0!')
    else:
        use_scannet = False

    logger.cprint('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()

    model.eval()
    end = time.time()

    pred_labels_list = []
    gt_labels_list = []
    pred_labels_list_geo2sem = []
    gp_acc_meter = AverageMeter()
    gp_novel_acc_meter = AverageMeter()

    with torch.no_grad():
        gened_proto = gened_proto.unsqueeze(0).repeat(8, 1, 1) # (b, 13, 192)
        for i, (input, target, segment_label) in enumerate(val_loader):
            data_time.update(time.time() - end)
            input = input.cuda()
            target = target.cuda() # (1, 2048)
            segment_label = segment_label.cuda()
            output, gp_acc, gp_novel_acc = model(x=input, y=target, eval_model=True, gen_proto=False, gened_proto=gened_proto,
                           base_class_coding=base_class_coding, novel_class_coding=novel_class_coding) # logits (b, cls, n)

            query_pred = torch.argmax(output, dim=1, keepdim=False) # (b, n)
            pred_labels_list.append(query_pred)
            gt_labels_list.append(target)


            gp_acc_meter.update(gp_acc, 1)
            gp_novel_acc_meter.update(gp_novel_acc, 1)

        # get iou
        mean_iou, base_iou, novel_iou, hm_iou, iou_list = evaluate_metric_GFS(logger, pred_labels_list, gt_labels_list,
                                                            all_classes, novel_classes, all_learning_order, scannet=use_scannet) # need to change the class order.


    logger.cprint('---------- gp acc: {}, gp_novel_acc: {} ----------'.format(gp_acc_meter.avg, gp_novel_acc_meter.avg))
    return mean_iou, base_iou, novel_iou, hm_iou, iou_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')

    parser.add_argument('--train_gpu', default=[0]) # doesn't use
    parser.add_argument('--batch_size_val', type=int, default=1)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--manual_seed', type=int, default=321)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--save_path', type=str, default='log_s3dis/S0_K5/debug')
    parser.add_argument('--start_val_epoch', type=int, default=25)
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--ngpus_per_node', type=int, default=1)
    # my
    # data
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'])

    # data
    parser.add_argument('--dataset', type=str, default='s3dis', help='Dataset name: s3dis|scannet')
    parser.add_argument('--cvfold', type=int, default=0, help='Fold left-out for testing in leave-one-out setting '
                                                              'Options:{0,1}')
    parser.add_argument('--data_path', type=str,
                        default='/home/yating/Documents/3d_segmentation/GFS_pcd_seg/datasets/S3DIS_Area6AsTest_SP/blocks_bs1.0_s1.0',
                        help='Directory to the source data')
    parser.add_argument('--testing_data_path', type=str,
                        default='/home/yating/Documents/3d_segmentation/GFS_pcd_seg/datasets/S3DIS_Area6AsTest_SP/blocks_bs1.0_s1.0_test')
    parser.add_argument('--total_classes', type=int, default=13, help='number of classes to be evaluate in the gfs')

    # model weight
    parser.add_argument('--use_pretrain_weight', action='store_true',
                        help='whether use pretrain weight of the feature extractor')
    parser.add_argument('--pretrain_checkpoint_path', type=str,
                        default='/home/yating/Documents/3d_segmentation/GFS_pcd_seg/mpti/log_s3dis/log_pretrain_s3dis_S0_LongTail/',
                        help='Path to the checkpoint of pre model for resuming')
    parser.add_argument('--model_checkpoint_path', type=str, default='log_s3dis/S0_K5/train_epoch_35_0.3247954127747135_Base_0.4056141974051477_Novel_0.23050683070587352.pth',
                        help='Path to the checkpoint of model for resuming')


    # optimization
    parser.add_argument('--batch_size', type=int, default=16, help='Number of samples/tasks in one batch')
    parser.add_argument('--n_workers', type=int, default=16, help='number of workers to load data')
    parser.add_argument('--n_iters', type=int, default=100, help='number of iterations/epochs to train')
    parser.add_argument('--step_size', type=int, default=50, help='Iterations of learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')


    # few-shot episode setting
    parser.add_argument('--k_shot', type=int, default=5, help='Number of samples/shots for each class: 1|5')

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
                        help='it incorporate attention learner')  # set to True by default


    parser.add_argument('--seed', default=321, type=int, help='seed')
    parser.add_argument('--only_evaluate', action='store_true', default=False)
    parser.add_argument('--basis_path', type=str, default='log_s3dis/S0_K5/GlobalKmeans_EdgeConv123_cnt=100_energy=095_SVDReconstruct.pkl', help='path of basis')
    parser.add_argument('--base_class_gp_coding_path', type=str, default='log_s3dis/S0_K5/BaseClass_gp.pkl', help='path of base_class_gp_coding_path')
    parser.add_argument('--energy', type=float, default=0.9, help='frequency limit in alg.1. must <= 1!!')
    parser.add_argument('--eval_weight', type=float, default=1., help='beta weight for re-weighting. validation=1., testing > 1.')
    args = parser.parse_args()

    args.edgeconv_widths = ast.literal_eval(args.edgeconv_widths)
    args.dgcnn_mlp_widths = ast.literal_eval(args.dgcnn_mlp_widths)
    args.base_widths = ast.literal_eval(args.base_widths)
    args.pc_in_dim = len(args.pc_attribs)

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    assert args.energy <= 1
    main(args, basis_path=args.basis_path)
