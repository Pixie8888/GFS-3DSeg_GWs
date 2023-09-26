""" Attetion-aware Multi-Prototype Transductive Inference for Few-shot 3D Point Cloud Semantic Segmentation [Our method]

Author: Zhao Na, 2020
"""
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from runs.eval import test_few_shot, evaluate_metric_GFS
from dataloaders.loader import MyDataset, MyTestDataset, batch_test_task_collate, Testing_Dataset, ValSupp_Dataset, MyPretrainDataset
from models.mpti_learner import MPTILearner
from utils.cuda_util import cast_cuda
from utils.logger import init_logger
from gfs import load_base_protoes
import numpy as np
import pickle
import ast
import argparse
import os

def train(args):
    logger = init_logger(args.log_dir, args)

    # os.system('cp models/mpti_learner.py %s' % (args.log_dir))
    # os.system('cp models/mpti.py %s' % (args.log_dir))
    # os.system('cp models/dgcnn.py %s' % (args.log_dir))

    # init model and optimizer
    MPTI = MPTILearner(args)

    #Init datasets, dataloaders, and writer
    PC_AUGMENT_CONFIG = {'scale': args.pc_augm_scale,
                         'rot': args.pc_augm_rot,
                         'mirror_prob': args.pc_augm_mirror_prob,
                         'jitter': args.pc_augm_jitter
                         }

    TRAIN_DATASET = MyDataset(args.data_path, args.dataset, cvfold=args.cvfold, num_episode=args.n_iters,
                              n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
                              phase=args.phase, mode='train',
                              num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                              pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG)

    # VALID_DATASET = MyTestDataset(args.data_path, args.dataset, cvfold=args.cvfold,
    #                               num_episode_per_comb=args.n_episode_test,
    #                               n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
    #                               num_point=args.pc_npts, pc_attribs=args.pc_attribs)
    # VALID_CLASSES = list(VALID_DATASET.classes)

    TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size=1, collate_fn=batch_test_task_collate)
    # VALID_LOADER = DataLoader(VALID_DATASET, batch_size=1, collate_fn=batch_test_task_collate)

    WRITER = SummaryWriter(log_dir=args.log_dir)



    # define validation data

    # -------- get testing dataset
    if args.dataset == 's3dis':
        from dataloaders.s3dis import S3DISDataset
        test_DATASET = S3DISDataset(args.cvfold, args.testing_data_path)
    elif args.dataset == 'scannet':
        from dataloaders.scannet import ScanNetDataset
        test_DATASET = ScanNetDataset(args.cvfold, args.testing_data_path)
    else:
        raise NotImplementedError('Unknown dataset %s!' % args.dataset)

    base_classes = np.sort(test_DATASET.train_classes)
    novel_classes = np.sort(test_DATASET.test_classes)

    all_classes = test_DATASET.train_classes
    all_classes.extend(test_DATASET.test_classes)
    all_classes = sorted(all_classes)
      # make sure CLASSES are in name order before sending them into dataset.

    test_CLASS2SCANS = {c: test_DATASET.class2scans[c] for c in all_classes}

    VALID_DATASET = Testing_Dataset(args.testing_data_path, all_classes, test_CLASS2SCANS, mode='test',
                                    num_point=args.pc_npts, pc_attribs=args.pc_attribs, pc_augm=False)

    print('GFS testing classes : {}, length: {}, data path: {}'.format(all_classes, len(VALID_DATASET), args.testing_data_path))

    VALID_LOADER = DataLoader(VALID_DATASET, batch_size=1, num_workers=args.n_workers, shuffle=False, drop_last=False)


    # ------------------ novel class training data ----------------------
    val_supp_data = ValSupp_Dataset(data_path=args.data_path, dataset_name=args.dataset, cvfold=args.cvfold,
                                    k_shot=args.k_shot, mode='test',
                                    num_point=args.pc_npts, pc_attribs=args.pc_attribs, pc_augm=False,
                                    pc_augm_config=None, seed=10)

    val_supp_loader = torch.utils.data.DataLoader(val_supp_data, batch_size=1, shuffle=False,
                                                  num_workers=args.n_workers, drop_last=False)
    logger.cprint('novel class training data path: {}, length: {}'.format(val_supp_data.save_path, len(val_supp_data)))


    # --------------- base class prototype data: base_dataset ---------------------.
    if args.dataset == 's3dis':
        from dataloaders.s3dis import S3DISDataset
        base_DATASET = S3DISDataset(args.cvfold, args.data_path)  # clean class2scan
    elif args.dataset == 'scannet':
        from dataloaders.scannet import ScanNetDataset
        base_DATASET = ScanNetDataset(args.cvfold, args.data_path)
    else:
        raise NotImplementedError('Unknown dataset %s!' % args.dataset)


    base_CLASS2SCANS = {c: base_DATASET.class2scans[c] for c in base_classes}  # only select the meta-train class2scane


    base_dataset = MyPretrainDataset(args.data_path, base_classes, base_CLASS2SCANS, mode='train',
                                               num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                                               pc_augm=False)
    logger.cprint('train class: {}, len: {}'.format(base_classes, len(base_dataset)))
    base_dataloader = DataLoader(base_dataset, batch_size=1, num_workers=args.n_workers, shuffle=False, drop_last=False)


    # # --- few-shot validation
    # VALID_DATASET_fs = MyTestDataset(args.data_path, args.dataset, cvfold=args.cvfold,
    #                               num_episode_per_comb=args.n_episode_test,
    #                               n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
    #                               num_point=args.pc_npts, pc_attribs=args.pc_attribs)
    # VALID_CLASSES = list(VALID_DATASET_fs.classes)
    #
    # VALID_LOADER_fs = DataLoader(VALID_DATASET_fs, batch_size=1, collate_fn=batch_test_task_collate)


    # -------------- meta-train --------------
    best_iou = 0
    for batch_idx, (data, sampled_classes) in enumerate(TRAIN_LOADER):

        if torch.cuda.is_available():
            data = cast_cuda(data)

        loss, accuracy = MPTI.train(data)

        logger.cprint('==[Train] Iter: %d | Loss: %.4f |  Accuracy: %f  ==' % (batch_idx, loss, accuracy))
        WRITER.add_scalar('Train/loss', loss, batch_idx)
        WRITER.add_scalar('Train/accuracy', accuracy, batch_idx)

        if (batch_idx+1) % args.eval_interval == 0:

            # # normal validation
            # valid_loss, mean_IoU = test_few_shot(VALID_LOADER_fs, MPTI, logger, VALID_CLASSES)
            # logger.cprint('\n=====[VALID] Loss: %.4f | Mean IoU: %f  =====\n' % (valid_loss, mean_IoU))
            # WRITER.add_scalar('Valid/loss', valid_loss, batch_idx)
            # WRITER.add_scalar('Valid/meanIoU', mean_IoU, batch_idx)

            logger.cprint('evaluation: All_classes: {}, base_classes: {}, novel_classes: {}'.format(all_classes, base_classes, novel_classes))
            pred_labels_list, gt_labels_list, cls_proto_dict = MPTI.test_gfs(base_dataloader, val_supp_loader, VALID_LOADER, base_classes, novel_classes, all_classes,
                 k_connect=args.k_connect, log_dir=args.log_dir, iter=batch_idx+1)

            # get iou
            mean_iou, base_iou, novel_iou, hm, _ = evaluate_metric_GFS(logger, pred_labels_list, gt_labels_list, all_classes, novel_classes)

            logger.cprint('\n=====[VALID] Mean IoU: %f, base_iou: %f, novel_iou: %f  =====\n' % (mean_iou, base_iou, novel_iou))
            WRITER.add_scalar('Valid/mean_iou', mean_iou, batch_idx)
            WRITER.add_scalar('Valid/base_iou', base_iou, batch_idx)
            WRITER.add_scalar('Valid/novel_iou', novel_iou, batch_idx)

            if mean_iou > best_iou:
                best_iou = mean_iou
                logger.cprint('*******************Model Saved*******************')
                save_dict = {'iteration': batch_idx + 1,
                             'model_state_dict': MPTI.model.state_dict(),
                             'optimizer_state_dict': MPTI.optimizer.state_dict(),
                             # 'loss': valid_loss,
                             'IoU': best_iou
                             }
                torch.save(save_dict, os.path.join(args.log_dir, 'checkpoint.tar'))
                # save base proto
                with open(os.path.join(args.log_dir, 'base_proto.pkl'), 'wb') as f:
                    pickle.dump(cls_proto_dict, f)

            # # save every 2k
            # save_dict = {'iteration': batch_idx + 1,
            #              'model_state_dict': MPTI.model.state_dict(),
            #              'optimizer_state_dict': MPTI.optimizer.state_dict(),
            #              # 'loss': valid_loss,
            #              'IoU': best_iou
            #              }
            # torch.save(save_dict, os.path.join(args.log_dir, 'checkpoint_{}.tar'.format(batch_idx+1)))

    WRITER.close()