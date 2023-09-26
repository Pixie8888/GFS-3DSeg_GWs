""" Data Loader for Generating Tasks

Author: Zhao Na, 2020
"""
import os
import random
import math
import glob
import numpy as np
import h5py as h5
import transforms3d
from itertools import  combinations

import torch
from torch.utils.data import Dataset
import pickle


def sample_K_pointclouds(data_path, num_point, pc_attribs, pc_augm, pc_augm_config,
                         scan_names, sampled_class, sampled_classes, is_support=False):
    '''sample K pointclouds and the corresponding labels for one class (one_way)'''
    ptclouds  = []
    labels = []
    segment_labels = []
    for scan_name in scan_names:
        ptcloud, label, segment_label = sample_pointcloud(data_path, num_point, pc_attribs, pc_augm, pc_augm_config,
                                           scan_name, sampled_classes, sampled_class, support=is_support) # (2048, 9) (2048, 1)
        ptclouds.append(ptcloud)
        labels.append(label)
        segment_labels.append(segment_label)

    ptclouds = np.stack(ptclouds, axis=0) # (Ns, 2048, 9)
    labels = np.stack(labels, axis=0) # (Ns, 2048)
    segment_labels = np.stack(segment_labels, axis=0) # (Ns, 2048)

    return ptclouds, labels, segment_labels


def sample_pointcloud(data_path, num_point, pc_attribs, pc_augm, pc_augm_config, scan_name,
                      sampled_classes, sampled_class=0, support=False, random_sample=False, use_all_classes=False):
    '''
    Args:
        data_path:
        num_point:
        pc_attribs:
        pc_augm:
        pc_augm_config:
        scan_name:
        sampled_classes:
        sampled_class:
        support:
        random_sample:
        use_all_classes: in base stage of mpti (pretrain+meta-train), we keep a 'bg' class and set its idx 0. In novel stage and testing stage, there is no reservation of 'bg'!

    Returns:

    '''
    sampled_classes = list(sampled_classes) # pre-train: train classes.  train: 2-way classes name
    # # make sure sampled_classes follows order
    # sampled_classes = sorted(sampled_classes)

    data = np.load(os.path.join(data_path, 'data', '%s.npy' %scan_name))
    N = data.shape[0] #number of points in this scan (BLOACK)

    if random_sample:
        sampled_point_inds = np.random.choice(np.arange(N), num_point, replace=(N < num_point)) # random sample 2048 points in this block
    else:
        # If this point cloud is for support/query set, make sure that the sampled points contain target class
        valid_point_inds = np.nonzero(data[:,6] == sampled_class)[0]  # indices of points belonging to the sampled class

        if N < num_point:
            sampled_valid_point_num = len(valid_point_inds)
        else:
            valid_ratio = len(valid_point_inds)/float(N)
            sampled_valid_point_num = int(valid_ratio*num_point)

        sampled_valid_point_inds = np.random.choice(valid_point_inds, sampled_valid_point_num, replace=False)
        sampled_other_point_inds = np.random.choice(np.arange(N), num_point-sampled_valid_point_num,
                                                    replace=(N<num_point))
        sampled_point_inds = np.concatenate([sampled_valid_point_inds, sampled_other_point_inds])

    data = data[sampled_point_inds]
    xyz = data[:, 0:3]
    rgb = data[:, 3:6]


    xyz_min = np.amin(xyz, axis=0)
    xyz -= xyz_min
    if pc_augm:
        xyz = augment_pointcloud(xyz, pc_augm_config)
    if 'XYZ' in pc_attribs:
        xyz_min = np.amin(xyz, axis=0)
        XYZ = xyz - xyz_min
        xyz_max = np.amax(XYZ, axis=0)
        XYZ = XYZ/xyz_max

    ptcloud = []
    if 'xyz' in pc_attribs: ptcloud.append(xyz)
    if 'rgb' in pc_attribs: ptcloud.append(rgb/255.)
    if 'XYZ' in pc_attribs: ptcloud.append(XYZ)
    ptcloud = np.concatenate(ptcloud, axis=1) # (2048, 9)

    # get labels
    labels = data[:, 6].astype(np.int)
    if use_all_classes == False:
        if support:
            groundtruth = labels==sampled_class # binary label
        else:
            groundtruth = np.zeros_like(labels) # labels that only availabel in the given classes
            for i, label in enumerate(labels):
                if label in sampled_classes:
                    groundtruth[i] = sampled_classes.index(label)+1
    else:
        if support:
            groundtruth = labels==sampled_class # binary label
        else:
            groundtruth = np.zeros_like(labels) # labels that only availabel in the given classes
            for i, label in enumerate(labels):
                if label in sampled_classes:
                    groundtruth[i] = sampled_classes.index(label) # no reservation of 'bg' class
            assert groundtruth.max() <= max(sampled_classes)

    # get segment label. not used
    if data.shape[1] == 8:
        segment_label = data[:, 7] # (n,)
    else:
        segment_label = np.zeros(data.shape[0], dtype=data.dtype)

    return ptcloud, groundtruth, segment_label


def augment_pointcloud(P, pc_augm_config):
    """" Augmentation on XYZ and jittering of everything """
    M = transforms3d.zooms.zfdir2mat(1)
    if pc_augm_config['scale'] > 1:
        s = random.uniform(1 / pc_augm_config['scale'], pc_augm_config['scale'])
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
    if pc_augm_config['rot'] == 1:
        angle = random.uniform(0, 2 * math.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0, 0, 1], angle), M)  # z=upright assumption
    if pc_augm_config['mirror_prob'] > 0:  # mirroring x&y, not z
        if random.random() < pc_augm_config['mirror_prob'] / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), M)
        if random.random() < pc_augm_config['mirror_prob'] / 2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 1, 0]), M)
    P[:, :3] = np.dot(P[:, :3], M.T)

    if pc_augm_config['jitter']:
        sigma, clip = 0.01, 0.05  # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
        P = P + np.clip(sigma * np.random.randn(*P.shape), -1 * clip, clip).astype(np.float32)
    return P






################################################  Pre-train Dataset ################################################
class MyPretrainDataset(Dataset):
    def __init__(self, data_path, classes, class2scans, mode='train', num_point=4096, pc_attribs='xyz',
                       pc_augm=False, pc_augm_config=None):
        ''' dataset of base classes
        Args:
            data_path:
            classes: make sure they are in order.
            class2scans:
            mode:
            num_point:
            pc_attribs:
            pc_augm:
            pc_augm_config:
        '''
        super(MyPretrainDataset).__init__()
        self.data_path = data_path # train dataset
        self.classes = classes # train classes name
        self.num_point = num_point # 2048
        self.pc_attribs = pc_attribs # xyzrgbXYZ
        self.pc_augm = pc_augm
        self.pc_augm_config = pc_augm_config

        train_block_names = []
        all_block_names = []
        for k, v in sorted(class2scans.items()):
            all_block_names.extend(v)
            n_blocks = len(v)
            n_test_blocks = int(n_blocks * 0.1)
            n_train_blocks = n_blocks - n_test_blocks
            train_block_names.extend(v[:n_train_blocks])

        if mode == 'train':
            self.block_names = list(set(all_block_names)) # use all training data.
        elif mode == 'test':
            self.block_names = list(set(all_block_names) - set(train_block_names))
        else:
            raise NotImplementedError('Mode is unknown!')

        self.block_names = sorted(self.block_names)
        print('[Pretrain Dataset] Mode: {0} | Num_blocks: {1}'.format(mode, len(self.block_names)))

    def __len__(self):
        return len(self.block_names)

    def __getitem__(self, index):
        block_name = self.block_names[index]
        # print(block_name)
        ptcloud, label, segment_label = sample_pointcloud(self.data_path, self.num_point, self.pc_attribs, self.pc_augm,
                                           self.pc_augm_config, block_name, self.classes, random_sample=True)

        return torch.from_numpy(ptcloud.transpose().astype(np.float32)), torch.from_numpy(label.astype(np.int64)), torch.from_numpy(segment_label.astype(np.float32))


class MyPretrainDataset_CheckBasis(Dataset):
    def __init__(self, data_path, classes, class2scans, mode='train', num_point=4096, pc_attribs='xyz',
                       pc_augm=False, pc_augm_config=None):
        ''' training dataset. use all classes. for fully-supervised training
        Args:
            data_path:
            classes: all the class_name. class_name order.
            class2scans:
            mode:
            num_point:
            pc_attribs:
            pc_augm:
            pc_augm_config:
        '''
        super(MyPretrainDataset_CheckBasis).__init__()
        self.data_path = data_path # train dataset
        self.classes = classes # train classes name
        self.num_point = num_point # 2048
        self.pc_attribs = pc_attribs # xyzrgbXYZ
        self.pc_augm = pc_augm
        self.pc_augm_config = pc_augm_config

        train_block_names = []
        all_block_names = []
        for k, v in sorted(class2scans.items()):
            all_block_names.extend(v)
            n_blocks = len(v)
            n_test_blocks = int(n_blocks * 0.1)
            n_train_blocks = n_blocks - n_test_blocks
            train_block_names.extend(v[:n_train_blocks])

        if mode == 'train':
            self.block_names = list(set(all_block_names)) # use all training data.
        elif mode == 'test':
            self.block_names = list(set(all_block_names) - set(train_block_names))
        else:
            raise NotImplementedError('Mode is unknown!')

        self.block_names.sort()
        print('[Pretrain Dataset] Mode: {0} | Num_blocks: {1}'.format(mode, len(self.block_names)))

    def __len__(self):
        return len(self.block_names)

    def __getitem__(self, index):
        # print(self.block_names[:3])
        block_name = self.block_names[index]
        print(block_name)

        ptcloud, label,_ = sample_pointcloud(self.data_path, self.num_point, self.pc_attribs, self.pc_augm,
                                           self.pc_augm_config, block_name, self.classes, random_sample=True, use_all_classes=True)

        return torch.from_numpy(ptcloud.transpose().astype(np.float32)), torch.from_numpy(label.astype(np.int64))


# ------------------------------------ GFS dataset ----------------------------------

class ValSupp_Dataset(Dataset):
    def __init__(self, data_path, dataset_name, cvfold=0, k_shot=5, mode='train', num_point=2048, pc_attribs='xyz', pc_augm=False, pc_augm_config=None,
                 seed=1, learning_order=None):
        ''' get training data for the new classes. binary mask.
        Args:
            data_path:
            dataset_name:
            cvfold: 0 / 1
            k_shot: 1/ 5
            mode: test. decide the test classes
            num_point:
            pc_attribs:
            pc_augm:
            pc_augm_config:
            seed: 1/2/3 three differen seed to get novel class training data
            learning_order:
        '''
        super(ValSupp_Dataset).__init__()
        self.data_path = data_path # should be train_data

        # self.n_way = n_way # 2
        self.k_shot = k_shot # 1
        self.mode = mode
        self.num_point = num_point # 2048
        self.pc_attribs = pc_attribs
        self.pc_augm = pc_augm
        self.pc_augm_config = pc_augm_config
        self.seed = seed
        self.cvfold = cvfold
        self.learning_order = learning_order

        if dataset_name == 's3dis':
            from dataloaders.s3dis import S3DISDataset
            self.dataset = S3DISDataset(cvfold, data_path) # load train class2scan
        elif dataset_name == 'scannet':
            from dataloaders.scannet import ScanNetDataset
            self.dataset = ScanNetDataset(cvfold, data_path)
        else:
            raise NotImplementedError('Unknown dataset %s!' % dataset_name)

        if mode == 'train':
            self.classes = np.array(self.dataset.train_classes) # meta-train classes, ['door', 'floor', 'sofa', 'table', 'wall', 'window']
        elif mode == 'test':
            self.classes = np.array(self.dataset.test_classes)
        else:
            raise NotImplementedError('Unkown mode %s! [Options: train/test]' % mode)
        self.classes = np.sort(self.classes)

        print('MODE: {0} | Classes: {1}'.format(mode, self.classes))
        self.class2scans = self.dataset.class2scans

        # initialize data_list
        self.save_path = os.path.join(self.data_path, 'ValSupp_S{}_K{}_Seed{}'.format(self.cvfold, self.k_shot, self.seed)) # novel training data store here.
        self.data_list = self.initialize_dataset()
        print('ValSupp path: {}'.format(self.save_path))

    def initialize_dataset(self):
        ''' generate static training samples for new classes
        Returns:
            data_list: record all the pcd path. (pcd_class1_0.pkl, pcd_class1_1.pkl, ...)
            save them in the local: blocks_bs1_s1/ValSupp_S0_K5_Seed1/
        '''
        dst_path = self.save_path
        if os.path.exists(dst_path):
            # training data already exist:
            data_list = os.listdir(os.path.join(dst_path, 'pcd')) # collect all the pcd files
        else:
            # generate pcd, label
            self.generate_one_episode(sampled_classes=self.classes, save_path=dst_path)
            data_list = os.listdir(os.path.join(dst_path, 'pcd'))  # collect all the pcd files

        return data_list


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        '''
        Args:
            index:

        Returns:
            pcd: tensor (9, 2048)
            mask: tensor (2048, )
            cls: its learning order in the whole testing classes.

        '''
        pcd_name = self.data_list[index]
        # load pcd
        with open(os.path.join(self.save_path, 'pcd', pcd_name), 'rb') as f:
            pcd = pickle.load(f)  # (2048, 9)

        # load its mask
        with open(os.path.join(self.save_path, 'mask', pcd_name), 'rb') as f:
            mask = pickle.load(f)  # (2048)

        # get its class
        pcd_cls = int(pcd_name.split('_')[0][5:]) # abosulte class id.
        pcd_cls = self.learning_order.index(pcd_cls) # learning order

        return torch.from_numpy(pcd.transpose().astype(np.float32)), torch.from_numpy(mask.astype(np.float32)), torch.tensor(pcd_cls)


    def generate_one_episode(self, sampled_classes, save_path):
        '''
        Args:
            sampled_classes: all the classes in the novel classes
            save_path: blocks_bs1_s1/ValSupp_S0_K5_Seed1/
        Returns: save pcd, mask in the local blocks_bs1_s1/ValSupp_S0_K5_Seed1/pcd, blocks_bs1_s1/ValSupp_S0_K5_Seed1/mask

        '''
        # torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        # torch.cuda.manual_seed(self.seed)
        # torch.cuda.manual_seed_all(self.seed)
        random.seed(self.seed)
        print('----- creating static ValSupp dataset ----------------')
        black_list = []  # to store the sampled scan names, in order to prevent sampling one scan several times...
        for sampled_class in sampled_classes:
            all_scannames = self.class2scans[sampled_class].copy()
            if len(black_list) != 0:
                all_scannames = [x for x in all_scannames if x not in black_list]
            selected_scannames = np.random.choice(all_scannames, self.k_shot, replace=False)
            black_list.extend(selected_scannames)
            support_scannames = selected_scannames[:]


            support_ptclouds_one_way, support_masks_one_way, support_segment_label_one_way = sample_K_pointclouds(self.data_path, self.num_point,
                                                                                self.pc_attribs, self.pc_augm,
                                                                                self.pc_augm_config,
                                                                                support_scannames,
                                                                                sampled_class,
                                                                                sampled_classes,
                                                                                is_support=True) # will get a binary mask.


            # save in the local
            os.makedirs(os.path.join(save_path, 'pcd'), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'mask'), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'segment_label'), exist_ok=True)
            for k in range(self.k_shot):
                # save pcd:
                with open(os.path.join(save_path, 'pcd', 'class{}_{}.pkl'.format(sampled_class, k)), 'wb') as f:
                    pickle.dump(support_ptclouds_one_way[k], f) # (2048, 9)
                # save mask
                with open(os.path.join(save_path, 'mask', 'class{}_{}.pkl'.format(sampled_class, k)), 'wb') as f:
                    pickle.dump(support_masks_one_way[k], f) # (2048,)
                # save mask
                with open(os.path.join(save_path, 'segment_label', 'class{}_{}.pkl'.format(sampled_class, k)), 'wb') as f:
                    pickle.dump(support_segment_label_one_way[k], f)  # (2048,)

        print('------------- done creating static ValSupp dataset --------------')




class Testing_Dataset(Dataset):
    def __init__(self, data_path, class_names, learning_order, class2scans, mode='train', num_point=4096, pc_attribs='xyz',
                       pc_augm=False, pc_augm_config=None):
        ''' this is only for testing. evaluate on all classes. First Call will generate static_test/
        Args:
            data_path:
            class_names: sorted all class_names (in order) in the testing classes
            learning_order: learning order of all classes.
            class2scans:
            mode:
            num_point:
            pc_attribs:
            pc_augm:
            pc_augm_config:
        '''
        super(Testing_Dataset).__init__()
        self.data_path = data_path # blocks_bs1_s1_test
        self.classes = class_names # all classes (train + test). in order.
        self.learning_order = learning_order # learning order of all classes.
        self.num_point = num_point # 2048
        self.pc_attribs = pc_attribs # xyzrgbXYZ
        self.pc_augm = pc_augm
        self.pc_augm_config = pc_augm_config

        # get block name.
        # train_block_names = []
        all_block_names = []
        for k, v in sorted(class2scans.items()):
            all_block_names.extend(v)
            # n_blocks = len(v)
            # n_test_blocks = int(n_blocks * 0.1)
            # n_train_blocks = n_blocks - n_test_blocks
            # train_block_names.extend(v[:n_train_blocks])

        if mode == 'test':
            self.block_names = list(set(all_block_names))
        else:
            raise NotImplementedError('this dataset is only for testing!')

        print('[Pretrain Dataset] Mode: {0} | Num_blocks: {1}'.format(mode, len(self.block_names)))

        # create staic testing data:
        # initialize data_list
        self.save_path = os.path.join(self.data_path, 'static_test_{}'.format(num_point))  # testing data store here.
        self.data_list = self.initialize_dataset()
        assert len(self.data_list) == len(self.block_names)

    def initialize_dataset(self):
        ''' generate static training samples for new classes
        Returns:
            data_list: record all the pcd path. (pcd_class1_0.pkl, pcd_class1_1.pkl, ...)
            save them in the local: blocks_bs1_s1/ValSupp_S0_K5_Seed1/
        '''

        if os.path.exists(self.save_path):
            # training data already exist:
            data_list = os.listdir(os.path.join(self.save_path, 'pcd')) # collect all the pcd files
        else:
            # generate pcd, label
            self.create_static_testing_data()
            data_list = os.listdir(os.path.join(self.save_path, 'pcd'))  # collect all the pcd files

        return data_list


    def create_static_testing_data(self):
        '''create static testing data: the query label is defined in the class_name order.
        Returns: blocks_bs1_s1/static_test/pcd: 1.pkl, ...
                blocks_bs1_s1/static_test/label:1.pkl, ...

        '''
        print('----- creating static testing dataset -------')
        src_path = os.path.join(self.data_path, 'data')
        block_list = os.listdir(src_path)
        for i in range(len(block_list)):
            # read block
            block_name = block_list[i][:-4]
            # print('{}: {}'.format(i, block_name))
            pcd, label, segment_label = sample_pointcloud(self.data_path, self.num_point, self.pc_attribs, self.pc_augm,
                                               self.pc_augm_config, block_name, self.classes, random_sample=True, use_all_classes=True)
            # save in local dir
            os.makedirs(os.path.join(self.save_path, 'pcd'), exist_ok=True)
            os.makedirs(os.path.join(self.save_path, 'label'), exist_ok=True)
            os.makedirs(os.path.join(self.save_path, 'segment_label'), exist_ok=True)


            with open(os.path.join(self.save_path, 'pcd', '{}.pkl'.format(i)), 'wb') as f:
                pickle.dump(pcd, f)  # (2048, 9)
            # save mask
            with open(os.path.join(self.save_path, 'label', '{}.pkl'.format(i)), 'wb') as f:
                pickle.dump(label, f)  # (2048,)
            # save mask
            with open(os.path.join(self.save_path, 'segment_label', '{}.pkl'.format(i)), 'wb') as f:
                pickle.dump(segment_label, f)  # (2048,)

        print('----------- done creating static testing dataset -----------')
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        '''
        Args:
            index:
        Returns: return a pcd's label based on the learning order!

        '''
        pcd_name = self.data_list[index]
        # print(pcd_name)
        # load pcd
        with open(os.path.join(self.save_path, 'pcd', pcd_name), 'rb') as f:
            pcd = pickle.load(f)  # (2048, 9)

        # load its mask
        with open(os.path.join(self.save_path, 'label', pcd_name), 'rb') as f:
            label = pickle.load(f)  # (2048). its label follow the class_name order!

        # reorganize label based on learning order
        classes = np.unique(label)
        final_label = np.zeros_like(label)
        for cls in classes:
            # get its learning order
            tmp_order = self.learning_order.index(cls)
            # change to learning order
            mask = label == cls
            final_label[mask] = tmp_order

        # load segment_label:
        with open(os.path.join(self.save_path, 'segment_label', pcd_name), 'rb') as f:
            segment_label = pickle.load(f)  # (2048). its label follow the class_name order!

        return torch.from_numpy(pcd.transpose().astype(np.float32)), torch.from_numpy(final_label.astype(np.int64)), torch.from_numpy(segment_label.astype(np.float32))

