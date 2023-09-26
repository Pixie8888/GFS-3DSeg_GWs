# Generalized Few-Shot Point Cloud Segmentation Via Geometric Words
Created by Yating Xu from 
<a href="http://www.nus.edu.sg/" target="_blank">National University of Singapore</a>



## Introduction
This repository contains the PyTorch implementation for our ICCV 2023 Paper 
"[Generalized Few-Shot Point Cloud Segmentation Via Geometric Words](https://arxiv.org/abs/2309.11222)" by Yating Xu, Conghui Hu, Na Zhao, Gim Hee Lee.




## Installation
- python 3.8
- pytorch 1.8 with CUDA 11.1 
- tensorboard, h5py, transforms3d

## Usage
### Data preparation
#### S3DIS
1. Download [S3DIS Dataset Version 1.2](http://buildingparser.stanford.edu/dataset.html).
2. Copy ```pretrain/datasets/S3DIS/meta``` to `ROOT_DIR/datasets/S3DIS/`. Re-organize raw data into `npy` files by running
   ```
   cd pretrain/preprocess
   python collect_s3dis_data.py --data_path $path_to_S3DIS_raw_data --train
   ```
   The generated numpy files are stored in `ROOT_DIR/datasets/S3DIS/scenes/train_data`. 
3. To split rooms into blocks, run 

    ```python room2blocks.py --data_path ROOT_DIR/datasets/S3DIS/scenes/  --block_size 1 --stride 1 --train```
    
    One folder named `blocks_bs1.0_s1.0` will be generated under `ROOT_DIR/datasets/S3DIS/` by default. 

To generate testing data, remove ```--train```.
Data folder for S3DIS is like follow:
```
-- S3DIS
    |-- scenes
        |-- train_data
        |-- test_data
    |-- meta
    |--blocks_bs1.0_s1.0
        |-- data (training block data)
        |--class2scans.pkl (will be generated by train.py)
        |-- ValSupp_S0_K5_Seed10 (support dataset. Will be generated by train.py)
            |-- mask (support point cloud label)
            |-- pcd (support point cloud)
    |--blocks_bs1.0_s1.0_test
        |--data
        |--clas2scans.pkl (will be generated by train.py)
        |--static_test_2048 (Query dataset. Will be generated by train.py)
            |-- label
            |-- pcd

```


#### ScanNet
1. Download [ScanNet V2](http://www.scan-net.org/).
2. Copy ```pretrain/datasets/ScanNet/meta``` folder to `ROOT_DIR/datasets/ScanNet/`. Re-organize raw data into `npy` files by running
	```
	cd pretrain/preprocess
	python collect_scannet_data.py --data_path $path_to_ScanNet_raw_data --train
 	```
   The generated numpy files are stored in `ROOT_DIR/datasets/ScanNet/scenes/train_data` by default.
3. To split rooms into blocks, run 

    ```python room2blocks.py --data_path ROOT_DIR/datasets/ScanNet/scenes/ --dataset scannet --block_size 1 --stride 1 --train```
    
    One folder named `blocks_bs1.0_s1.0` will be generated under `ROOT_DIR/datasets/ScanNet/` by default. 


### Running 
#### Training
1. Pre-training:
```
cd pretrain
bash pretrain_segmentor.sh
```
Change ```DATA_PATH``` accordingly.


2. Compute geometric words from base class data:

Example on S3DIS:
``` 
python get_basis.py --save_path log_s3dis/S0_K5 --pretrain_checkpoint_path /home/yating/Documents/3d_segmentation/GFS_pcd_seg/mpti/log_s3dis/log_pretrain_s3dis_S0_LongTail/ --cvfold 0 
--data_path /home/yating/Documents/3d_segmentation/GFS_pcd_seg/datasets/S3DIS/blocks_bs1.0_s1.0
 --num_cnt 150 --dataset s3dis
```
Change ``` pretrain_checkpoint_path, data_path``` accordingly.

Example on ScanNet:
``` 
python get_basis.py --save_path log_scannet/S0_K5 --pretrain_checkpoint_path /home/yating/Documents/3d_segmentation/GFS_pcd_seg/mpti/log_scannet/log_pretrain_scannet_S0_LongTail/ 
--cvfold 0 --data_path /home/yating/Documents/3d_segmentation/GFS_pcd_seg/datasets/ScanNet/blocks_bs1.0_s1.0 
--num_cnt 180 --dataset scannet
```


3. Training on base class:

Example of 5-shot exp on S3DIS:
```
python train.py  --save_path log_s3dis/S0_K5/exp 
 --pc_augm  --dataset s3dis --k_shot 5 --phase train --cvfold 0  
 --basis_path log_s3dis/S0_K5/GlobalKmeans_EdgeConv123_cnt\=200_energy\=095_SVDReconstruct.pkl  
 --data_path /home/yating/Documents/3d_segmentation/GFS_pcd_seg/datasets/S3DIS/blocks_bs1.0_s1.0 
 --testing_data_path /home/yating/Documents/3d_segmentation/GFS_pcd_seg/datasets/S3DIS/blocks_bs1.0_s1.0_test  
 --use_pretrain_weight --pretrain_checkpoint_path /home/yating/Documents/3d_segmentation/GFS_pcd_seg/mpti/log_s3dis/log_pretrain_s3dis_S0_LongTail/ 
 --epochs 150 --energy 0.9 --total_classes 13 --eval_weight 1. 
```

Example of 5-shot exp on ScanNet:
``` 
python train.py  --save_path log_scannet/S0_K5/exp 
 --pc_augm  --dataset scannet --k_shot 5 --phase train --cvfold 0  
 --basis_path log_scannet/S0_K5/GlobalKmeans_EdgeConv123_cnt\=180_energy\=095_SVDReconstruct.pkl  
 --data_path /home/yating/Documents/3d_segmentation/GFS_pcd_seg/datasets/ScanNet/blocks_bs1.0_s1.0
 --testing_data_path /home/yating/Documents/3d_segmentation/GFS_pcd_seg/datasets/ScanNet/blocks_bs1.0_s1.0_test  
 --use_pretrain_weight --pretrain_checkpoint_path /data/ytxu/GFS_pcd_seg/mpti/log_scannet/log_pretrain_scannet_S0_LongTail_Last6AsNovel/ 
 --epochs 150 --energy 0.95 --total_classes 21 --eval_weight 1.
```
Please change ``` data_path, testing_data_path, pretrain_checkpoint_path``` accordingly. 



#### Evaluation
Evaluation on 5-shot S3DIS: 

Append following command to the training command.
```
 --only_evaluate --phase test --model_checkpoint_path log_s3dis/S0_K5/exp/train_epoch_xxx.pth
 --energy 0.9 --eval_weight 1.2
```

Evaluation on 5-shot ScanNet:

Append following command to the training command.
```
 --only_evaluate --phase test --model_checkpoint_path log_scannet/S0_K5/exp/train_epoch_xxx.pth
 --energy 0.95 --eval_weight 1.2
```
``` energy```  and ``` eval_weight``` correspond to frequency limit in Alg.1 and beta in Eqn.6 of the paper, respectively.



## Acknowledgement
We thank [attMPTI](https://github.com/Na-Z/attMPTI) and [GFS-Seg](https://github.com/dvlab-research/GFS-Seg) for sharing their source code.