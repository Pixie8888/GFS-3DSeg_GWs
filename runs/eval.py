"""Evaluating functions for Few-shot 3D Point Cloud Semantic Segmentation

Author: Zhao Na, 2020
"""
import os
import numpy as np


def evaluate_metric_GFS(logger, pred_labels_list, gt_labels_list, test_classes, novel_classes, all_learning_order, scannet=False):
    """ test on GFS
    :param pred_labels_list: a list of np array, each entry with shape (n_queries, num_points).
    :param gt_labels_list: a list of np array, each entry with shape (n_queries, num_points).
    :param test_classes: test on all classes (base + novel).
    :param novel_classes: novel classes list
    :param all_learning_order:
    :return:
    class-wise iou
    mean-iou: average ovel all classes
    base_iou: iou of base classes
    novel_iou: iou of novel classes
    """
    assert len(pred_labels_list) == len(gt_labels_list)

    logger.cprint('*****Test Classes: {0}*****'.format(test_classes))

    NUM_CLASS = len(test_classes) #
    gt_classes = [0 for _ in range(NUM_CLASS)] # class name as id
    positive_classes = [0 for _ in range(NUM_CLASS)]
    true_positive_classes = [0 for _ in range(NUM_CLASS)]

    for i, batch_gt_labels in enumerate(gt_labels_list):
        batch_pred_labels = pred_labels_list[i]  # (n_queries*n_way, num_points)
        # label2class = label2class_list[i]  # (n_way,)

        for j in range(batch_pred_labels.shape[0]):  # for each way (each 2048 query)
            for k in range(batch_pred_labels.shape[1]):  # for each point
                gt = int(batch_gt_labels[j, k]) # gt label id.
                pred = int(batch_pred_labels[j, k])

                # get gt_class: from learning order to class_name:
                gt_index = all_learning_order[gt]
                gt_classes[gt_index] += 1

                # get pred_class
                pred_index = all_learning_order[pred]
                positive_classes[pred_index] += 1

                true_positive_classes[gt_index] += int(gt == pred)

    iou_list = []
    base_iou_list = []
    novel_iou_list = []

    if scannet == False:
        # get iou based on class_names!
        for c in range(NUM_CLASS):
            iou = true_positive_classes[c] / float(gt_classes[c] + positive_classes[c] - true_positive_classes[c])
            logger.cprint('----- [class %d]  IoU: %f -----' % (c, iou))
            iou_list.append(iou)

            if c in novel_classes:
                novel_iou_list.append(iou)
            else:
                base_iou_list.append(iou)

        # get mean-iou
        mean_iou = np.array(iou_list).mean()
        logger.cprint('mean-iou: {}'.format(mean_iou))
        # get base iou
        base_iou = np.array(base_iou_list).mean()
        logger.cprint('base-iou: {}'.format(base_iou))
        # get novel iou
        novel_iou = np.array(novel_iou_list).mean()
        logger.cprint('novel-iou: {}'.format(novel_iou))
        # get hm
        hm = 2*base_iou*novel_iou / (base_iou+novel_iou)
        logger.cprint('hm-iou: {}'.format(hm))

    else:
        # get iou based on class_names! for scannet. skip class_name=0
        for c in range(NUM_CLASS):
            iou = true_positive_classes[c] / float(gt_classes[c] + positive_classes[c] - true_positive_classes[c])
            logger.cprint('----- [class %d]  IoU: %f -----' % (c, iou))
            iou_list.append(iou)

            if c == 0:
                continue

            if c in novel_classes:
                novel_iou_list.append(iou)
            else:
                base_iou_list.append(iou)

        # get mean-iou
        mean_iou = np.array(iou_list[1:]).mean()
        iou_list = iou_list[1:] # skip class 0
        logger.cprint('mean-iou: {}'.format(mean_iou))
        # get base iou
        base_iou = np.array(base_iou_list).mean()
        logger.cprint('base-iou: {}'.format(base_iou))
        # get novel iou
        novel_iou = np.array(novel_iou_list).mean()
        logger.cprint('novel-iou: {}'.format(novel_iou))
        # get hm
        hm = 2 * base_iou * novel_iou / (base_iou + novel_iou)
        logger.cprint('hm-iou: {}'.format(hm))

    return mean_iou, base_iou, novel_iou, hm, np.array(iou_list)


