"""Anomaly metrics."""
import numpy as np
from sklearn import metrics
import pdb
import shutil
from torchvision import transforms
from PIL import Image
import torchvision.utils as utils

import os

# train threshold 불러와서 FNR, FPR 계산 추가

def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels, image_list, image_file_path, train_roc_optimal_threshold, train_F1_optimal_threshold, \
    results_path, log_project, log_group, dataloaders_name, run_save_path
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """

    # print("=================check1=================")

    # print(type(anomaly_ground_truth_labels))
    # print(type(anomaly_prediction_weights))

    # print(anomaly_ground_truth_labels)
    # print(anomaly_prediction_weights)

    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    optimal_idx = np.argmax(tpr - fpr)
    test_roc_optimal_threshold = thresholds[optimal_idx]
    print("test_roc_Threshold value is:", test_roc_optimal_threshold)
    
    precision, recall, thresholds = metrics.precision_recall_curve(
    anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    
    test_F1_optimal_threshold = thresholds[np.argmax(F1_scores)]
    print("test_F1_Threshold value is:", test_F1_optimal_threshold)
    # print("test_F1_Threshold value is:", test_F1_optimal_threshold)
    # def inverse_normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    #         """
    #         :param img: numpy array. shape (height, width, channel). [-1~1]
    #         :return: numpy array. shape (height, width, channel). [0~1]
    #         """
    #         img[0,:,:] = ((img[0,:,:]) * std[0]) + mean[0]
    #         img[1,:,:] = ((img[1,:,:]) * std[1]) + mean[1]
    #         img[2,:,:] = ((img[2,:,:]) * std[2]) + mean[2]
    #         return img*255
    
    only_anomaly_prediction_weights = anomaly_prediction_weights[anomaly_ground_truth_labels == 1]

    # print("=================check2=================")

    # print(only_anomaly_prediction_weights)


    anomaly_image_list = np.array(image_list)[anomaly_ground_truth_labels == 1]
    anomaly_image_file_path = np.array(image_file_path)[anomaly_ground_truth_labels == 1]

    # print("=================check3=================")

    # print(anomaly_image_list)

    test_FNR_from_train_F1_optimal_threshold = np.sum(only_anomaly_prediction_weights < train_F1_optimal_threshold)
    test_FNR_from_train_roc_optimal_threshold = np.sum(only_anomaly_prediction_weights < train_roc_optimal_threshold)
    test_FNR_from_test_F1_optimal_threshold = np.sum(only_anomaly_prediction_weights < test_F1_optimal_threshold)
    test_FNR_from_test_roc_optimal_threshold = np.sum(only_anomaly_prediction_weights < test_roc_optimal_threshold)


    below_threshold_indices = [index for index, score in enumerate(only_anomaly_prediction_weights) if score < train_F1_optimal_threshold]
   
    # print("=================check4=================")

    # print(below_threshold_indices)

    for i in range(0, len(below_threshold_indices)):
        image = anomaly_image_list[below_threshold_indices[i]]
        image_file = anomaly_image_file_path[below_threshold_indices[i]]
        image = np.array(image)

        #image = inverse_normalize(image)
        pil_image = Image.fromarray(image.astype(np.uint8).transpose(1,2,0))
        
        os.makedirs(os.path.join(run_save_path, "FNR"), exist_ok=True)
        os.makedirs(os.path.join(run_save_path, "FNR", dataloaders_name), exist_ok=True)
        
        save_path = os.path.join(run_save_path, "FNR" , dataloaders_name)
        shutil.copy(image_file, save_path)
        #pil_image.save(save_path)
    
    
    only_normal_prediction_weights = anomaly_prediction_weights[anomaly_ground_truth_labels == 0]
    print(np.mean(only_normal_prediction_weights))
    normal_image_list = np.array(image_list)[anomaly_ground_truth_labels == 0]
    normal_image_file_path = np.array(image_file_path)[anomaly_ground_truth_labels == 0]

    test_FPR_from_train_F1_optimal_threshold = np.sum(only_normal_prediction_weights > train_F1_optimal_threshold)
    test_FPR_from_train_roc_optimal_threshold = np.sum(only_normal_prediction_weights > train_roc_optimal_threshold)
    test_FPR_from_test_F1_optimal_threshold = np.sum(only_normal_prediction_weights > test_F1_optimal_threshold)
    test_FPR_from_test_roc_optimal_threshold = np.sum(only_normal_prediction_weights > test_roc_optimal_threshold)

    upper_threshold_indices = [index for index, score in enumerate(only_normal_prediction_weights) if score > train_F1_optimal_threshold]

    test_FNR_from_train_F1_optimal_threshold = int(test_FNR_from_train_F1_optimal_threshold)
    test_FPR_from_train_F1_optimal_threshold = int(test_FPR_from_train_F1_optimal_threshold)

    for i in range(0, len(upper_threshold_indices)):
        image = normal_image_list[upper_threshold_indices[i]]
        image_file_path = normal_image_file_path[upper_threshold_indices[i]]

        image = np.array(image)
        #image = inverse_normalize(image)
        
        pil_image = Image.fromarray(image.astype(np.uint8).transpose(1,2,0))
        
        os.makedirs(os.path.join(run_save_path, "FPR"), exist_ok=True)
        os.makedirs(os.path.join(run_save_path, "FPR", dataloaders_name), exist_ok=True)

        save_path = os.path.join(run_save_path, "FPR" , dataloaders_name)
        shutil.copy(image_file_path, save_path)
    #print(anomaly_prediction_weights.shape)
    #pdb.set_trace()

    return{"auroc":auroc, "threshold":thresholds,\
    "test_FNR_from_train_F1_optimal_threshold":test_FNR_from_train_F1_optimal_threshold,\
    "test_FNR_from_test_F1_optimal_threshold":test_FNR_from_test_F1_optimal_threshold,\
    "test_FPR_from_train_F1_optimal_threshold":test_FPR_from_train_F1_optimal_threshold,\
    "test_FPR_from_test_F1_optimal_threshold":test_FPR_from_test_F1_optimal_threshold,\

    "test_FNR_from_train_roc_optimal_threshold":test_FNR_from_train_roc_optimal_threshold,\
    "test_FNR_from_test_roc_optimal_threshold":test_FNR_from_test_roc_optimal_threshold,\
    "test_FPR_from_train_roc_optimal_threshold":test_FPR_from_train_roc_optimal_threshold,\
    "test_FPR_from_test_roc_optimal_threshold":test_FPR_from_test_roc_optimal_threshold,\

    }




def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }


# train threshold output ㅎ함수


def compute_train_threshold(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    optimal_idx = np.argmax(tpr - fpr)
    roc_optimal_threshold = thresholds[optimal_idx]
    print("roc_Threshold value is:", roc_optimal_threshold)
    
    precision, recall, thresholds = metrics.precision_recall_curve(
    anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    only_normal_prediction_weights = anomaly_prediction_weights[anomaly_ground_truth_labels == 0]
    only_abnormal_prediction_weights = anomaly_prediction_weights[anomaly_ground_truth_labels == 1]
    # print("-----normal-----")
    # #mean = only_normal_prediction_weights.mean
    # print(np.mean(only_normal_prediction_weights))
    # print("-----abnormal-----")
    # print(only_abnormal_prediction_weights)
    # print("-----abnormal_mean-----")
    # print(np.mean(only_abnormal_prediction_weights))
    # print("-----abnormal_min_max-----")
    # print(np.min(only_abnormal_prediction_weights))
    # print(np.max(only_abnormal_prediction_weights))
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    
    F1_optimal_threshold = thresholds[np.argmax(F1_scores)]
    print("F1_Threshold value is:", F1_optimal_threshold)
    
    
    return {"train_auroc": auroc, "train_roc_optimal_threshold": roc_optimal_threshold, "train_F1_optimal_threshold": F1_optimal_threshold}
