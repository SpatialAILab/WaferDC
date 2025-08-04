#for normal non_aug

import numpy as np
import json
import os

num_test =[]
non_aug_multi_i_auroc_f1 = []
non_aug_multi_p_auroc_f1 = []
non_aug_multi_a_auroc_f1 = []
non_aug_multi_fnr_f1 = []
non_aug_multi_fpr_f1 = []
non_aug_multi_fnr_roc = []
non_aug_multi_fpr_roc = []
test_non_aug_multi_fnr_f1 = []
test_non_aug_multi_fpr_f1 = []
test_non_aug_multi_fnr_roc = []
test_non_aug_multi_fpr_roc = []

for i in range(len(os.listdir('./results_final/magnetic_cluster'))):
    with open('./results_final/magnetic_cluster/result_magnetic_'+str(i)+'.json', 'r') as f:
        data = json.load(f)
    num_test.append(data['number_of_test'])
    non_aug_multi_i_auroc_f1.append(data['instance_auroc'])
    non_aug_multi_p_auroc_f1.append(data['full_pixel_auroc'])
    non_aug_multi_a_auroc_f1.append(data['anomaly_pixel_auroc'])

    non_aug_multi_fnr_f1.append(data['test_FNR_from_train_F1_optimal_threshold'])
    non_aug_multi_fpr_f1.append(data['test_FPR_from_train_F1_optimal_threshold'])
    non_aug_multi_fnr_roc.append(data['test_FNR_from_train_roc_optimal_threshold'])
    non_aug_multi_fpr_roc.append(data['test_FPR_from_train_roc_optimal_threshold'])

    test_non_aug_multi_fnr_f1.append(data['test_FNR_from_test_F1_optimal_threshold'])
    test_non_aug_multi_fpr_f1.append(data['test_FPR_from_test_F1_optimal_threshold'])
    test_non_aug_multi_fnr_roc.append(data['test_FNR_from_test_roc_optimal_threshold'])
    test_non_aug_multi_fpr_roc.append(data['test_FPR_from_test_roc_optimal_threshold'])

    #print(num_test)


num_test = np.array(num_test)

non_aug_multi_i_auroc_f1 = np.array(non_aug_multi_i_auroc_f1) #ver5
non_aug_multi_p_auroc_f1 = np.array(non_aug_multi_p_auroc_f1)
non_aug_multi_a_auroc_f1 = np.array(non_aug_multi_a_auroc_f1)

non_aug_multi_fnr_f1 = np.array(non_aug_multi_fnr_f1)
non_aug_multi_fpr_f1 = np.array(non_aug_multi_fpr_f1)
non_aug_multi_fnr_roc = np.array(non_aug_multi_fnr_roc)
non_aug_multi_fpr_roc = np.array(non_aug_multi_fpr_roc)

num_test_ratio = num_test / sum(num_test)
non_aug_multi_i_auroc_f1 = sum(non_aug_multi_i_auroc_f1 * num_test_ratio)
non_aug_multi_p_auroc_f1 = sum(non_aug_multi_p_auroc_f1 * num_test_ratio)
non_aug_multi_a_auroc_f1 = sum(non_aug_multi_a_auroc_f1 * num_test_ratio)

non_aug_multi_fnr_f1 = sum(non_aug_multi_fnr_f1)
non_aug_multi_fpr_f1 = sum(non_aug_multi_fpr_f1)
non_aug_multi_fnr_roc = sum(non_aug_multi_fnr_roc)
non_aug_multi_fpr_roc = sum(non_aug_multi_fpr_roc)

test_non_aug_multi_fnr_f1 = sum(test_non_aug_multi_fnr_f1)
test_non_aug_multi_fpr_f1 = sum(test_non_aug_multi_fpr_f1)
test_non_aug_multi_fnr_roc = sum(test_non_aug_multi_fnr_roc)
test_non_aug_multi_fpr_roc = sum(test_non_aug_multi_fpr_roc)

print("num_test", num_test)
print("non_aug_multi_i_auroc_f1", non_aug_multi_i_auroc_f1)
print("non_aug_multi_p_auroc_f1", non_aug_multi_p_auroc_f1)
print("non_aug_multi_a_auroc_f1", non_aug_multi_a_auroc_f1)

print("non_aug_multi_fnr_f1", non_aug_multi_fnr_f1)
print("non_aug_multi_fpr_f1", non_aug_multi_fpr_f1)
print("non_aug_multi_fnr_roc", non_aug_multi_fnr_roc)
print("non_aug_multi_fpr_roc", non_aug_multi_fpr_roc)

print("test_non_aug_multi_fnr_f1", test_non_aug_multi_fnr_f1)
print("test_non_aug_multi_fpr_f1", test_non_aug_multi_fpr_f1)
print("test_non_aug_multi_fnr_roc", test_non_aug_multi_fnr_roc)
print("test_non_aug_multi_fpr_roc", test_non_aug_multi_fpr_roc)
