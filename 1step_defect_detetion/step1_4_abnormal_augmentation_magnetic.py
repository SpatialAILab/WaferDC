"""
==========================================to be changed==========================================
1. target size 
2. Patchcore.predict (for "is_defect_aug" optimization code)
3. segmentation_masks <-> images
4. cluster3 : real+need_to_aug < real normal dataset
5. thresholding
6. change np.save (143line)
7. why 

"cmd" : bash step1_4_abnormal_augmentation.py
=================================================================================================
"""

import contextlib
import gc
import logging
import os
import sys

import click
import numpy as np
import torch

import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

import dataloader


import matplotlib.pyplot as plt
import matplotlib
import random
import math

import cv2

import copy

from tqdm import tqdm

import natsort
import shutil

import json
LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--save_segmentation_images", is_flag=True)


def main(**kwargs):
    pass


@main.result_callback()
def run(methods, results_path, gpu, seed, save_segmentation_images):
    methods = {key: item for (key, item) in methods}
    #print("asdddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
    os.makedirs(results_path, exist_ok=True)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    dataloader_iter, n_dataloaders = methods["get_dataloaders_iter"]
    dataloader_iter = dataloader_iter(seed)
    patchcore_iter, n_patchcores = methods["get_patchcore_iter"]
    patchcore_iter = patchcore_iter(device)
    if not (n_dataloaders == n_patchcores or n_patchcores == 1):
        raise ValueError(
            "Please ensure that #PatchCores == #Datasets or #PatchCores == 1!"
        )

    for dataloader_count, dataloaders in enumerate(dataloader_iter):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["testing"].name, dataloader_count + 1, n_dataloaders
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["testing"].name

        with device_context:

            torch.cuda.empty_cache()
            if dataloader_count < n_patchcores:
                PatchCore_list = next(patchcore_iter)

            aggregator = {"scores": [], "segmentations": []}
            aggregator = {"train_scores": [], "train_segmentations": [], "scores": [], "segmentations": []}

            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                # #change is_defect_aug=True if abnormal_augmentation.py
                PatchCore.anomaly_segmentor.target_size = 480
                # scores, segmentations, _, images = PatchCore.predict(
                #     dataloaders["testing"], is_defect_aug=True
                # )
                # PatchCore.anomaly_segmentor.target_size = 512
                # scores, segmentations, _, images = PatchCore.predict(
                #     dataloaders["testing"], is_defect_aug=True
                # )
                scores, segmentations, labels_gt, images = PatchCore.predict(
                    dataloaders["testing"], is_defect_aug=True
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)


            labels_gt = np.array(labels_gt)

            print(labels_gt)

            segmentation_save_index = np.max(np.where(labels_gt == 1))

            print(segmentation_save_index)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            # segmentation mask
            np.save("results/magnetic_cluster/magnetic_cluster_num_0/models/magnetic_{i}/segmentations_tests_{i}.npy".format(i=dataloader_count), segmentations[:segmentation_save_index+1]) 
            os.makedirs("results/magnetic_cluster/magnetic_cluster_num_0/test_images", exist_ok=True)
            i=0
            for image in images:
                image = (image *255).astype(np.uint8)
                image = image.transpose((1, 2, 0))
                # print(image.shape)
                cv2.imwrite("results/magnetic_cluster/magnetic_cluster_num_0/test_images/input_image_for_binarymask_{i}.jpg".format(i=i), image)
                i+=1

            #for magnetic
            # np.save("results/magnetic_results/DINO_normal_onestep_10subsample/models/magnetic_{i}/segmentations_tests_{i}.npy".format(i=dataloader_count), segmentations) 
            # os.makedirs("results/magnetic_results/DINO_normal_onestep_10subsample/test_images", exist_ok=True)
            # i=0
            # for image in images:
            #     image = (image *255).astype(np.uint8)
            #     image = image.transpose((1, 2, 0))
            #     print(image.shape)
            #     cv2.imwrite("results/magnetic_results/DINO_normal_onestep_10subsample/test_images/input_image_for_binarymask_{i}.jpg".format(i=i), image)
            #     i+=1


            # # for dagm
            # np.save("results/dagm_results/DINO_normal_aug_before/models/dagm_{i}/segmentations_tests_{i}.npy".format(i=dataloader_count), segmentations) 
            # os.makedirs("results/dagm_results/DINO_normal_aug_before/test_images", exist_ok=True)
            # i=0
            # for image in images:
            #     image = (image *255).astype(np.uint8)
            #     image = image.transpose((1, 2, 0))
            #     print(image.shape)
            #     cv2.imwrite("results/dagm_results/DINO_normal_aug_before/test_images/input_image_for_binarymask_{i}.jpg".format(i=i), image)
            #     i+=1

            # image_auroc= patchcore.metrics.compute_train_threshold(
            #     scores, labels_gt
            # )
            
            # # train threshold
            # train_auroc = image_auroc["train_auroc"]
            # train_roc_optimal_threshold = image_auroc["train_roc_optimal_threshold"]
            # train_F1_optimal_threshold = image_auroc["train_F1_optimal_threshold"]

            # print("train_auroc",train_auroc)
            # print("train_roc_optimal_threshold",train_roc_optimal_threshold)
            # print("train_F1_optimal_threshold",train_F1_optimal_threshold)

            # result_collect.append(
            #     {
            #         "train_auroc": train_auroc,
            #         "train_roc_optimal_threshold":train_roc_optimal_threshold,
            #         "train_F1_optimal_threshold":train_roc_optimal_threshold,
            #     }
            # )

            # #for wafer
            # data_convert = {k:float(v) for k,v in result_collect[-1].items()}

            # result_folder = 'results/wafer_results/DINO_normal_aug_1000'
            # file_path = os.path.join(result_folder, f'train_threshold{i}.json')
            # with open(file_path,'w') as json_file:
            #     json.dump(data_convert, json_file, indent=4)

            # #for dagm
            # data_convert = {k:float(v) for k,v in result_collect[-1].items()}

            # result_folder = 'results/dagm_results/DINO_normal_aug_before'
            # file_path = os.path.join(result_folder, f'train_threshold{i}.json')
            # with open(file_path,'w') as json_file:
            #     json.dump(data_convert, json_file, indent=4)


@main.command("patch_core_loader")
# Pretraining-specific parameters.
@click.option("--patch_core_paths", "-p", type=str, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core_loader(patch_core_paths, faiss_on_gpu, faiss_num_workers):
    #print("dasdddddddddddddddddddddddddddddddddddddddddd")
    def get_patchcore_iter(device):
        for patch_core_path in patch_core_paths:
            loaded_patchcores = []
            gc.collect()
            n_patchcores = len(
                [x for x in os.listdir(patch_core_path) if ".faiss" in x]
            )
            if n_patchcores == 1:
                nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
                patchcore_instance = patchcore.patchcore.PatchCore(device)
                patchcore_instance.load_from_path(
                    load_path=patch_core_path, device=device, nn_method=nn_method
                )
                loaded_patchcores.append(patchcore_instance)
            else:
                for i in range(n_patchcores):
                    nn_method = patchcore.common.FaissNN(
                        faiss_on_gpu, faiss_num_workers
                    )
                    patchcore_instance = patchcore.patchcore.PatchCore(device)
                    patchcore_instance.load_from_path(
                        load_path=patch_core_path,
                        device=device,
                        nn_method=nn_method,
                        prepend="Ensemble-{}-{}_".format(i + 1, n_patchcores),
                    )
                    loaded_patchcores.append(patchcore_instance)

            yield loaded_patchcores

    return ("get_patchcore_iter", [get_patchcore_iter, len(patch_core_paths)])


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=1, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)



def dataset(
    name,
    data_path,
    subdatasets,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
):
    # dataset_info = _DATASETS[name]
    # dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders_iter(seed):
        dataloaders = []

        for subdataset in subdatasets:
            test_dataset = dataloader.MyDataset(dataset_path=data_path,
                                                class_name=subdataset,
                                                resize=resize,
                                                cropsize=imagesize,
                                                is_train=False,
                                                have_gt=False,
                                                is_defect_aug=True)

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader.name = name
            if subdataset is not None:
                test_dataloader.name += "_" + subdataset


            dataloader_dict = {
                "testing": test_dataloader,
            }

            yield dataloader_dict

    return ("get_dataloaders_iter", [get_dataloaders_iter, len(subdatasets)])

"================================================save_anomaly_segmentation_map==============================================="
#please change PatchCore.anomaly_segmentor.target_size = 224 appropirate for resolution!!!
#wafer --> 480, dagm --> 512, magnetic --> 224
  

#save segmentation masks for defect trainset
# logging.basicConfig(level=logging.INFO)
# LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
# main()
"================================================augmentation==============================================="
# # for waferp
# name="wafer"
# loadpath="./results/wafer_cluster"
# modelfolder="wafer_cluster_num_7"
# subdatasets=('0' '1' '2' '3' '4' '5' '6')
# resolution = 480

# #balance
# # # training number of each class
# defect_train_num=[500, 299, 179, 107, 64, 38, 23, 13, 8, 5]
# head_num = 800
# num_normal_aug = 1000

#defect_train_num_aug=[head_num-num for num in defect_train_num]

# #addon
# addon = 300
# defect_train_num_aug=[addon] * 10

# for magnetic
name= "magnetic"
loadpath="./results/magnetic_cluster"
modelfolder="magnetic_cluster_num_4"
subdatasets=('0' '1' '2' '3')
resolution = 224
# for one-class
#subdatasets=('0')

# for 1..? subsample
# modelfolder="DINO_normal_onestep"
# for 10 subsample
#modelfolder="DINO_normal_onestep_10subsample"

# # for multi-class
# subdatasets=('0'  '1')
# modelfolder="DINO_normal_aug_before"

# head_num = 200
# num_normal_aug = 852
# # num_to_aug = 852 // 5
# #num_to_aug = 95

# #balance
defect_train_num=[95, 37, 65, 12, 83]
# defect_train_num_aug=[head_num-num for num in defect_train_num]
# #defect_train_num_aug=[num_to_aug-num for num in defect_train_num]

#print(defect_train_num_aug)

#addon
addon = 300
defect_train_num_aug=[addon] * 10


# # for Dagm
# name= "dagm"
# subdatasets=('0'  '1'  '2' '3' '4' '5' '6' '7' '8' '9')
# loadpath="./results/dagm_results"
# modelfolder="DINO_normal_aug_before"
# resolution = 512

# head_num = 150
# num_normal_aug = 1000
# num_to_aug = 1000

# defect_train_num=[79, 66, 66, 82, 70, 83, 150, 150, 150, 150]
# # defect_train_num_aug=[head_num-num for num in defect_train_num] 
# defect_train_num_aug=[num_to_aug-num for num in defect_train_num]


# augmentation option
# original resize_alpha = 0.5 // resize_beta = 0.5
resize_alpha = 0.5
resize_beta = 0.5
contrast = 1.0  # Contrast control (1.0 means no change)
# bright = random.randint(-10, 10)    # Brightness control (0 means no change)
bright = 0    # Brightness control (0 means no change)


# # wafer_confirm_ver7_step1_1_multi_bank_normal_aug_1000_auroc_test
# data path
# cluster_path = "confirm_ver7_step1_1_multi_bank_normal_aug_1000_auroc_test_3sigma"
# copy_classifier_path = "confirm_ver7_normal_aug_1000_auroc_test_3sigma"


# dagm, magnetic_confirm_ver4_step1_1_one_bank_auroc
cluster_path = "4_cluster"
copy_classifier_path = "test_addon"

dataset_path=f"data/{name}/lets_cluster/{name}_{cluster_path}"



defect_train_count_dict={}

for subdataset in subdatasets:

    each_cluster={}

    x=[]
    phase = 'test'
    img_dir = os.path.join(dataset_path, subdataset, phase)
    img_types = natsort.natsorted(os.listdir(img_dir)) # class 별로 있으므로 이를 정렬

    for img_type in img_types:
        img_type_dir = os.path.join(img_dir, img_type)
        if not os.path.isdir(img_type_dir):
            continue
        img_fpath_list = natsort.natsorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.jpg') or f.endswith('.png')])

        x.extend(img_fpath_list)

    defect_path = list(x)

    for filename in defect_path:
        filename = filename.split('/')[-1]
        filename = filename.split('.')[0]

        cls_idx = filename.split('_')
        cls_idx = int(cls_idx[0][0])

        if filename.startswith(f"{cls_idx}class"):
            each_cluster[cls_idx] = each_cluster.get(cls_idx,0) +1

    defect_train_count_dict['cluster_{}'.format(subdataset)] = each_cluster


print("================before_count_dict================")
for key, value in defect_train_count_dict.items():
    print(key,value)

defect_need_to_aug_count_dict = copy.deepcopy(defect_train_count_dict)
# Iterate over each cluster
for cluster_key, cluster_values in defect_need_to_aug_count_dict.items():
    # Iterate over each key-value pair in the cluster
    for index, value in cluster_values.items():
        # Update the value based on the corresponding index in defect_train_num
        cluster_values[index] = value / defect_train_num[index]

print("================after_counter_dict================")

for key, value in defect_need_to_aug_count_dict.items():
    print(key,value)

for cluster_key, cluster_values in defect_need_to_aug_count_dict.items():
    # Iterate over each key-value pair in the cluster
    for index, value in cluster_values.items():
        # Update the value based on the corresponding index in defect_train_num
        cluster_values[index] = value * defect_train_num_aug[index]

for key, value in defect_need_to_aug_count_dict.items():
    print(key,value)

print("================after ceil================")

for cluster_key, cluster_values in defect_need_to_aug_count_dict.items():
    # Iterate over each key-value pair in the cluster
    for index, value in cluster_values.items():
        # Update the value based on the corresponding index in defect_train_num
        cluster_values[index] = math.ceil(cluster_values[index])

for key, value in defect_need_to_aug_count_dict.items():
    print(key,value)


print("================sum up================")

# Initialize a dictionary to store the total count for each number
total_count_dict = {i: 0 for i in range(len(defect_train_num))}

# Iterate over each cluster
for cluster, counts in defect_need_to_aug_count_dict.items():
    # Iterate over each count in the cluster
    for number, count in counts.items():
        # Update the total count for the number
        total_count_dict[number] += count

for key, value in total_count_dict.items():
    print(key,value)

for subdataset in subdatasets:

    # # aug test (subdataset to 4)
    # subdataset = '1'
    
    print("================aug clusters{}================".format(subdataset))
    x=[]
    phase = 'test'
    img_dir = os.path.join(dataset_path, subdataset, phase)
    img_types = natsort.natsorted(os.listdir(img_dir)) # class 별로 있으므로 이를 정렬

    for img_type in img_types:
        img_type_dir = os.path.join(img_dir, img_type)
        if not os.path.isdir(img_type_dir):
            continue
        img_fpath_list = natsort.natsorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.jpg') or f.endswith('.png')])

        x.extend(img_fpath_list)

    defect_path = list(x)

    cluster_idx = int(subdataset)

    num_to_aug = defect_need_to_aug_count_dict['cluster_{i}'.format(i=subdataset)]

    num_already = defect_train_count_dict['cluster_{i}'.format(i=subdataset)]

    print("num_to_aug of each cluster:", num_to_aug)
    print("all augmentation number of defects:", num_already)

    sum1=0
    sum2=0

    for key, value in num_to_aug.items():
        sum1 +=value

    for key, value in num_already.items():
        sum2 +=value
    print("all augmentation number of defects:", sum1)
    print("all train number of defects:",sum2)
    #print("num_normal_aug:", num_normal_aug)

    # # check if any cluster of augmentation defet + original defect is shorter than normal_aug
    # if num_normal_aug >= sum1+sum2:
    #     num_to_aug_keys = len(num_to_aug)
    #     num_to_add = math.ceil((num_normal_aug - (sum1+sum2))/ num_to_aug_keys)
    #     num_to_aug = {key: value + num_to_add for key, value in num_to_aug.items()}
    #     print(num_to_aug_keys)
    #     print(num_to_add)
    #     print(num_to_aug)
    # else:
    #     pass

    sum=0
    for key, value in num_to_aug.items():
        sum +=value

    print(sum)
    print(sum+sum2)

    num_binary_mask = defect_train_count_dict['cluster_{i}'.format(i=subdataset)]

    # print(defect_path)

    binary_masks = np.load(os.path.join(loadpath,modelfolder,"models","{name}_{i}".format(name=name,i=cluster_idx), "segmentations_tests_{i}.npy".format(i=cluster_idx)))

    os.makedirs('mask_test',exist_ok=True)

    for i in range(binary_masks.shape[0]):
        heatmap_data_scaled = (binary_masks[i] *255).astype(np.uint8)
        heatmap_image = cv2.applyColorMap(heatmap_data_scaled, cv2.COLORMAP_JET)
        cv2.imwrite('./mask_test/heatmap_{}.png'.format(i), heatmap_image)

    #binary mask using segmentation masks
    thresholded_mask=[]

    # Define the threshold value

    #min-max criteria
    for i,filename in tqdm(enumerate(defect_path)):

        image = cv2.imread(filename)
        cv2.imwrite("./mask_test/original_defect_{}.jpg".format(i), image)


        std_deviation = np.std(binary_masks[i])

        min_per_image = np.min(binary_masks[i])
        max_per_image = np.max(binary_masks[i])
        upper_threshold = min_per_image + (max_per_image - min_per_image) * (3/4)
        lower_threshold = min_per_image + (max_per_image - min_per_image) * (1/2) - (3 * std_deviation)
        # lower_threshold = min_per_image + (max_per_image - min_per_image) * (1/8)

    
        # upper_threshold = 0.0000001
        # lower_threshold = 0.0
        #original -- 3/4 and 1/8

        threshold_mask_one = np.where(binary_masks[i] > upper_threshold, 1, 0)
        cv2.imwrite("./mask_test/threshold_mask_one{}.png".format(i), threshold_mask_one*255)


        threshold_mask_two = np.where(((binary_masks[i] > lower_threshold) & (binary_masks[i] < upper_threshold)), 0.1*binary_masks[i], 0)
        cv2.imwrite("./mask_test/threshold_mask_two{}.png".format(i), threshold_mask_two*255)

        threshold_image = threshold_mask_one + threshold_mask_two
        cv2.imwrite("./mask_test/threshold_mask{}.png".format(i), threshold_image*255)

        thresholded_mask.append(threshold_image)


    # mean or median-std criteria
    # for i,filename in tqdm(enumerate(defect_path)):

    #     image = cv2.imread(filename)
    #     cv2.imwrite("./mask_test/original_defect_{}.jpg".format(i), image)

    #     # print(binary_masks[i].shape)
    #     # print(binary_masks[i])
    #     # print(mean)
    #     # print(std_deviation)

    #     # flatten = binary_masks[i].flatten()

    #     # hist, bins = np.histogram(flatten, bins = 'auto')

    #     # print(hist)
    #     # print(bins)

    #     # print(len(hist))
    #     # print(len(bins))

    #     # plt.bar(bins[:-1], hist)
    #     # plt.xticks()

    #     # plt.xlabel('pixel value')
    #     # plt.ylabel('number of pixels')
    #     # plt.title('pixel value distribution')

    #     # plt.savefig('test.png')
    #     std_deviation = np.std(binary_masks[i])

    #     mean = np.mean(binary_masks[i])
    #     #median= np.median(binary_masks[i])

    #     upper_threshold = mean
    #     lower_threshold = mean - 1 * std_deviation

    #     #upper_threshold = median
    #     #lower_threshold = median - 3 * std_deviation

    #     # similiar with CutMIx
    #     # upper_threshold = 0.00001
    #     # lower_threshold = 0

    #     threshold_mask_one = np.where(binary_masks[i] > upper_threshold, 1, 0)
    #     cv2.imwrite("./mask_test/threshold_mask_one{}.png".format(i), threshold_mask_one*255)

    #     threshold_mask_two = np.where(((binary_masks[i] > lower_threshold) & (binary_masks[i] < upper_threshold)), 0.1*binary_masks[i], 0)
    #     cv2.imwrite("./mask_test/threshold_mask_two{}.png".format(i), threshold_mask_two*255)

    #     threshold_image = threshold_mask_one + threshold_mask_two
    #     cv2.imwrite("./mask_test/threshold_mask{}.png".format(i), threshold_image*255)

    #     thresholded_mask.append(threshold_image)

    # # train-threshold criteria
    # for i,filename in tqdm(enumerate(defect_path)):

    #     with open(f'./results/wafer_results/DINO_normal_aug_1000/models/wafer_{cluster_idx}/train_threshold_{cluster_idx}.json', 'r') as f:
    #         threshold_json = json.load(f)

    #     train_threshold = threshold_json['train_F1_optimal_threshold']

    #     print("train_threshold:", train_threshold)

    #     image = cv2.imread(filename)
    #     cv2.imwrite("./mask_test/original_defect_{}.jpg".format(i), image)

    #     std_deviation = np.std(binary_masks[i])

    #     upper_threshold = train_threshold + 2 * std_deviation
    #     lower_threshold = train_threshold - 2 * std_deviation

    #     threshold_mask_one = np.where(binary_masks[i] > upper_threshold, 1, 0)
    #     cv2.imwrite("./mask_test/threshold_mask_one{}.png".format(i), threshold_mask_one*255)


    #     threshold_mask_two = np.where(((binary_masks[i] > lower_threshold) & (binary_masks[i] < upper_threshold)), 0.1*binary_masks[i], 0)
    #     cv2.imwrite("./mask_test/threshold_mask_two{}.png".format(i), threshold_mask_two*255)

    #     threshold_image = threshold_mask_one + threshold_mask_two
    #     cv2.imwrite("./mask_test/threshold_mask{}.png".format(i), threshold_image*255)

    #     thresholded_mask.append(threshold_image)

    np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    binary_masks = thresholded_mask

    print("binary_masks_check")

    # print(binary_masks.dtype)

    # print(binary_masks[-1])

    stop_index_dict = {}

    for key in num_to_aug:
        stop_index_dict[key] = 0

    #load normal train image
    normal_path = './{}/{}/train/good'.format(dataset_path,cluster_idx)
    normal_files = [file for file in os.listdir(normal_path) if file.endswith('.jpg') or file.endswith('.png')]

    #detach defect using segmentation masks
    os.makedirs("{}/{}/test/augmented_defect".format(dataset_path,cluster_idx),exist_ok=True)
    os.makedirs("{}/{}/test/augmented_defect/masked_defect".format(dataset_path,cluster_idx),exist_ok=True)
    os.makedirs("{}/{}/test/augmented_defect/masked_normal".format(dataset_path,cluster_idx),exist_ok=True)
    os.makedirs("{}/{}/test/augmented_defect/result".format(dataset_path,cluster_idx),exist_ok=True)

    while(True):
        mask_index=0
        for filename in defect_path:
            # Load the original image
            image = cv2.imread(filename)

            #print(filename)


            filename = filename.split('/')[-1]
            filename = filename.split('.')[0]

            cls_idx = filename.split('_')
            cls_idx = int(cls_idx[0][0])


            if stop_index_dict[cls_idx] >= num_to_aug[cls_idx]:
                # print("continue")
                mask_index +=1
                continue

            cv2.imwrite("{}/{}/test/augmented_defect/masked_defect/{}class_aug_{}th_from_original_defect_{}.jpg".format(dataset_path,cluster_idx,cls_idx,stop_index_dict[cls_idx],filename), image)

            #print(mask_index)

            # Ensure the dimensions of the image and the binary mask match
            if image.shape[:2] != binary_masks[mask_index].shape:
                image = cv2.resize(image, dsize=(resolution,resolution))
                # raise ValueError(f"Image and binary mask dimensions do not match for {filename}.")

            scaling_factor = np.random.rand() * resize_alpha + resize_beta
            scaling_size = int(resolution * scaling_factor)

            resized_mask = cv2.resize(binary_masks[mask_index], (scaling_size,scaling_size))
            resized_image = cv2.resize(image, (scaling_size,scaling_size))

            # augmentation_option_list = ['rotate_center_diff', 'rotate_center', 'brightness', 'pass']
            augmentation_option_list = ['rotate_center_diff', 'rotate_center', 'pass']

            augmentation_option = random.choice(augmentation_option_list)
            #print(augmentation_option)

            # rotation from top left augmentation
            # if augmentation_option == 'rotate_topleft':            
                # rotation_option_list = [1/8, 2/8, 3/8]
                # rotation_option = random.choice(rotation_option_list)
                # rad = math.pi * rotation_option
                # affine = np.array([[math.cos(rad),math.sin(rad),0],
                #                     [-math.sin(rad),math.cos(rad),0]], dtype=np.float32)
                # resized_mask = cv2.warpAffine(resized_mask, affine, (0,0))
                # resized_image = cv2.warpAffine(resized_image, affine, (0,0))

            if augmentation_option == 'rotate_center_diff':
            # rotation_option_list = [1/3, 2/3, 4/3, 5/3]
                rotation_option_list = [1/5, 2/5, 3/5, 4/5, 6/5, 7/5, 8/5, 9/5]
                rotation_option = random.choice(rotation_option_list)
                scale = 1
                affine = cv2.getRotationMatrix2D((resized_image.shape[0], resized_image.shape[1]), rotation_option, scale)
                resized_mask = cv2.warpAffine(resized_mask, affine, (0,0))
                resized_image = cv2.warpAffine(resized_image, affine, (0,0))

            # rotation from center augmentation
            elif augmentation_option == 'rotate_center':
                rotation_option_list = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
                rotation_option = random.choice(rotation_option_list)
                resized_mask = cv2.rotate(resized_mask, rotation_option)
                resized_image = cv2.rotate(resized_image, rotation_option)

            # Brightness and Contrast Adjustment
            elif augmentation_option == 'brightness':
                resized_mask = resized_mask
                resized_image = cv2.convertScaleAbs(resized_image, alpha=contrast, beta=bright)
            
            # # gaussian fileter
            # resized_mask = resized_mask
            # kernel = np.ones((scaling_size, scaling_size), np.float32) / (scaling_size*scaling_size)
            # resized_image = cv2.filter2D(resized_image, -1, kernel)
            elif augmentation_option == "pass":
                pass

            resized_object = resized_image * resized_mask[:, :, np.newaxis]
            inversed_binary_masks = 1-resized_mask[:, :, np.newaxis]

            cv2.imwrite("{}/{}/test/augmented_defect/masked_defect/{}class_resized_mask_defect_{}th_from_original_defect_{}.jpg".format(dataset_path,cluster_idx,cls_idx,stop_index_dict[cls_idx],filename), resized_object)
            cv2.imwrite("{}/{}/test/augmented_defect/masked_defect/{}class_resized_mask_{}th_from_original_defect_{}.jpg".format(dataset_path,cluster_idx,cls_idx,stop_index_dict[cls_idx],filename), resized_mask*255)

            x, y = np.random.randint(0, resolution-scaling_size+1), np.random.randint(0, resolution-scaling_size+1)

            # Randomly select one normal file
            random_normal_file = random.choice(normal_files)
        
            # Print the randomly selected file
            #print("Randomly selected JPG file:", random_normal_file)
            random_normal_image = cv2.imread(os.path.join(normal_path, random_normal_file))

            random_normal_file = random_normal_file.split('.')[0]

            result = random_normal_image


            if random_normal_image.shape[:2] != binary_masks[mask_index].shape:
                random_normal_image = cv2.resize(random_normal_image, dsize=(resolution,resolution))
                result = random_normal_image
                #print(f"resized Image and binary mask dimensions do not match for {filename}.")

            inversed_normal = random_normal_image[y:y+scaling_size, x:x+scaling_size, :] * inversed_binary_masks

            result[y:y+scaling_size, x:x+scaling_size, :] = inversed_normal + resized_object


            cv2.imwrite("{}/{}/test/augmented_defect/masked_normal/{}class_masked_random_normal_{}th_from_original_defect_{}.jpg".format(dataset_path,cluster_idx,cls_idx,stop_index_dict[cls_idx],filename), inversed_normal)
            cv2.imwrite("{}/{}/test/augmented_defect/result/{}class_aug_result_{}th_from_original_defect_{}.jpg".format(dataset_path,cluster_idx,cls_idx,stop_index_dict[cls_idx],filename), result)


            stop_index_dict[cls_idx] += 1
            mask_index +=1

        # Function to check if all values in num_to_aug are greater than num_binary_mask
        def check_condition():
            return all(num_to_aug[key] <= stop_index_dict.get(key, 0) for key in num_to_aug)

        def check_condition_delete(defect_path, binary_masks):
            for key in num_to_aug:
                # print('======================================check num_to_aug_key_{}==========================='.format(key))
                if num_to_aug[key] <= stop_index_dict.get(key,0):
                    # print('======================================check===========================')
                    filtered_defect_path = [item for item in defect_path if '{}class_sorted'.format(key) not in item]
                    # print(filtered_defect_path)
                    # print(len(filtered_defect_path))
                    # Identifying the indices to keep (those not containing '0class_sorted')
                    keep_indices = [i for i, item in enumerate(defect_path) if '{}class_sorted'.format(key) not in item]
                    # print(keep_indices)
                    # print(len(keep_indices))
                    # print(len(binary_masks))

                    # Filter the numpy array based on these indices
                    binary_masks = [binary_masks[i] for i in keep_indices]
                    # print(len(binary_masks))
                    # print(binary_masks)
                    defect_path = filtered_defect_path

            return defect_path, binary_masks

        print(num_to_aug)
        print(stop_index_dict)

        defect_path, binary_masks = check_condition_delete(defect_path, binary_masks)
        # print(defect_path)2step_PEL_magnetic

        if check_condition():
            print("All conditions are true. Finishing the program.")

            break

        else:
            pass



# create data folder for 2step_PEL_magnetic
folder_list = ['train', 'val', 'test']

os.makedirs(f"../2step_PEL_magnetic/data/{name}/{name}_{copy_classifier_path}", exist_ok=True)

class_num = len(defect_train_num)

for subfolder in folder_list: 
    os.makedirs(f"../2step_PEL_magnetic/data/{name}/{name}_{copy_classifier_path}/{subfolder}", exist_ok=True)
    
    if subfolder == 'train':
        for subdataset in range(class_num):
            os.makedirs(f"../2step_PEL_magnetic/data/{name}/{name}_{copy_classifier_path}/{subfolder}/{subdataset}", exist_ok=True)
    
    else:
        src = f"../2step_PEL_magnetic/data/{name}/{name}_confirm_ver4/{subfolder}"
        dst = f"../2step_PEL_magnetic/data/{name}/{name}_{copy_classifier_path}/{subfolder}"
        if not os.path.exists(dst):
            os.makedirs(dst)

        # 원본 폴더 내의 모든 파일과 폴더를 순회하며 복사
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            shutil.copytree(s, d, dirs_exist_ok=True)  # 하위 폴더 복사


# paste augmentation defect set and original defect set
for subfolder in tqdm(range(len(subdatasets))): 
    original_defect_path = os.path.join(dataset_path, str(subfolder), "test", "clustered_abnormal")
    augmented_defect_path = os.path.join(dataset_path, str(subfolder), "test", "augmented_defect", "result")

    original_defect_path_list = os.listdir(original_defect_path)
    augmented_defect_path_list = os.listdir(augmented_defect_path)
    
    for filename in original_defect_path_list:
        
        class_index = int(filename[0])
        copy_target_folder = os.path.join(f"../2step_PEL_magnetic/data/{name}/{name}_{copy_classifier_path}/train/{class_index}", filename)

        shutil.copy(os.path.join(original_defect_path, filename), copy_target_folder)

    augmented_defect_path_list = os.listdir(augmented_defect_path)

    for filename in augmented_defect_path_list:
        
        class_index = int(filename[0])
        copy_target_folder = os.path.join(f"../2step_PEL_magnetic/data/{name}/{name}_{copy_classifier_path}/train/{class_index}", filename)

        shutil.copy(os.path.join(augmented_defect_path, filename), copy_target_folder)
