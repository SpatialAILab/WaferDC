"""
==========================================to be changed==========================================
1. target size 
2. Patchcore.predict (for "is_defect_aug" optimization code)
3. segmentation_masks <-> images
4. cluster3 : real+need_to_aug < real normal dataset
5. thresholding

"cmd" : bash_sample_
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
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                #change is_defect_aug=True if abnormal_augmentation.py
                PatchCore.anomaly_segmentor.target_size = 480
                scores, segmentations, _, _ = PatchCore.predict(
                    dataloaders["testing"], is_defect_aug=True
                )
                # scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                #     dataloaders["testing"], is_defect_aug=True
                # )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

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

            np.save("results/wafer_Results/DINO_normal_aug/models/wafer_{i}/segmentations_tests_{i}.npy".format(i=dataloader_count), segmentations)

@main.command("patch_core_loader")
# Pretraining-specific parameters.
@click.option("--patch_core_paths", "-p", type=str, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core_loader(patch_core_paths, faiss_on_gpu, faiss_num_workers):
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

# save segmentation masks for defect trainset
# logging.basicConfig(level=logging.INFO)
# LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
# main()

"================================================augmentation==============================================="

dataset_path="data/wafer_confirm_ver4_multi_bank_normal_cluster_normal_aug_each_cluster_ver6"
name="wafer"
loadpath="./results/wafer_Results"
modelfolder="DINO_normal_aug"

# for wafer

subdatasets=('0'  '1'  '2' '3' '4')

# training number of each dfclass
defect_train_num=[500, 299, 179, 107, 64, 38, 23, 13, 8, 5]
defect_train_num_aug=[500-num for num in defect_train_num]

# for Dagm
# subdatasets=('0'  '1'  '2' '3' '4' '5')
# defect_train_num=[79, 66, 66, 82, 70, 83]
# defect_train_num_aug=[2000-num for num in defect_train_num]
print(defect_train_num_aug)

np.random.seed(0)
random.seed(0)

defect_train_count_dict={}

for subdataset in subdatasets:

    each_cluster={}

    x=[]
    phase = 'test'
    img_dir = os.path.join(dataset_path, subdataset, phase)
    img_types = sorted(os.listdir(img_dir)) # class 별로 있으므로 이를 정렬

    for img_type in img_types:
        img_type_dir = os.path.join(img_dir, img_type)
        if not os.path.isdir(img_type_dir):
            continue
        img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.jpg') or f.endswith('.png')])

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

# defect_train_num=[500, 299, 179, 107, 64, 38, 23, 13, 8, 5]
# defect_train_num_aug=[500-num for num in defect_train_num]

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
total_count_dict = {i: 0 for i in range(10)}

# Iterate over each cluster
for cluster, counts in defect_need_to_aug_count_dict.items():
    # Iterate over each count in the cluster
    for number, count in counts.items():
        # Update the total count for the number
        total_count_dict[number] += count

for key, value in total_count_dict.items():
    print(key,value)


for subdataset in subdatasets:
    print("================aug clusters{}================".format(subdataset))
    x=[]
    phase = 'test'
    img_dir = os.path.join(dataset_path, subdataset, phase)
    img_types = sorted(os.listdir(img_dir)) # class 별로 있으므로 이를 정렬

    for img_type in img_types:
        img_type_dir = os.path.join(img_dir, img_type)
        if not os.path.isdir(img_type_dir):
            continue
        img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.jpg') or f.endswith('.png')])

        x.extend(img_fpath_list)

    defect_path = list(x)

    cluster_idx = int(subdataset)

    num_to_aug = defect_need_to_aug_count_dict['cluster_{i}'.format(i=subdataset)]

    num_binary_mask = defect_train_count_dict['cluster_{i}'.format(i=subdataset)]

    print("num_to_aug: ",num_to_aug)
    print("num_binary_mask",num_binary_mask)

    print(defect_path)

    binary_masks = np.load(os.path.join(loadpath,modelfolder,"models","{name}_{i}".format(name=name,i=cluster_idx), "segmentations_tests_{i}.npy".format(i=cluster_idx)))

    print("binary_masks len:{}".format(binary_masks.shape[0]))
    print("binary_masks len:{}".format(binary_masks.shape))

    os.makedirs('mask_test',exist_ok=True)

    for i in range(binary_masks.shape[0]):
        heatmap_data_scaled = (binary_masks[i] *255).astype(np.uint8)
        heatmap_image = cv2.applyColorMap(heatmap_data_scaled, cv2.COLORMAP_JET)
        cv2.imwrite('./mask_test/heatmap_{}.png'.format(i), heatmap_image)

    #binary mask using segmentation masks
    thresholded_mask=[]
    # Define the threshold value
    for i,filename in tqdm(enumerate(defect_path)):

        image = cv2.imread(filename)
        cv2.imwrite("./mask_test/original_defect_{}.jpg".format(i), image)

        min_per_image = np.min(binary_masks[i])
        max_per_image = np.max(binary_masks[i])
        upper_threshold = min_per_image + (max_per_image - min_per_image) * (3/4)
        lower_threshold = min_per_image + (max_per_image - min_per_image) * (7/8)


        # print(min_per_image)
        # print(max_per_image)

        # print(upper_threshold)
        # print(lower_threshold)

        threshold_mask_one = np.where(binary_masks[i] > upper_threshold, 1, 0)

        # print(threshold_mask_one)

        cv2.imwrite("./mask_test/threshold_mask_one{}.png".format(i), threshold_mask_one*255)


        threshold_mask_two = np.where(((binary_masks[i] > lower_threshold) & (binary_masks[i] < upper_threshold)), 0.5*binary_masks[i], 0)

        cv2.imwrite("./mask_test/threshold_mask_two{}.png".format(i), threshold_mask_two*255)

        threshold_image = threshold_mask_one + threshold_mask_two

        cv2.imwrite("./mask_test/threshold_mask{}.png".format(i), threshold_image*255)

        thresholded_mask.append(threshold_image)

    # np.set_printoptions(threshold=np.inf, linewidth=np.inf)

    binary_masks = thresholded_mask

    # print("binary_masks_check")
    # print(binary_masks.dtype)

    # print(binary_masks[0])

    stop_index_dict = {}

    for key in num_to_aug:
        stop_index_dict[key] = 0

    #load normal train image
    normal_path = './{}/{}/train/good'.format(dataset_path,cluster_idx)
    normal_files = [file for file in os.listdir(normal_path) if file.endswith('.jpg')]

    #detach defect using segmentation masks
    os.makedirs("{}/{}/test/augmented_defect".format(dataset_path,cluster_idx),exist_ok=True)
    os.makedirs("{}/{}/test/augmented_defect/masked_defect".format(dataset_path,cluster_idx),exist_ok=True)
    os.makedirs("{}/{}/test/augmented_defect/masked_normal".format(dataset_path,cluster_idx),exist_ok=True)
    os.makedirs("{}/{}/test/augmented_defect/result".format(dataset_path,cluster_idx),exist_ok=True)

    while(True):
        mask_index=0
        for filename in defect_path:
            
            # if filename.endswith('.jpg'):
            # Load the original image
            image = cv2.imread(filename)


            filename = filename.split('/')[-1]
            filename = filename.split('.')[0]

            cls_idx = filename.split('_')
            cls_idx = int(cls_idx[0][0])


            if stop_index_dict[cls_idx] >= num_to_aug[cls_idx]:
                # print("continue")
                continue

            cv2.imwrite("{}/{}/test/augmented_defect/masked_defect/{}class_aug_{}th_from_original_defect_{}.jpg".format(dataset_path,cluster_idx,cls_idx,stop_index_dict[cls_idx],filename), image)

            print(mask_index)

            # Ensure the dimensions of the image and the binary mask match
            if image.shape[:2] != binary_masks[mask_index].shape:
                image = cv2.resize(image, dsize=(480,480))
                # raise ValueError(f"Image and binary mask dimensions do not match for {filename}.")

            scaling_factor = np.random.rand() * 0.1 + 0.9
            scaling_size = int(480 * scaling_factor)

            resized_mask = cv2.resize(binary_masks[mask_index], (scaling_size,scaling_size))
            resized_image = cv2.resize(image, (scaling_size,scaling_size))

            # resize the binary mask
            # kernel_size = 3  # Adjust the kernel size as needed
            # kernel = np.ones((kernel_size, kernel_size), np.uint8)
            # resized_mask = cv2.erode(binary_masks[mask_index],kernel, iterations=1)
            # masked_image = cv2.bitwise_and(image, image, mask=resized_mask)
            # scaling_factor = 0.5
            # resized_object = cv2.resize(masked_image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            # augmentation_option_list = ['rotate_center_diff', 'rotate_center', 'brightness', 'pass']
            augmentation_option_list = ['rotate_center_diff', 'rotate_center', 'pass']

            augmentation_option = random.choice(augmentation_option_list)
            print(augmentation_option)

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
                rotation_option_list = [1/3, 2/3, 4/3, 5/3]
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
                alpha = 1.2  # Contrast control (1.0 means no change)
                beta = 10    # Brightness control (0 means no change)
                resized_image = cv2.convertScaleAbs(resized_image, alpha=alpha, beta=beta)
            # 1/5 just translation
            else:
                pass

            # resized_object = cv2.bitwise_and(resized_image, resized_image, mask=resized_mask)

            resized_object = resized_image * resized_mask[:, :, np.newaxis]

            # inversed_binary_masks = (resized_mask==0).astype(int)
            # inversed_binary_masks = (1-resized_mask).astype(np.uint8)

            inversed_binary_masks = 1-resized_mask[:, :, np.newaxis]

            cv2.imwrite("{}/{}/test/augmented_defect/masked_defect/{}class_resized_mask_defect_{}th_from_original_defect_{}.jpg".format(dataset_path,cluster_idx,cls_idx,stop_index_dict[cls_idx],filename), resized_object)
            cv2.imwrite("{}/{}/test/augmented_defect/masked_defect/{}class_resized_mask_{}th_from_original_defect_{}.jpg".format(dataset_path,cluster_idx,cls_idx,stop_index_dict[cls_idx],filename), resized_mask*255)

            # masked_image = cv2.bitwise_and(image, image, mask=binary_masks[mask_index])

            # output_filename = os.path.join(output_folder, filename)

            # select random bounding_box
            # x, y = np.random.randint(0, 241), np.random.randint(0, 241)

            x, y = np.random.randint(0, 480-scaling_size+1), np.random.randint(0, 480-scaling_size+1)

            # Randomly select one normal file
            random_normal_file = random.choice(normal_files)
        
            # Print the randomly selected file
            print("Randomly selected JPG file:", random_normal_file)
            random_normal_image = cv2.imread(os.path.join(normal_path, random_normal_file))
            result = random_normal_image

            random_normal_file = random_normal_file.split('.')[0]

            if random_normal_image.shape[:2] != binary_masks[mask_index].shape:
                random_normal_image = cv2.resize(random_normal_image, dsize=(480,480))
                print(f"resized Image and binary mask dimensions do not match for {filename}.")

            inversed_normal = random_normal_image[y:y+scaling_size, x:x+scaling_size, :] * inversed_binary_masks

            result[y:y+scaling_size, x:x+scaling_size, :] = inversed_normal + resized_object

            cv2.imwrite("{}/{}/test/augmented_defect/masked_normal/{}class_masked_random_normal_{}th_from_original_defect_{}.jpg".format(dataset_path,cluster_idx,cls_idx,stop_index_dict[cls_idx],filename), inversed_normal)
            cv2.imwrite("{}/{}/test/augmented_defect/result/{}class_aug_result_{}th_from_original_defect_{}.jpg".format(dataset_path,cluster_idx,cls_idx,stop_index_dict[cls_idx],filename), result)

            # bounding_box = random_normal_image[y:y+scaling_size, x:x+scaling_size].astype(np.uint8)
            # masked_random_normal_image = cv2.bitwise_and(bounding_box, bounding_box, mask=inversed_binary_masks)
            # cv2.imwrite("{}/{}/test/augmented_defect/masked_normal/{}class_masked_random_normal_{}th_from_original_defect_{}.jpg".format(dataset_path,cluster_idx,cls_idx,stop_index_dict[cls_idx],random_normal_file), masked_random_normal_image)
            # masked_random_normal_image = cv2.add(masked_random_normal_image, resized_object)

            # # for bounding box area:
            # masked_random_normal_image = random_normal_file * binary_masks[mask_index]

            # cv2.imwrite("{}/{}/test/augmented_defect/masked_normal/{}class_masked_normal_{}th_from_original_defect_{}.jpg".format(dataset_path,cluster_idx,cls_idx,stop_index_dict[cls_idx],random_normal_file), masked_random_normal_image)

            # random_normal_image[y:y+scaling_size, x:x+scaling_size] = masked_random_normal_image
            # result = random_normal_image
            # cv2.imwrite("{}/{}/test/augmented_defect/result/{}class_aug_result_{}th_from_original_defect_{}.jpg".format(dataset_path,cluster_idx,cls_idx,stop_index_dict[cls_idx],filename), result)


            # masked_random_normal_image = random_normal_file[y:y+scaling_size, x:x+scaling_size, :] * inversed_binary_masks
            # cv2.imwrite("{}/{}/test/augmented_defect/masked_normal/{}class_masked_random_normal_{}th_from_original_defect_{}.jpg".format(dataset_path,cluster_idx,cls_idx,stop_index_dict[cls_idx],random_normal_file), masked_random_normal_image)

            # cv2.imshow('gray',result)
            # cv2.waitKey(0)

            stop_index_dict[cls_idx] += 1
            mask_index +=1

        # Function to check if all values in num_to_aug are greater than num_binary_mask
        def check_condition():
            return all(num_to_aug[key] <= stop_index_dict.get(key, 0) for key in num_to_aug)

        print(num_to_aug)
        print(stop_index_dict)

        if check_condition():
            print("All conditions are true. Finishing the program.")

            break

        else:
            pass
