"""
==========================================to be changed==========================================
1. split augmented normal to k-means clustering results (automatically)
2. automatically balancing normal aug to multi-class bank for thresholding based on normal set

cmd : python3 step1_2_normal_augmentation.py
=================================================================================================
"""

import os
from torchvision import transforms
from PIL import Image
import random
from tqdm import tqdm

import numpy as np
import copy
import math

def process_images(input_folder, output_folder, check_folder, n_cluster, num_images=2000):
    # Ensure the output folder exists or create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    # Define the transformation

    
    for root, dirs, files in os.walk(input_folder):
        j= 0
        while j < num_images:
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):  # Check for image files
                    img_path = os.path.join(root, file)
                    image = Image.open(img_path)
                    w,h=image.size
                    height_cropsize = random.randrange(int(h*2/4), int(h*4/5)) # 각각의 이미지 사이즈 별로 다른 높이와 너비로 이미지를 자름
                    width_cropsize = random.randrange(int(w*2/4), int(w*4/5))
                    randint = random.randrange(300,400)
                    transform = transforms.Compose([
            #transforms.CenterCrop(randint),
        transforms.RandomCrop([height_cropsize, width_cropsize]),
        #transforms.Resize(480)
    ])
                    cropped_images = transform(image)
                    
                    # Save a randomly chosen crop
                    cropped_images.save(f"{output_folder}/cluster_{n_cluster}_normal_aug_random_crop_sorted_{j}.jpg")
                    cropped_images.save(f"{check_folder}/cluster_{n_cluster}_normal_aug_random_crop_sorted_{j}.jpg")
                    j=j+1
                    
                    if j==num_images:
                        break
                    # print(f"Processed {file}")
    print("Image processing completed.")


# dataset_path="data/wafer_confirm_ver4_multi_bank_normal_cluster_normal_aug_each_cluster_ver6"
# name="wafer"
# loadpath="./results/wafer_Results"
# modelfolder="DINO_normal_aug"

# # for wafer

# subdatasets=('0'  '1'  '2' '3' '4')

# # training number of each dfclass
# defect_train_num=[500, 299, 179, 107, 64, 38, 23, 13, 8, 5]
# defect_train_num_aug=[500-num for num in defect_train_num]

# # for Dagm
# # subdatasets=('0'  '1'  '2' '3' '4' '5')
# # defect_train_num=[79, 66, 66, 82, 70, 83]
# # defect_train_num_aug=[2000-num for num in defect_train_num]
# print(defect_train_num_aug)

# np.random.seed(0)
# random.seed(0)

# defect_train_count_dict={}

# for subdataset in subdatasets:

#     each_cluster={}

#     x=[]
#     phase = 'test'
#     img_dir = os.path.join(dataset_path, subdataset, phase)
#     img_types = sorted(os.listdir(img_dir)) # class 별로 있으므로 이를 정렬

#     for img_type in img_types:
#         img_type_dir = os.path.join(img_dir, img_type)
#         if not os.path.isdir(img_type_dir):
#             continue
#         img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.jpg') or f.endswith('.png')])

#         x.extend(img_fpath_list)

#     defect_path = list(x)

#     for filename in defect_path:
#         filename = filename.split('/')[-1]
#         filename = filename.split('.')[0]

#         cls_idx = filename.split('_')
#         cls_idx = int(cls_idx[0][0])

#         if filename.startswith(f"{cls_idx}class"):
#             each_cluster[cls_idx] = each_cluster.get(cls_idx,0) +1

#     defect_train_count_dict['cluster_{}'.format(subdataset)] = each_cluster

# print("================before_count_dict================")
# for key, value in defect_train_count_dict.items():
#     print(key,value)

# num_defect_each_cluster = []

# for cluster_key, cluster_values in defect_train_count_dict.items():
#     # Iterate over each key-value pair in the cluster
#     each_cluster_sum = 0
#     for index, value in cluster_values.items():
#         # Update the value based on the corresponding index in defect_train_num
#         each_cluster_sum += value
#     num_defect_each_cluster.append(each_cluster_sum)

# print(num_defect_each_cluster)

# # defect_train_num=[500, 299, 179, 107, 64, 38, 23, 13, 8, 5]
# # defect_train_num_aug=[500-num for num in defect_train_num]

# defect_need_to_aug_count_dict = copy.deepcopy(defect_train_count_dict)
# # Iterate over each cluster
# for cluster_key, cluster_values in defect_need_to_aug_count_dict.items():
#     # Iterate over each key-value pair in the cluster
#     for index, value in cluster_values.items():
#         # Update the value based on the corresponding index in defect_train_num
#         cluster_values[index] = value / defect_train_num[index]

# print("================after_counter_dict================")

# for key, value in defect_need_to_aug_count_dict.items():
#     print(key,value)

# for cluster_key, cluster_values in defect_need_to_aug_count_dict.items():
#     # Iterate over each key-value pair in the cluster
#     for index, value in cluster_values.items():
#         # Update the value based on the corresponding index in defect_train_num
#         cluster_values[index] = value * defect_train_num_aug[index]

# for key, value in defect_need_to_aug_count_dict.items():
#     print(key,value)

# print("================after ceil================")

# for cluster_key, cluster_values in defect_need_to_aug_count_dict.items():
#     # Iterate over each key-value pair in the cluster
#     for index, value in cluster_values.items():
#         # Update the value based on the corresponding index in defect_train_num
#         cluster_values[index] = math.ceil(cluster_values[index])

# for key, value in defect_need_to_aug_count_dict.items():
#     print(key,value)


# print("================check need to all sum up================")

# # Initialize a dictionary to store the total count for each number
# total_count_dict = {i: 0 for i in range(10)}

# # Iterate over each cluster
# for cluster, counts in defect_need_to_aug_count_dict.items():
#     # Iterate over each count in the cluster
#     for number, count in counts.items():
#         # Update the total count for the number
#         total_count_dict[number] += count

# for key, value in total_count_dict.items():
#     print(key,value)

# print("================check each cluster needed to be augmentation================")

# # Initialize a dictionary to store the total count for each number
# need_augmentation = []

# for cluster_key, cluster_values in defect_need_to_aug_count_dict.items():
#     # Iterate over each key-value pair in the cluster
#     each_cluster_sum = 0
#     for index, value in cluster_values.items():
#         # Update the value based on the corresponding index in defect_train_num
#         each_cluster_sum += value
#     need_augmentation.append(each_cluster_sum)

# print(need_augmentation)


# print("================check each cluster after_defect_augmentation================")
# after_defect_augmentation = [x + y for x, y in zip(num_defect_each_cluster, need_augmentation)]

# print(after_defect_augmentation)
# import sys
# sys.exit(0)

# #wafer
# dataset = "wafer"
# n_clusters = 5
# subdatasets=('0'  '1'  '2' '3' '4')
# num_images = 1000

# # dagm
dataset = "dagm"
n_clusters = 10
subdatasets=('0'  '1'  '2' '3' '4' '5' '6' '7' '8' '9')
num_images = 1000

# magnetic
dataset = "magnetic"
n_clusters = 4
subdatasets=('0' '1' '2' '3')
num_images = 500

dataset_path=f"data/{dataset}/lets_cluster/{dataset}_{n_clusters}_cluster"

# np.random.seed(0)
# random.seed(0)
normal_train_num_aug=[]

for subdataset in subdatasets:
    phase = 'train'
    img_dir = os.path.join(dataset_path, subdataset, phase)
    img_types = sorted(os.listdir(img_dir)) # class 별로 있으므로 이를 정렬

    for img_type in img_types:
        img_type_dir = os.path.join(img_dir, img_type)
        if not os.path.isdir(img_type_dir):
            continue
        img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.jpg') or f.endswith('.png')])

        normal_train_num_aug.append(len(img_fpath_list))

print(normal_train_num_aug)

normal_train_num_aug=[num_images-num for num in normal_train_num_aug]
#normal_train_num_aug=[509, 0, 505, 0, 0, 0, 493, 492, 509, 496]
print(normal_train_num_aug)

# Set input and output folders

for i, num_aug in tqdm(enumerate(normal_train_num_aug)):
    input_folder_path = f"data/{dataset}/lets_cluster/{dataset}_{n_clusters}_cluster/{i}/train/good".format(i=i)

    os.makedirs(f"data/{dataset}/lets_cluster/{dataset}_{n_clusters}_cluster_aug", exist_ok=True)
    os.makedirs(f"data/{dataset}/lets_cluster/{dataset}_{n_clusters}_cluster_aug/{i}".format(i=i), exist_ok=True)

    output_folder_path = f"data/{dataset}/lets_cluster/{dataset}_{n_clusters}_cluster_1_5/{i}/test/good".format(i=i)
    check_folder_path = f"data/{dataset}/lets_cluster/{dataset}_{n_clusters}_cluster_aug/{i}".format(i=i)

# Process images
    process_images(input_folder_path, output_folder_path, check_folder_path, n_cluster=i, num_images=num_aug)