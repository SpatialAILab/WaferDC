"""
==========================================to be changed==========================================
1. extract_features_one functions to extract_features
2. normal_label ==10
3. str(i) // os.path.join optimization
4. in patchcore code, need to be implemeneted for having no gt masks for trainsets
5. in setting normal bank based on clustering -- saved npy file for each normal bank
(two feature extractor step for )

cmd : python3 step1_1_train_k_means.py
=================================================================================================
"""


# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from PIL import Image

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
import cv2
import seaborn as sns
import shutil
import torch
from PIL import Image
import torchvision.transforms as transforms
import shutil
from tqdm import tqdm

def extract_features(input_folder, image_files, preprocess):
    def _detach(features):
        return [x.detach().cpu().numpy() for x in features]
    featuress = []
    for image_file in tqdm(image_files):
        image_path = os.path.join(input_folder, image_file)
    # load the image as a 224x224 array
        img = Image.open(image_path).convert('RGB')
        tensor_image = preprocess(img)
        img =tensor_image.unsqueeze(0).to(device)
        #print(img.shape)
        with torch.no_grad():
            feature = feature_extractor.get_intermediate_layers(img)[0]
        x_norm = feature[:,0,:]
        x_norm = x_norm.squeeze()
        # x_norm = x_norm.unsqueeze(1)
        # x_norm = x_norm.detach().cpu().numpy()

        x_prenorm = feature[:,1:,:]
        # x_prenorm = x_prenorm.detach().cpu().numpy()
        x_prenorm = x_prenorm.squeeze()
        
        x_norm = torch.repeat_interleave(x_norm.unsqueeze(0), x_prenorm.shape[0], dim=0)
        # x_norm = np.repeat(np.expand_dims(x_norm, axis=1), x_prenorm.shape[1], axis=1)
        features = torch.concat([x_norm, x_prenorm], axis=-1)
        features = features.reshape(28, 28, features.shape[-1])
        features = features.unsqueeze(0)
        features = features.reshape(-1, features.shape[-1])
        featuress.append(features)

    return _detach(featuress)

def extract_features_one(input_folder, image_file, preprocess):
    def _detach(features):
        return [x.detach().cpu().numpy() for x in features]
    
    image_path = os.path.join(input_folder, image_file)
# load the image as a 224x224 array
    img = Image.open(image_path).convert('RGB')
    tensor_image = preprocess(img)
    img =tensor_image.unsqueeze(0).to(device)
    #print(img.shape)
    with torch.no_grad():
        feature = feature_extractor.get_intermediate_layers(img)[0]
    x_norm = feature[:,0,:]
    x_norm = x_norm.squeeze()
    # x_norm = x_norm.unsqueeze(1)
    # x_norm = x_norm.detach().cpu().numpy()

    x_prenorm = feature[:,1:,:]
    # x_prenorm = x_prenorm.detach().cpu().numpy()
    x_prenorm = x_prenorm.squeeze()
    
    x_norm = torch.repeat_interleave(x_norm.unsqueeze(0), x_prenorm.shape[0], dim=0)
    # x_norm = np.repeat(np.expand_dims(x_norm, axis=1), x_prenorm.shape[1], axis=1)
    features = torch.concat([x_norm, x_prenorm], axis=-1)
    features = features.reshape(28, 28, features.shape[-1])
    features = features.unsqueeze(0)
    features = features.reshape(-1, features.shape[-1])
    # convert from 'PIL.Image.Image' to numpy array
    
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    #reshaped_img = img.reshape(1,224,224,3)
    # prepare image for model
    #imgx = preprocess_input(reshaped_img)
    # get the feature vector
    #features = model.predict(imgx, use_multiprocessing=True)
    return _detach(features)

#==============================================================================================================================
# split defect trainset based on l2 distance
cluster_num = 4
# wafer
#dataset = "wafer"

# dagm
#dataset = "dagm"

# magnetic
dataset = "magnetic"


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

# extract features for each normal bank
#non-augwafer_confirm_ver4_step1_1_multi_bank_normal_aug_false_1000_real
path_normal_cluster = f"data/{dataset}/lets_cluster/{dataset}_{cluster_num}_cluster"
#aug
#path_normal_cluster = f"data/{dataset}/{dataset}_confirm_ver4_step1_1_multi_bank_normal_aug_false"

preprocess = transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=mean, std=std),
                        ])
device = torch.device('cuda:0')
feature_extractor = torch.hub.load("facebookresearch/dino:main", "dino_vitb8").to(device)


folder_names = sorted(os.listdir(path_normal_cluster))
mean_feature = []
output_folders = []

for i in range(len(folder_names)):
    image_folder = os.path.join(path_normal_cluster, folder_names[i], 'train', 'good')
    print(image_folder)
    image_files = os.listdir(image_folder)
    features = extract_features(image_folder, image_files, preprocess)
    #tensor_features = torch.tensor(np.array(features))
    #gap = torch.mean(tensor_features, dim=(0, 1))
    mean = np.mean(features, axis=0)
    #print(gap)
    #print(median)
    print(np.array(mean).shape)
    #print(np.array(median).shape)
    mean_feature.append(mean)

    output_folders.append(str(i))

print(np.array(mean_feature).shape)


# Split abnormal trainset for each cluster(normal_bank)

#aug
path_test_cluster = f"../test_data/{dataset}_test/{dataset}_{cluster_num}_cluster_test"
#non-aug
#path_test_cluster = f"../test_data/{dataset}_test/{dataset}_confirm_ver4_test_step1_1_multi_bank_normal_aug_false"

os.makedirs(path_test_cluster, exist_ok=True)


for folder in output_folders:
    result_folder = os.path.join(path_test_cluster,folder,'test')
    os.makedirs(result_folder, exist_ok=True)

    result_folder = os.path.join(path_test_cluster,folder,'test','clustered_abnormal')
    os.makedirs(result_folder, exist_ok=True)

    result_folder = os.path.join(path_test_cluster,folder,'test','good')
    os.makedirs(result_folder, exist_ok=True)

    result_folder = os.path.join(path_test_cluster,folder,'ground_truth','clustered_abnormal')
    os.makedirs(result_folder, exist_ok=True)

folder_abnormal_test = f"../test_data/{dataset}_confirm_ver4/test"
folder_abnormal_test_mask = f"../test_data/{dataset}_confirm_ver4/ground_truth"

image_abnormal_test = os.listdir(folder_abnormal_test)
# mask_abnormal_train = os.listdir(folder_abnormal_test_mask)

print(image_abnormal_test)

for i in tqdm(image_abnormal_test):

    folder_abnormal_test_split = os.path.join(folder_abnormal_test, i)
    image_files = os.listdir(folder_abnormal_test_split)


    if i != '10':
        folder_abnormal_test_mask_split = os.path.join(folder_abnormal_test_mask, i)
        mask_files = os.listdir(folder_abnormal_test_mask_split)

        for image_file in image_files:
            #print(image_file)
            common_text = image_file[:-4]  # Remove the '.png' extension
            common_text += '_mask'

            for item in mask_files:

                if item.startswith(common_text):
                    mask_file = item

            features = extract_features_one(folder_abnormal_test_split, image_file, preprocess)
            #output = torch.mean(torch.tensor(np.array(features)), dim=0)
            #print(np.array(output).shape)
            distances = [np.linalg.norm(features - vec) for vec in mean_feature]
            closest_index = np.argmin(distances)
            # print(distances)
            # print(closest_index)

            image_path = os.path.join(folder_abnormal_test_split, image_file)
            image_result_folder = os.path.join(path_test_cluster,str(closest_index),'test','clustered_abnormal')
            shutil.copy(image_path, image_result_folder)

            mask_path = os.path.join(folder_abnormal_test_mask_split, mask_file)
            mask_result_folder = os.path.join(path_test_cluster,str(closest_index),'ground_truth','clustered_abnormal')
            shutil.copy(mask_path, mask_result_folder)
    else:
        for image_file in image_files:

            features = extract_features_one(folder_abnormal_test_split, image_file, preprocess)
            #output = torch.mean(torch.tensor(np.array(features)), dim=0)
            #print(np.array(output).shape)
            distances = [np.linalg.norm(features - vec) for vec in mean_feature]
            closest_index = np.argmin(distances)

            image_path = os.path.join(folder_abnormal_test_split, image_file)
            image_result_folder = os.path.join(path_test_cluster,str(closest_index),'test','good')
            shutil.copy(image_path, image_result_folder)

