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
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

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

def scatter_cluster(numofcluster, pca_trans):
    
    for i in range(2, numofcluster):
        kmeans = KMeans(n_clusters=i , random_state=22)
        
        kmeans.fit(pca_trans)
        centroids = kmeans.cluster_centers_
        labels = kmeans.fit_predict(pca_trans)

        u_labels = np.unique(labels)

        for i in u_labels:
            plt.scatter(y[labels == i , 0] , y[labels == i , 1] , label = i)
        plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
        plt.legend()
        plt.show()

# def centor_cluster2(numofcluster, pca_trans, pca_trans2):

#     kmeans = KMeans(n_clusters=4 , random_state=22)
#     kmeans2 = KMeans(n_clusters=4 , random_state=22)
#     kmeans.fit(pca_trans)
#     kmeans2.fit(pca_trans2)
    
#     centroids = kmeans.cluster_centers_
#     centroids2 = kmeans2.cluster_centers_
    
#     labels = kmeans.fit_predict(pca_trans)
#     labels2 = kmeans2.fit_predict(pca_trans2)
    
#     u_labels = np.unique(labels)
#     u_labels2 = np.unique(labels2)

#     #for i in u_labels:
#     #    plt.scatter(x[labels == i , 0] , x[labels == i , 1] , label = i)
#     # for j in u_labels2:
#     #     plt.scatter(x[labels2 == j , 0] , x[labels2 == j , 1] , label2 = j)
#     plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
#     #plt.scatter(centroids2[:,0] , centroids2[:,1] , s = 80, color = 'c')
#     plt.legend()
#     plt.show()


def visualize_silhouette_layer(data, param_init='random', param_n_init=10, param_max_iter=300):
    clusters_range = range(2,15)
    results = []

    for i in clusters_range:
        clusterer = KMeans(n_clusters=i, init=param_init, n_init=param_n_init, max_iter=param_max_iter, random_state=0)
        cluster_labels = clusterer.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        results.append([i, silhouette_avg])

    result = pd.DataFrame(results, columns=["n_clusters", "silhouette_score"])
    pivot_km = pd.pivot_table(result, index="n_clusters", values="silhouette_score")

    plt.figure()
    sns.heatmap(pivot_km, annot=True, linewidths=.5, fmt='.3f', cmap=sns.cm._rocket_lut)
    plt.tight_layout()
    plt.show()
    
def visualize_elbowmethod(data, param_init='random', param_n_init=10, param_max_iter=300):
    distortions = []
    for i in range(1, 10):
        km = KMeans(n_clusters=i, init=param_init, n_init=param_n_init, max_iter=param_max_iter, random_state=0)
        km.fit(data)
        distortions.append(km.inertia_)

    plt.plot(range(1, 10), distortions, marker='o')
    plt.xlabel('Number of Cluster')
    plt.ylabel('Distortion')
    plt.show()


def splittofolder(n_cluster, folder_normal_train, path_normal_cluster, groups):
    # gets the list of filenames for a cluster
    for i in range(n_cluster):
        savepath = path_normal_cluster + '/' + str(i)
        files = groups[i]

        savepath_train = os.path.join(savepath, 'train')
        savepath_train_good = os.path.join(savepath, 'train', 'good')
        os.makedirs(savepath, exist_ok=True)
        os.makedirs(savepath_train, exist_ok=True)
        os.makedirs(savepath_train_good, exist_ok=True)

        for index, file in enumerate(files):
            copyfilepath = folder_normal_train + '/' + file
            savepath_file = savepath_train_good + '/' + file
            shutil.copyfile(copyfilepath, savepath_file)

def view_cluster(cluster):
    plt.figure(figsize = (15,15))
    # gets the list of filenames for a cluster
    files = groups[cluster]
    # only allow up to 30 images to preprocessbe shown at a time
    if len(files) > 100:
        print(f"Clipping cluster size from {len(files)} to 30")
        files = files[:99]
    # plot each image in the cluster
    for index, file in enumerate(files):
        plt.subplot(10,10,index+1)
        img = Image.open(file).convert('L')
        
        #data = Image.fromarray(img.astype(np.uint8))
        #img = np.array(data)
        #img = np.array(img)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

#==============================================================================================================================
# extract feature for  normal trainset based on DINO encoder

# wafer
#dataset = "wafer"
#n_clusters = 5

# dagm
# dataset = "dagm"
# n_clusters = 10

# magnetic
dataset = "magnetic"
n_clusters = 2

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

folder_normal_train = f"../test_data/{dataset}_confirm_ver4/train/10"

# this list holds all the image filename
image_normal_train = os.listdir(folder_normal_train)
preprocess = transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = mean,std = std),
                        ])
           
device = torch.device('cuda:3')
feature_extractor = torch.hub.load("facebookresearch/dino:main", "dino_vitb8").to(device)

data = {}
data2 = {}

feat = extract_features(folder_normal_train, image_normal_train, preprocess)
feat = np.array(feat)
nsamples, nx, ny = feat.shape
feat = feat.reshape((nx*ny,nsamples))
print(feat.shape)
# reduce the amount of dimensions in the feature vector
#pca = PCA(n_components=100, random_state=22)
#pca.fit(feat)
#y = pca.transform(feat)

dbscan = DBSCAN(eps=100, min_samples=1)

# 모델 학습
dbscan.fit(feat)

# 클러스터 레이블 얻기
labels = dbscan.labels_

# 클러스터링 결과 시각화
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # 잡음 포인트를 뺀 클러스터의 개수
print("클러스터 개수:", num_clusters)

# 각 클러스터링 결과 시각화
plt.figure(figsize=(12, 6))
plt.subplot(121)


plt.subplot(122)
plt.hist(labels, bins=num_clusters)
plt.title("클러스터링 결과")
plt.xlabel("클러스터 번호")
plt.ylabel("포인트 개수")
plt.show()

#==============================================================================================================================
# split normal trainset based on k-means clustering
# path_normal_cluster = f"data/{dataset}/lets_cluster"
# os.makedirs(path_normal_cluster, exist_ok=True)

# path_normal_cluster = f"data/{dataset}/lets_cluster/{dataset}_confirm_ver4_step1_1_multi_bank"
# os.makedirs(path_normal_cluster, exist_ok=True)

# kmeans = KMeans(n_clusters=n_clusters, random_state=22)
# kmeans.fit(y)
# # kmeans2 = KMeans(n_clusters=4 , random_state=22)
# # kmeans2.fit(y)

# groups = {}
# for file, cluster in zip(image_normal_train,kmeans.labels_):
#     if cluster not in groups.keys():
#         groups[cluster] = []
#         groups[cluster].append(file)
#     else:
#         groups[cluster].append(file)

# # view_cluster(0)
# # view_cluster(1)
# # view_cluster(2)
# # view_cluster(3)
# # view_cluster(4)
# #scatter_cluster(6,y)
# #splittofolder(n_clusters, folder_normal_train, path_normal_cluster, groups)
# #visualize_elbowmethod(y)
# #visualize_silhouette_layer(y)

# #==============================================================================================================================
# # split defect trainset based on l2 distance

# # extract features for each normal bank

# device = torch.device('cuda:3')
# feature_extractor = torch.hub.load("facebookresearch/dino:main", "dino_vitb8").to(device)


# folder_names = os.listdir(path_normal_cluster)
# mean_feature = []
# output_folders = []

# for i in range(len(folder_names)):
#     image_folder = os.path.join(path_normal_cluster, folder_names[i], 'train', 'good')
#     print(image_folder)
#     image_files = os.listdir(image_folder)
#     features = extract_features(image_folder, image_files, preprocess)
#     features = np.mean(features, axis=0)
#     mean_feature.append(features)
#     output_folders.append(str(i))

# print(np.array(mean_feature).shape)


# # Split abnormal trainset for each cluster(normal_bank)

# for folder in output_folders:
#     result_folder = os.path.join(path_normal_cluster,folder,'test')
#     os.makedirs(result_folder, exist_ok=True)

#     result_folder = os.path.join(path_normal_cluster,folder,'test','clustered_abnormal')
#     os.makedirs(result_folder, exist_ok=True)


# folder_abnormal_train = f"../test_data/{dataset}_confirm_ver4/train"
# image_abnormal_train = os.listdir(folder_abnormal_train)

# # erase normal_label path
# image_abnormal_train = [file for file in image_abnormal_train if not file.startswith('10')]
# print(image_abnormal_train)

# for i in tqdm(image_abnormal_train):

#     folder_abnormal_train_split = os.path.join(folder_abnormal_train, i)
#     image_files = os.listdir(folder_abnormal_train_split)

#     for image_file in image_files:

#         features = extract_features_one(folder_abnormal_train_split, image_file, preprocess)
#         # print(np.array(features).shape)
#         distances = [np.linalg.norm(features - vec) for vec in mean_feature]
#         closest_index = np.argmin(distances)
#         # print(distances)
#         # print(closest_index)

#         image_path = os.path.join(folder_abnormal_train_split, image_file)
#         result_folder = os.path.join(path_normal_cluster,str(closest_index),'test','clustered_abnormal')
        
#         shutil.copy(image_path, result_folder)


