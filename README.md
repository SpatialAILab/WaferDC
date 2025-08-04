# Long-tailed Detection and Classification of Wafer Defects from SEM Images Robust to Diverse Image Backgrounds and Defect Scales

Official Pytorch implementation of [Long-tailed Detection and Classification of Wafer Defects from SEM Images Robust to Diverse Image Backgrounds and Defect Scales] (Under review).

Authors: Taekyeong Park*, Yongho Son*, Sanghyuk Moon*, Seungju Han, Je Hyeong Hong


## Abstract
In semiconductor engineering, high yield of wafers relies on accurate detection and classification of wafer defects.
The dataset for detecting wafer defects presents three primary challenges: i) different background types, ii) variable image or defect scales, and iii) imbalanced data with a long-tailed distribution of defect types. These challenges create significant limitations for traditional classification techniques. To address these issues, we propose a stratified framework called WaferDC, designed specifically for detecting and classifying wafer defects from scanning electron microscope (SEM) images.
Our framework achieves high defect detection performance on SEM wafer images by utilizing a multi-cluster memory bank, which effectively handles the challenges of i) variable background types and ii) differing image or defect scales.
Building on this robust detection, we propose SegMix, a novel defect augmentation technique based on anomaly heatmaps, which enhances the reliability of defect detection and classification in a iii) long-tailed imbalanced environment. 
Finally, we pass defect-classified images through a parameter-efficient fine-tuning (PEFT)-based classifier utilizing a vision transformer (ViT) architecture, further improving overall defect detection and classification performance.
We rigorously tested WaferDC on a proprietary SEM wafer dataset and the public DTD-Synthetic and Magnetic Tile Defect (MTD) datasets. The results confirm the effectiveness of our method in improving defect detection and classification in wafer manufacturing.

## Contribution
- We introduce WaferDC, designed to address the three challenges mentioned above and enhance wafer defect detection and classification accuracies in SEM images.

- To improve the performance of wafer defect detection across various SEM image backgrounds and scales, we propose a novel multi-cluster memory bank strategy, where each cluster represents a different background type and scale in wafer defects. This also contributes to the enhanced quality of augmented defect images later on.

- We introduce a new defect augmentation method called SegMix, which utilizes the anomaly segmentation map from the multi-cluster memory bank to extract precise defect areas and adopts smooth blending to synthesize realistic SEM images with defects.
As will be shown later, the added synthetic data is not just confined to train the defect classifier but also used to adjust defect detection thresholds for the multi-cluster memory bank, leading to an overall enhanced framework performance.

- We have extensively evaluated WaferDC on a proprietary SEM wafer dataset and public DTD-Synthetic and the Magnetic Tile Defect (MTD) datasets. Our experimental results demonstrate the capability of our approach in enhancing defect detection and classification performance.

## Requirements:
Our results were computed using Python 3.8, with packages and respective version noted in requirements.txt
````
conda create -n waferdc python=3.8 -y
conda activate waferdc

pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
conda install pytorch::faiss-gpu

git clone https://github.com/P-taetae/Wafer_defect_detection.git
cd Wafer_defect_detection
pip install -r requirement.txt
````

## Data preparation:
Download the following dataset:
- MTD dataset [[our link]](https://drive.google.com/file/d/1HbOv2rG2ODKjGvFx4wYm3iI01cOCOsRR/view?usp=sharing)

## How to run our code
1. defect_detection

Follow the 1step_defect_detection

Training :
```
cd 1step_defect_detetion/
python step1_1_train_k_means.py --train_data_path your_train_data_path --test_data_path your_test_data_path
python step1_2_normal_augmentation.py --dataset dataset_name --n_clusters num_of_chosen_clusters
bash step1_3_run_patchcore_magnetic.sh
bash step1_4_abnormal_augmentation_magnetic.sh
python step1_5_reset_folder.py
```

Testing :
```
python test1_1_k_means.py
bash test1_2_load_and_evaluate_magnetic.sh
```

2. defect_classification


After the 1step_defect_detection

Use following command to train and test the model.
```
python main.py
```

This code is written based on the following code : https://github.com/amazon-science/patchcore-inspection , https://github.com/shijxcs/LIFT

## Citation
---

## License
This project is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.](https://creativecommons.org/licenses/by-nc-nd/4.0/)

## Acknowledgement
This work was in part supported by Samsung Advanced Institute of Technology, Samsung Electronics Co., Ltd, in part by the Technology Innovation Program (1415178807, Development of Industrial Intelligent Technology for Manufacturing, Process, and Logistics) funded by the Ministry of Trade, Industry & Energy (Korea) and in part by the Institute of Information & communications Technology Planning & Evaluation (IITP) under the artificial intelligence semiconductor support program to nurture the best talents (IITP-(2024)-RS-2023-00253914) grant funded by the Korean government.
