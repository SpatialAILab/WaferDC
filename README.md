# [EAAI 2025] Long-tailed Detection and Classification of Wafer Defects from SEM Images Robust to Diverse Image Backgrounds and Defect Scales

Official Pytorch implementation of [Long-tailed Detection and Classification of Wafer Defects from SEM Images Robust to Diverse Image Backgrounds and Defect Scales] (Engineering Applications of Artificial Intelligence)

Authors: Taekyeong Park*, Yongho Son*, Sanghyuk Moon*, Seungju Han, and Je Hyeong Hong
<br>[Spatial AI Lab](https://sail.hanyang.ac.kr/). Hanyang University.<br>

<a href="https://pdf.sciencedirectassets.com/271095/1-s2.0-S0952197625X00300/1-s2.0-S0952197625023504/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjENH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIA5AGuqBiq5n1DyzdGu%2BuFLcjkP5fCce8hefn1OdmAVPAiA4kF6lBVlUSNL4tQ8KSktMSEftwrjcviSge4q6dYCr%2BCqyBQhZEAUaDDA1OTAwMzU0Njg2NSIMaMPPSTdGdI02Q5ruKo8FZK6NrOuAHBYymXEkLtyWGwFaTCs9uMMhkUA5FZQ2vy2t1akgRH%2FWCrjKLTwEodZCZUWsSrpq0ZHuoYlD%2FdFx1I8gVudtm0VdnWkcCjSJUUOa8%2B9xUM5v1wTN0bNiH5RwjqdWJsnI5vHrkwGH7wwH%2Bdb4H%2FHF7BWeVs%2BC%2BwHK3aoTl8eVwpA%2B7Vpby23mD6AaAn5qmiaZIqnQy6xjFf%2BG%2BzepFecTMO2bxS6xHxI1Wyak64MpPVBBKPQwPVJcVpgFlJvR%2F95%2BrUQra%2FMu6IJg1i4i538SgMFQ9nqwr6TVRUqM62tRpGfbL4r%2FXerRjVGR%2FHas%2Bo7YRtHfWXWl6PBUX%2BZ44VA%2F33SjADV5Mze6Mykb0ezCgrPNOf83FM9mj%2FUPC%2Frb90SDW9j8p0QKKKPx5QRLuUttECo80hst1Zi4fow%2FxQo1b1hXDHEaelS9eWCnwcO3Zq8brroSxpFsyBajlIQhsQ8fXNqOajQfR9waM4jKtKhOxQVdNs4ZzchJVDVveJACeLLbH8KQxOx%2BQ%2F2fcrxpS39tw2HZLB0xscxHJYlc%2FoTMZcnOgdAt93bRyP6PwYRdfNC%2FE7LEeTAKLk3FkxML55rKyWDciEMzUYvkwhZP1hwEL4uYri3X6Q4w7p%2FQJgLFkKN0vmtt%2BNC9zCBG%2Bf69n8tDtRXMLC9SpJF3R4cAemDNOmWWK7DRnovX83GfOdw3G7QleIFsITAMQtpF%2F7YA7gLZgD3W8gMGnMwoxeoRsTI9OUf%2F7SYncZN7WFahn6ORFwRwGZ0%2FKrDlFYGw30OG8oNZYuCIM2CAe1pvQg9eoNwpaUuCRpldAuaRdcpbXtPNP5OV2jwCfX8CNoEt4LfYR80QnrmfbSZ6YKC%2BbjDI0s7GBjqyAbkiYr8nCkMjI2SqtLRHeviPzFoOZbqAKeegKcsOjjEKqrpDy95wfagssYBA74LN6T0bsVmiuRj5EquZwToEoCD2kfTPWyjBE2IOgpguIs699DtiV1CL9s4UDLjKbDcAv2febDo%2FnLo%2B1QnwS8EZJ8o%2F9cqJnTZtyPbDAYhk9uK%2BkhbtVXzBffpOSxUL55M3bPvjaRWH7tzpBioGqIG8d%2F7kkj0A3GNtqXnbPtWd65K2fTI%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250924T091205Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYW3PRRPX5%2F20250924%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=354c4306b8b5d824d19c3e9d85a2d14a1c729ce4a1fad8f5508198b5a9ac92ad&hash=7ed62bec5fbe533ea9f12aba394c5bacc02aca541aa2e4256612b34f726b8f51&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0952197625023504&tid=spdf-760ad74e-2f3e-41a9-ad29-5c05a1d82854&sid=05e00d3212b7d64305484334be6449c166f0gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=0514595a5454575e50&rr=984128777a1b961c&cc=kr">
    <img src="https://img.shields.io/badge/Journal-EAAI-red?logo=arXiv" alt="arXiv">
</a>

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
