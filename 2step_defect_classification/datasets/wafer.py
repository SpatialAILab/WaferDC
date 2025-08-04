import os
from .lt_data import LT_Dataset
import numpy as np
import torchvision
import torch

class wafer(LT_Dataset):

    # 1step
    # classnames_txt = "./datasets/wafer/wafer_confirm_ver7_ver3_1step_nonpatch/classnames.txt"
    # train_txt = "./datasets/wafer/wafer_confirm_ver7_ver3_1step_nonpatch/Wafer_LT_train.txt"
    # test_txt = "./datasets/wafer/wafer_confirm_ver7_ver3_1step_nonpatch/Wafer_LT_test.txt" 

    # classnames_txt = "./datasets/wafer/wafer_confirm_ver7_ver3_1step_patch/classnames.txt"
    # train_txt = "./datasets/wafer/wafer_confirm_ver7_ver3_1step_patch/Wafer_LT_train.txt"
    # test_txt = "./datasets/wafer/wafer_confirm_ver7_ver3_1step_patch/Wafer_LT_test.txt" 

    classnames_txt = "./datasets/wafer/wafer_test/classnames.txt"
    train_txt = "./datasets/wafer/wafer_test/Wafer_LT_train.txt"
    test_txt = "./datasets/wafer/wafer_test/Wafer_LT_test.txt"

    # classnames_txt = "./datasets/wafer/wafer_confirm_ver7_ver3_half_5std/classnames.txt"
    # train_txt = "./datasets/wafer/wafer_confirm_ver7_ver3_half_5std/Wafer_LT_train.txt"
    # test_txt = "./datasets/wafer/wafer_confirm_ver7_ver3_half_5std/Wafer_LT_test.txt"

    # classnames_txt = "./datasets/wafer/wafer_confirm_ver7_ver3_half_3std_underbalance/classnames.txt"
    # train_txt = "./datasets/wafer/wafer_confirm_ver7_ver3_half_3std_underbalance/Wafer_LT_train.txt"
    # test_txt = "./datasets/wafer/wafer_confirm_ver7_ver3_half_3std_underbalance/Wafer_LT_test.txt"


    # def __init__(self, root, train=True, transform=None):
    def __init__(self, root, imb_factor=None, rand_number=0, train=True,
        transform=None, target_transform=None, download=True, cmo=False, head_num=None, use_randaug=False, weighted_alpha=1):

        self.use_randaug = use_randaug
        self.weighted_alpha = weighted_alpha
        self.cmo = cmo
        
        super().__init__(root, train, transform, target_transform, use_randaug)

        self.classnames = self.read_classnames()

        self.names = []
        with open(self.txt) as f:
            for line in f:
                self.names.append(self.classnames[int(line.split()[1])])

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        # name = self.names[index]
        return image, label

    @classmethod
    def read_classnames(self):
        classnames = []
        with open(self.classnames_txt, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames.append(classname)
        return classnames

    def get_weighted_sampler(self):
        # cls_num_list = super().cls_num_list
        cls_num_list = self.cls_num_list
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in self.labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.labels), replacement=True)
        return sampler

