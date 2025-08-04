from collections import defaultdict
import numpy as np
import torchvision
import torch
from PIL import Image
import random

class IMBALANCEMNIST(torchvision.datasets.MNIST):
    cls_num = 10

    def __init__(self, root, imb_factor=None, rand_number=0, train=True,
                 transform=None, target_transform=None, download=True, cmo=False, patch=False, head_num=None, use_randaug=False, weighted_alpha=1):
        super().__init__(root, train, transform, target_transform, download)

        self.use_randaug = use_randaug
        self.weighted_alpha = weighted_alpha
        self.cmo = cmo
        self.head_num = head_num
        self.patch = patch

        if self.patch:
            print("==================check1===================")
            targets_np = np.array(self.targets, dtype=np.int64)
            print(self.data.shape)
            print(targets_np.shape)

            del_index = np.where(targets_np==0)[0]
            targets_np = np.delete(targets_np, del_index)
            print("==================check2===================")
            self.data = np.delete(self.data, del_index, axis=0)
            print(self.data.shape)
            print(targets_np.shape)
            targets_np -= 1

            self.targets = targets_np.tolist()

        if train and imb_factor is not None:
            np.random.seed(rand_number)
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_factor, self.head_num, self.patch)
            self.gen_imbalanced_data(img_num_list)
            

        self.classnames = self.classes
        self.labels = self.targets
        self.cls_num_list = self.get_cls_num_list()
        self.num_classes = len(self.cls_num_list)

        print(self.classnames)

        if self.patch:
            self.classnames.pop(0)

        print(self.classnames)

        print("=================check==============")

        # print(self.classnames)
        # print(self.num_classes)

        # self.cls_num_list.insert(0, 0)
        # self.num_classes +=1

    def get_img_num_per_cls(self, cls_num, imb_factor, head_num, patch):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
        if patch:
            img_num_per_cls.pop(0)
        if head_num!=None:
            img_num_per_cls[0] = head_num
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        counter = defaultdict(int)
        for label in self.labels:
            counter[label] += 1
        labels = list(counter.keys())
        labels.sort()
        cls_num_list = [counter[label] for label in labels]
        return cls_num_list

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # image = Image.fromarray(image, 'RGB')
        image = Image.fromarray(image)
        transform = torchvision.transforms.Grayscale(3)


        if self.use_randaug:
            image = transform(image)
            r = random.random()
            if r < 0.5:
                image = self.transform[0](image)
            else:
                image = self.transform[1](image)
        else:
            if self.transform is not None:
                image = transform(image)
                image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def get_weighted_sampler(self):
        cls_num_list = self.cls_num_list
        cls_weight = 1.0 / (np.array(cls_num_list) ** self.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in self.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(self.targets), replacement=True)
        return sampler


class MNIST(IMBALANCEMNIST):
    def __init__(self, root, train=True, transform=None, cmo=False, patch=False, head_num=None, use_randaug=False, weighted_alpha=1):
        super().__init__(root, imb_factor=None, train=train, transform=transform, cmo=cmo, patch=patch, head_num=head_num, use_randaug=use_randaug, weighted_alpha=weighted_alpha)


class MNIST_IR10(IMBALANCEMNIST):
    def __init__(self, root, train=True, transform=None, cmo=False, patch=False, head_num=None, use_randaug=False, weighted_alpha=1):
        super().__init__(root, imb_factor=0.1, train=train, transform=transform, cmo=cmo, patch=patch, head_num=head_num, use_randaug=use_randaug, weighted_alpha=weighted_alpha)


class MNIST_IR50(IMBALANCEMNIST):
    def __init__(self, root, train=True, transform=None, cmo=False, patch=False, head_num=None, use_randaug=False, weighted_alpha=1):
        super().__init__(root, imb_factor=0.02, train=train, transform=transform, cmo=cmo, patch=patch, head_num=head_num, use_randaug=use_randaug, weighted_alpha=weighted_alpha)


class MNIST_IR100(IMBALANCEMNIST):
    def __init__(self, root, train=True, transform=None, cmo=False, patch=False, head_num=None, use_randaug=False, weighted_alpha=1):
        super().__init__(root, imb_factor=0.01, train=train, transform=transform, cmo=cmo, patch=patch, head_num=head_num, use_randaug=use_randaug, weighted_alpha=weighted_alpha)
