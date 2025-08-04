import torch
import os
import math
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import natsort
class MyDataset(Dataset):
  def __init__(self, dataset_path="/Users/jiin/Desktop/mvtec",
                 class_name='brats', is_train=True, cropsize=224, resize=256, have_gt=True, random=False, is_defect_aug=False):
#      assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
    self.dataset_path = dataset_path
    self.class_name = class_name
    self.is_train = is_train
    self.cropsize = cropsize
    self.resize = resize
    self.have_gt = have_gt
    self.random = random
    self.is_defect_aug = is_defect_aug
    if self.is_train | (not self.have_gt):
      self.x, self.y = self.load_dataset_folder()
    else:
      self.x, self.y, self.mask = self.load_dataset_folder()
    self.transform_x = transforms.Compose([transforms.Resize((self.resize,self.resize), Image.LANCZOS),
                                           transforms.CenterCrop(self.cropsize),
                                           transforms.ToTensor(),
                                           #transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            #                    std=[0.5, 0.5, 0.5])])
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
                                          
                                           
    self.transform_mask = transforms.Compose([transforms.Resize((self.resize,self.resize), Image.NEAREST),
                                              transforms.CenterCrop(self.cropsize),
                                              transforms.ToTensor()])
  def __len__(self):
    return len(self.x)
  def __getitem__(self, idx):
    x, y = self.x[idx], self.y[idx]
    x = Image.open(x).convert('RGB')
    x = self.transform_x(x)
    if self.is_train | (not self.have_gt):
      return x, y
    else:
      mask = self.mask[idx]
      if y == 0:
        mask = torch.zeros([1, self.cropsize, self.cropsize], dtype=torch.int32)
      else:
        mask = Image.open(mask).convert('L')
        mask = self.transform_mask(mask)
        mask = torch.ceil(mask)
        mask = mask.type(torch.int32)
      return x, y, mask
  # dataset 기본 class
  def load_dataset_folder(self):
    np.random.seed(0)
    random.seed(0)
    phase = 'train' if self.is_train else 'test'
    img_dir = os.path.join(self.dataset_path, self.class_name, phase)
    if self.is_train | (not self.have_gt):
      x, y = [], []
    else:
      gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
      x, y, mask = [], [], []
    img_types = sorted(os.listdir(img_dir)) # class 별로 있으므로 이를 정렬
    if not self.is_defect_aug:
      number = len(os.listdir(os.path.join(img_dir, 'good')))
    if self.random:
      anomaly_x = []
      if not self.is_train:
        anomaly_mask = []
    for img_type in img_types:
      img_type_dir = os.path.join(img_dir, img_type)
     
      if not os.path.isdir(img_type_dir):
        continue
      # img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.jpg') or f.endswith('.png')])
      img_fpath_list = natsort.natsorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.jpg') or f.endswith('.png')])
      img_filename_list = os.listdir(img_type_dir)
      if self.is_train | (not self.have_gt):
        x.extend(img_fpath_list)
        if img_type == 'good':
          y.extend([0] * len(img_fpath_list))
        else:
          y.extend([1] * len(img_fpath_list))
      # y labeling
      else:
        if img_type == 'good':
          x.extend(img_fpath_list)
          y.extend([0] * len(img_fpath_list))
          mask.extend([None] * len(img_fpath_list))
        else:
          gt_type_dir = os.path.join(gt_dir, img_type)
          # img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
          img_fname_list = os.listdir(gt_type_dir)
          gt_fpath_list = natsort.natsorted([os.path.join(gt_type_dir, img_fname) for img_fname in img_fname_list])
          #print(img_fpath_list)
          #print(gt_fpath_list)
          if self.random:
            anomaly_x.extend(img_fpath_list)
            #print(x)
            anomaly_mask.extend(gt_fpath_list)
          else:
            x.extend(img_fpath_list)
           
            y.extend([1] * len(img_fpath_list)) # abnormal
            mask.extend(gt_fpath_list)
    if self.random and (not self.is_defect_aug):
      _list = []
      if len(anomaly_x) > int(0.1*number):
        for i in range(int(0.1*number)):
          idx = np.random.randint(len(anomaly_x))
          while idx in _list:
            idx = np.random.randint(len(img_fpath_list))
          _list.append(idx)
          x.append(anomaly_x[idx])
          y.append(1)
          mask.append(anomaly_mask[idx])
      else:
        x.extend(anomaly_x)
        y.extend([1] * len(anomaly_x))
        mask.extend(anomaly_mask)
#    assert len(x) == len(y), 'number of x and y should be same'
    #print(img_fpath_list)
    if self.is_train | (not self.have_gt):
      return list(x), list(y)
    else:
      return list(x), list(y), list(mask)