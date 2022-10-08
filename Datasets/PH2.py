import os
import PIL
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from PIL import Image
from utils.transform import itensity_normalize
from torch.utils.data.dataset import Dataset


class ph2_dataset(Dataset):
    def __init__(self, dataset_folder='/data/project_ddw/0112_skin_lesion_segment/CA-Net-master/data/PH2_npy_224_320',
                       image_list = "/data/project_ddw/0112_skin_lesion_segment/CA-Net-master/data/PH2_npy_224_320/ph2_list.txt",
                       train_type='train',
                       with_name=True, 
                       transform=None):
        
        self.train_type = train_type
        self.transform = transform
        self.with_name = with_name
        self.image_list = image_list
        
        with open(self.image_list, "r") as ff:
            self.image_list = ff.readlines()
        
        self.image_list = [item.replace('\n', '') for item in self.image_list]
     
        self.img_folder = [join(dataset_folder, 'image', x) for x in self.image_list]
        self.mask_folder = [join(dataset_folder, 'label', x.split('.')[0] + '_segmentation.npy') for x in self.image_list]        
            
        assert len(self.img_folder) == len(self.mask_folder)   
            
            
    def __getitem__(self, item: int):
        image = np.load(self.img_folder[item])
        label = np.load(self.mask_folder[item])
        label[label<125] = 0
        label[label>=125] = 255
        name = self.img_folder[item].split('/')[-1]

        sample = {'image': image, 'label': label}

        if self.transform is not None:
            # TODO: transformation to argument datasets  
            sample = self.transform(sample, self.train_type)
            
        if self.with_name:
            return name, sample['image'], sample['label']    
        else:
            return sample['image'], sample['label']

    def __len__(self):
        return len(self.img_folder)

