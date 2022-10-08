#!/usr/bin/python3
# these code is for ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection
# -*- coding: utf-8 -*-
# @Author  : Duwei Dai

import os
import random
import numpy as np
from skimage import io
from PIL import Image

root_dir = '/data/ddw_data/skin-image/ISIC_2018'                # change it in your saved original data path
save_dir = './data/ISIC2018_npy_224_224'


if __name__ == '__main__':
    imgfile = os.path.join(root_dir, 'ISIC2018_Task1-2_Training_Input')
    labfile = os.path.join(root_dir, 'ISIC2018_Task1_Training_GroundTruth')
    filename = sorted([os.path.join(imgfile, x) for x in os.listdir(imgfile) if x.endswith('.jpg')])
    random.shuffle(filename)
    labname = [filename[x].replace('ISIC2018_Task1-2_Training_Input', 'ISIC2018_Task1_Training_GroundTruth'
                                   ).replace('.jpg', '_segmentation.png') for x in range(len(filename))]

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir+'/image')
        os.makedirs(save_dir+'/label')
    
    kk = 0
    for i in range(len(filename)):
        fname = filename[i].rsplit('/', maxsplit=1)[-1].split('.')[0]
        lname = labname[i].rsplit('/', maxsplit=1)[-1].split('.')[0]

        image = Image.open(filename[i])
        label = Image.open(labname[i])

        image = image.resize((224, 224))
        label = label.resize((224, 224))
        image = np.array(image)
        label = np.array(label)

        images_img_filename = os.path.join(save_dir, 'image', fname)
        labels_img_filename = os.path.join(save_dir, 'label', lname)
        np.save(images_img_filename, image)
        np.save(labels_img_filename, label)
        kk += 1
        print("KK: ", kk)
        
    print('Successfully saved preprocessed data')
