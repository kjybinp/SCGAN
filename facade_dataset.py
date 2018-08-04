import os

import numpy
from PIL import Image
import six

import numpy as np

from io import BytesIO
import os
import pickle
import json
import numpy as np

import skimage.io as io

from chainer.dataset import dataset_mixin

# download `BASE` dataset from http://cmp.felk.cvut.cz/~tylecr1/facade/
class FacadeDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataDir='./ISTD', data_range=(1,300)):
        print("load dataset start")
        print("    from: %s"%dataDir)
        print("    range: [%d, %d)"%(data_range[0], data_range[1]))
        self.dataDir = dataDir
        self.dataset = []
        file_list_input = os.listdir(dataDir + '/train_A')
        file_list_mask = os.listdir(dataDir + '/train_B')
        file_list_removal = os.listdir(dataDir + '/train_C')
        for i in range(data_range[0],data_range[1]):
            file_nm = file_list_input[i]
            if ((file_nm in file_list_mask[i]) and (file_nm in file_list_removal[i]) ):
                img = Image.open(dataDir+"/train_A/"+file_nm)
                mask = Image.open(dataDir+"/train_B/"+file_nm)
                removal = Image.open(dataDir+"/train_C/"+file_nm)
                w,h = img.size
                r = 286 / float(min(w,h))
                # resize images so that min(w, h) == 286
                img = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
                mask = mask.resize((int(r*w), int(r*h)), Image.NEAREST)

                img = np.asarray(img).astype("f").transpose(2,0,1)/128.0-1.0
                removal = np.asarray(removal).astype("f").transpose(2, 0, 1) / 128.0 - 1.0
                mask_ = np.asarray(mask)/255# [0, 12)
                mask = np.zeros((2, img.shape[1], img.shape[2])).astype("i")
                for j in range(2):
                    mask[j,:] = mask_==j
                self.dataset.append((img,mask,removal))

        print("load dataset done. number of file: " + str(len(self.dataset)))
    
    def __len__(self):
        return len(self.dataset)

    # return (label, img)
    def get_example(self, i, crop_width=256):
        _,h,w = self.dataset[i][0].shape
        x_l = np.random.randint(0,w-crop_width)
        x_r = x_l+crop_width
        y_l = np.random.randint(0,h-crop_width)
        y_r = y_l+crop_width
        return self.dataset[i][0][:,y_l:y_r,x_l:x_r], self.dataset[i][1][:,y_l:y_r,x_l:x_r]
    
