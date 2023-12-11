from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import random
from torchvision.transforms import RandomCrop
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize

def Itk2array(nii_path,type):
    '''
    read nii.gz and convert to numpy array
    '''
    itk_data = sitk.ReadImage(nii_path,type)
    data = sitk.GetArrayFromImage(itk_data)
    return data


def CTNormalization(image, lower_bound,upper_bound,arrtype):
    image = image.astype(arrtype)
    image = np.clip(image, lower_bound, upper_bound)
    image = (image - lower_bound) / (upper_bound-lower_bound)
    return image

class Val_Dataset(dataset):
    def __init__(self, args):

        self.args = args

        self.data_path = args.val_path
        self.filename_list = os.listdir(os.path.join(self.data_path, 'label'))

        self.upper = args.upper
        self.lower = args.lower
        self.transforms = Compose([
                RandomCrop(self.args.crop_sizeXY,self.args.crop_sizeZ),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
            ])

    def __getitem__(self, index):
        paid = self.filename_list[index].split(".nii.gz")[0]

        imgfile = os.path.join(self.data_path,"img",paid+"_0.nii.gz")
        ctimg = Itk2array(imgfile, sitk.sitkInt16)

        reffile = os.path.join(self.data_path,"img",paid+"_1.nii.gz")
        refmask = Itk2array(reffile, sitk.sitkUInt8)

        segfile = os.path.join(self.data_path,"label",paid+".nii.gz")
        seg = Itk2array(segfile, sitk.sitkUInt8)

        ctimg = CTNormalization(ctimg,self.lower,self.upper,np.float32)
        refmask = refmask.astype(np.float32)

        ctimg = torch.FloatTensor(ctimg).unsqueeze(0)
        refmask = torch.FloatTensor(refmask).unsqueeze(0)      

        seg_array = torch.FloatTensor(seg).unsqueeze(0)

        if self.transforms:
            ct_array1,ct_array2,seg_array1 = self.transforms(ctimg,refmask, seg_array)  

        ct_array = torch.cat((ct_array1,ct_array2),dim=0)
        return ct_array, seg_array1.squeeze(0)

    def __len__(self):
        return len(self.filename_list)

