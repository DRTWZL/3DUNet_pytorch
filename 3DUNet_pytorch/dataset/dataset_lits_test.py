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
    return data,itk_data


def CTNormalization(image, lower_bound,upper_bound,arrtype):
    image = image.astype(arrtype)
    image = np.clip(image, lower_bound, upper_bound)
    image = (image - lower_bound) / (upper_bound-lower_bound)
    return image

class Test_Dataset(dataset):
    def __init__(self, args,filename_list):
        self.args = args

        self.data_path = args.test_data_path
        self.filename_list = filename_list

        self.upper = args.upper
        self.lower = args.lower

    def __getitem__(self, index):

        paid = self.filename_list[index].split(".nii.gz")[0]

        imgfile = os.path.join(self.data_path,"img",paid+"_0.nii.gz")
        ctimg,itkdata = Itk2array(imgfile, sitk.sitkInt16)

        reffile = os.path.join(self.data_path,"img",paid+"_1.nii.gz")
        refmask,itkdata = Itk2array(reffile, sitk.sitkUInt8)

        ctimg = CTNormalization(ctimg,self.lower,self.upper,np.float32)
        refmask = refmask.astype(np.float32)

        ctimg = torch.FloatTensor(ctimg).unsqueeze(0)
        refmask = torch.FloatTensor(refmask).unsqueeze(0)      

        ct_array = torch.cat((ctimg,refmask),dim=0)

        return ct_array

    def __len__(self):
        return len(self.filename_list)


if __name__ == "__main__":
    sys.path.append('/home/img/code/3DUNet_pytorch')
    from config import args
    test_ds = Test_Dataset(args)

    # 定义数据加载
    test_dl = DataLoader(test_ds, 1, num_workers=0,shuffle=False )

    for i, (ct, paid,seg) in enumerate(test_dl):
        print(i,ct.size(),paid,seg.size())