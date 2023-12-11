"""
This part is based on the dataset class implemented by pytorch, 
including train_dataset and test_dataset, as well as data augmentation
"""
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize

#----------------------data augment-------------------------------------------
class Resize:
    def __init__(self, scale):
        # self.shape = [shape, shape, shape] if isinstance(shape, int) else shape
        self.scale = scale

    def __call__(self, img, ref, mask):
        img, ref,mask = img.unsqueeze(0), ref.unsqueeze(0).float(), mask.unsqueeze(0).float()
        img = F.interpolate(img, scale_factor=(1,self.scale,self.scale),mode='trilinear', align_corners=False, recompute_scale_factor=True)
        ref = F.interpolate(ref, scale_factor=(1,self.scale,self.scale), mode="nearest", recompute_scale_factor=True)
        mask = F.interpolate(mask, scale_factor=(1,self.scale,self.scale), mode="nearest", recompute_scale_factor=True)

        return img[0], ref[0], mask[0]

class RandomResize:
    def __init__(self,s_rank, w_rank,h_rank):
        self.w_rank = w_rank
        self.h_rank = h_rank
        self.s_rank = s_rank

    def __call__(self, img, ref,mask):
        random_w = random.randint(self.w_rank[0],self.w_rank[1])
        random_h = random.randint(self.h_rank[0],self.h_rank[1])
        random_s = random.randint(self.s_rank[0],self.s_rank[1])
        self.shape = [random_s,random_h,random_w]
        img, mask = img.unsqueeze(0), mask.unsqueeze(0).float()
        img = F.interpolate(img, size=self.shape,mode='trilinear', align_corners=False)
        ref = F.interpolate(ref, size=self.shape, mode="nearest")
        mask = F.interpolate(mask, size=self.shape, mode="nearest")
        return img[0], ref[0].long(), mask[0].long()

class RandomCrop:
    def __init__(self, cropXY,cropZ):
        self.cropXY = cropXY
        self.cropZ = cropZ

    def _get_range(self, bs,zs,ys,xs):
        if xs < self.cropXY:
            Xsart = 0
        else:
            Xsart = random.randint(0, xs - self.cropXY)
        if ys < self.cropXY:
            Ysart = 0
        else:
            Ysart = random.randint(0, ys - self.cropXY)

        if zs < self.cropZ:
            Zsart = 0
        else:
            Zsart = random.randint(0, zs - self.cropZ)
        return Xsart, Ysart, Zsart

    def __call__(self, img, ref, mask):
        bs,zs,ys,xs = img.shape
        Xsart, Ysart, Zsart = self._get_range(bs,zs,ys,xs)      
        # print("Orisize: ",img.shape,ref.shape, mask.shape)
        # print("Cropsize: ",Xsart, Ysart, Zsart)
    
        tmp_img = torch.zeros((img.size(0), self.cropZ, self.cropXY, self.cropXY))
        tmp_ref = torch.zeros((ref.size(0), self.cropZ, self.cropXY, self.cropXY))
        tmp_mask = torch.zeros((mask.size(0), self.cropZ, self.cropXY, self.cropXY))
        tmp_img[:,:,:,:] = img[:,Zsart:Zsart+self.cropZ,Ysart:Ysart+self.cropXY,Xsart:Xsart+self.cropXY]
        tmp_ref[:,:,:,:] = ref[:,Zsart:Zsart+self.cropZ,Ysart:Ysart+self.cropXY,Xsart:Xsart+self.cropXY]
        tmp_mask[:,:,:,:] = mask[:,Zsart:Zsart+self.cropZ,Ysart:Ysart+self.cropXY,Xsart:Xsart+self.cropXY]
        
        return tmp_img,tmp_ref, tmp_mask

class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = img.flip(2)
        return img

    def __call__(self, img, ref,mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(ref, prob),self._flip(mask, prob)

class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[1] <= self.prob:
            img = img.flip(3)
        return img

    def __call__(self, img, ref, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(ref, prob), self._flip(mask, prob)

class RandomRotate:
    def __init__(self, max_cnt=3):
        self.max_cnt = max_cnt

    def _rotate(self, img, cnt):
        img = torch.rot90(img,cnt,[1,2])
        return img

    def __call__(self, img, ref, mask):
        cnt = random.randint(0,self.max_cnt)
        return self._rotate(img, cnt), self._rotate(ref, cnt),self._rotate(mask, cnt)

class Center_Crop:
    def __init__(self, base, max_size):
        self.base = base  # 
        self.max_size = max_size 
        if self.max_size%self.base:
            self.max_size = self.max_size - self.max_size%self.base 
    def __call__(self, img ,ref, label):
        if img.size(1) < self.base:
            return None
        slice_num = img.size(1) - img.size(1) % self.base
        slice_num = min(self.max_size, slice_num)

        left = img.size(1)//2 - slice_num//2
        right =  img.size(1)//2 + slice_num//2

        crop_img = img[:,left:right]
        crop_ref = ref[:,left:right]
        crop_label = label[:,left:right]
        return crop_img, crop_ref,crop_label

class ToTensor:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, mask):
        img = self.to_tensor(img)
        mask = torch.from_numpy(np.array(mask))
        return img, mask[None]

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, ref,mask):
        for t in self.transforms:
            img,ref, mask = t(img, ref,mask)
        return img, ref, mask