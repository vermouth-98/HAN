
import torch
import os
import torchvision,config
import matplotlib.pyplot as plt
import cv2,random
import numpy as np
def random_crop(LR,HR):
    x = random.choice(np.arange(0,LR.shape[1]//2))
    y = random.choice(np.arange(0,LR.shape[2]//2))
    rect = torch.tensor([x,y,config.SIZE,config.SIZE])
    rect2 = rect*4
    LR = torchvision.transforms.functional.crop(LR,*rect)
    HR = torchvision.transforms.functional.crop(HR,*rect2)
    
    return LR,HR
def transfrom(LR,HR):
    """
    LR,HR: img type cv2.YCR_CB
    
    """
    Flip = [0,1]
    Degree = [90,180,270]
    flip_flag = random.choice(Flip)
    degree_flag = random.choice(Degree)
    transforms_head = torchvision.transforms.ToTensor()
    LR= transforms_head(LR.astype(np.float32))
    HR= transforms_head(HR.astype(np.float32))
    LR,HR = random_crop(LR,HR)
    LR= torchvision.transforms.functional.rotate(LR,degree_flag)
    HR= torchvision.transforms.functional.rotate(HR,degree_flag)
    if flip_flag:
        LR = torchvision.transforms.functional.hflip(LR)
        HR = torchvision.transforms.functional.hflip(HR)
    
    return LR,HR
class Dataset(torch.utils.data.Dataset):
    def __init__(self,HR_path,LR_path):
        self.LR_path = LR_path
        self.HR_path = HR_path
        self.LR_list_name =os.listdir(self.LR_path)
        self.HR_list_name =os.listdir(self.HR_path)
        
    def __len__(self):
        return len(self.HR_list_name)
    def __getitem__(self,idx):
        LR = cv2.imread(os.path.join(self.LR_path,self.LR_list_name[idx]))
        HR = cv2.imread(os.path.join(self.HR_path,self.HR_list_name[idx]))
        LR = cv2.cvtColor(LR, cv2.COLOR_BGR2YCrCb)
        HR = cv2.cvtColor(HR, cv2.COLOR_BGR2YCrCb)
        LR,HR = transfrom(LR,HR)

        return LR,HR
