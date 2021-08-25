import torch
from model import HAN
from dataset import Dataset
from process import train
import config

if __name__ == '__main__':
    net = HAN().cuda()
    checkpoint = torch.load("/content/drive/MyDrive/project/Super_resolution/runs/HAN_BIX4.pt")
    net.load_state_dict(checkpoint)
    train_data = Dataset(config.HR_TRAIN,config.LR_TRAIN)
    val_data = Dataset(config.HR_VAL,config.LR_VAL)
    checkpoint_last,checkpoint_best=train(net,train_data,val_data)
    torch.save(checkpoint_last,"/content/drive/MyDrive/project/Super_resolution/runs/last.pth")
    torch.save(checkpoint_best,"/content/drive/MyDrive/project/Super_resolution/runs/best.pth")