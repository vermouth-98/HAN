import torch
from labml import logger
import numpy as np
import config
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset,random_split,SubsetRandomSampler,DataLoader
def EvalLoss(net,val_loader):
    net.eval()
    accuracy = {
        "loss": 0,
        "number":0,
    }
    criterion=torch.nn.MSELoss(reduction = "mean")
    for features,labels in val_loader:
        features,labels  = features.to(config.DEVICE),labels.to(config.DEVICE)
        predict = net(features)
        loss = criterion(predict,labels)
        accuracy["loss"]+= loss.item()
        accuracy["number"]+= 1
    del features,labels
    return accuracy["loss"]/accuracy["number"]


def train(net,train_data,val_data):
    
    # scaler = torch.cuda.amp.GradScaler()
    torch.manual_seed(42)
    dataset = ConcatDataset([train_data,val_data])
    criterion = torch.nn.MSELoss(reduction = "mean")
    optimizer = torch.optim.Adam(net.parameters(),lr = config.LEARNING_RATE,weight_decay=config.WEIGHT_DECAY)
    splits = KFold(n_splits = config.K_FOLD,shuffle=True,random_state =42)
    best_loss = 1000
    for fold,(train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        logger.log("Fold {}".format(fold + 1))
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, sampler=val_sampler)
        for epoch in range(config.NUM_EPOCH):
            loop = tqdm(train_loader,position= 0,leave = True)
            net.train()
            for features,labels in loop:
                features,labels  = features.to(config.DEVICE),labels.to(config.DEVICE)
                optimizer.zero_grad()
                
                predict = net(features)
                loss = criterion(predict,labels)
                    
                loss.backward()
                optimizer.step()
                loop.set_description("Epoch: {}. Loss: {:.5f}".format(epoch + 1,loss.item()))
            del features
            del labels
            loss_val = EvalLoss(net,val_loader)
            logger.log("loss in validation: {:.5f}".format(loss_val))
            if loss_val<best_loss:
                best_loss = loss_val
                checkpoint_best = {
                "model_state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                }
        logger.log("the best of loss in fold is {}".format(best_loss))
    checkpoint_last = {
        "model_state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    return checkpoint_last,checkpoint_best





