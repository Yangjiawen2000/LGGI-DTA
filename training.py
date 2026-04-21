import os
# Removed MPS fallback for Linux environment

import numpy as np
import pandas as pd
import sys
from random import shuffle
import torch
import torch.nn as nn
from models.lggi_dta import LGGIDTANet
from utils import *

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        try:
            output = model(data)
        except RuntimeError as e:
            # 张量维度不匹配时，打印详细的 batch 诊断信息并优雅退出
            print('\n' + '='*60)
            print('[ERROR] 前向传播异常，以下是导致报错的 batch 数据诊断信息：')
            print('='*60)
            print(f'  batch_idx:            {batch_idx}')
            print(f'  data.x shape:         {data.x.shape}')
            print(f'  data.edge_index shape:{data.edge_index.shape}')
            print(f'  data.batch shape:     {data.batch.shape}')
            print(f'  data.protein_feat shape: {data.protein_feat.shape}')
            print(f'  data.y shape:         {data.y.shape}')
            print(f'  data.x device:        {data.x.device}')
            print(f'  data.protein_feat device: {data.protein_feat.device}')
            print(f'  unique batch indices: {data.batch.unique().shape[0]}')
            print(f'  Error message: {e}')
            print('='*60)
            raise e
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            try:
                output = model(data)
            except RuntimeError as e:
                print('\n' + '='*60)
                print('[ERROR] 预测阶段前向传播异常，batch 诊断信息：')
                print('='*60)
                print(f'  batch_idx:            {batch_idx}')
                print(f'  data.x shape:         {data.x.shape}')
                print(f'  data.edge_index shape:{data.edge_index.shape}')
                print(f'  data.batch shape:     {data.batch.shape}')
                print(f'  data.protein_feat shape: {data.protein_feat.shape}')
                print(f'  data.y shape:         {data.y.shape}')
                print(f'  Error message: {e}')
                print('='*60)
                raise e
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


# LGGI-DTA: 固定使用 LGGIDTANet 模型
modeling = LGGIDTANet
model_st = modeling.__name__

# 数据集选择：0=davis, 1=kiba
datasets = [['davis','kiba'][int(sys.argv[1])]]

# 设备自适应：CUDA (GPU) > CPU
if torch.cuda.is_available():
    cuda_name = "cuda:0"
    if len(sys.argv) > 2:
        cuda_name = "cuda:" + str(int(sys.argv[2]))
    device = torch.device(cuda_name)
else:
    device = torch.device("cpu")
print("Device:", device)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset )
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    train_data = TestbedDataset(root='data', dataset=dataset+'_train')
    test_data = TestbedDataset(root='data', dataset=dataset+'_test')
    
    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

    # training the model
    model = modeling().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_mse = 1000
    best_ci = 0
    best_epoch = -1
    model_file_name = 'model_' + model_st + '_' + dataset +  '.model'
    result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch+1)
        G,P = predicting(model, device, test_loader)
        ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
        if ret[1]<best_mse:
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name,'w') as f:
                f.write(','.join(map(str,ret)))
            best_epoch = epoch+1
            best_mse = ret[1]
            best_ci = ret[-1]
            print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)
        else:
            print(ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)
