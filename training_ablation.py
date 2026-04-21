import os
import argparse
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils import *

# 设备自适应逻辑
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# 动态模型映射
MODEL_MAP = {
    'gnn_only': ('models.gnn_only', 'GNNOnlyNet'),
    'gnn_llm_concat': ('models.gnn_llm_concat', 'GNNLLMConcatNet'),
    'lggi_dta': ('models.lggi_dta', 'LGGIDTANet')
}

def get_model(model_key):
    module_path, class_name = MODEL_MAP[model_key]
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)

# 复用 training.py 的训练/预测逻辑
def train(model, device, train_loader, optimizer, epoch, loss_fn, log_interval):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train epoch: {epoch} [{batch_idx * train_loader.batch_size}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

def main():
    parser = argparse.ArgumentParser(description='GraphDTA Ablation Study')
    # 支持位置参数作为数据集索引 (0: davis, 1: kiba)
    parser.add_argument('dataset_idx', type=int, nargs='?', help='Dataset index (0: davis, 1: kiba)')
    parser.add_argument('--model', type=str, required=True, choices=MODEL_MAP.keys(), help='Model architecture to use')
    parser.add_argument('--dataset', type=str, choices=['davis', 'kiba'], help='Dataset name (overrides positional idx)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    args = parser.parse_args()

    # 确定数据集名称
    dataset_name = 'davis' # 默认
    if args.dataset:
        dataset_name = args.dataset
    elif args.dataset_idx is not None:
        dataset_name = ['davis', 'kiba'][args.dataset_idx]

    # 获取模型类
    modeling = get_model(args.model)
    model_st = modeling.__name__
    
    print(f'\n[Ablation] Running {model_st} on {dataset_name}')
    print(f'Device: {device} | LR: {args.lr} | Epochs: {args.epochs}')

    # 数据准备
    processed_data_file_train = f'data/processed/{dataset_name}_train.pt'
    processed_data_file_test = f'data/processed/{dataset_name}_test.pt'
    
    if not os.path.isfile(processed_data_file_train) or not os.path.isfile(processed_data_file_test):
        print('Error: Processed data not found. Please run create_data.py first.')
        return

    train_data = TestbedDataset(root='data', dataset=dataset_name+'_train')
    test_data = TestbedDataset(root='data', dataset=dataset_name+'_test')
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # 模型初始化
    model = modeling().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    best_mse = 1000
    model_file_name = f'model_{model_st}_{args.dataset}.model'
    result_file_name = f'result_{model_st}_{args.dataset}.csv'

    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, epoch + 1, loss_fn, 20)
        G, P = predicting(model, device, test_loader)
        ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
        
        if ret[1] < best_mse:
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
            best_mse = ret[1]
            print(f'--> Epoch {epoch+1} improved. Best MSE: {best_mse:.6f}')
        else:
            print(f'    Epoch {epoch+1} result: MSE={ret[1]:.6f} (Best: {best_mse:.6f})')

    # 打印对比表
    print_comparison_table(dataset_name)

def print_comparison_table(dataset):
    print('\n' + '='*80)
    print(f' PERFORMANCE COMPARISON ON {dataset.upper()} '.center(80, '='))
    print(f'{"Model":<20} | {"RMSE":<10} | {"MSE":<10} | {"Pearson":<10} | {"Spearman":<10} | {"CI":<10}')
    print('-'*80)
    
    for key, (mod_path, cls_name) in MODEL_MAP.items():
        res_file = f'result_{cls_name}_{dataset}.csv'
        if os.path.exists(res_file):
            with open(res_file, 'r') as f:
                metrics = f.read().strip().split(',')
                # 假设 metrics 顺序为: rmse, mse, pearson, spearman, ci
                m = [f'{float(x):.4f}' for x in metrics]
                print(f'{cls_name:<20} | {m[0]:<10} | {m[1]:<10} | {m[2]:<10} | {m[3]:<10} | {m[4]:<10}')
        else:
            print(f'{cls_name:<20} | {"N/A":<10} | {"N/A":<10} | {"N/A":<10} | {"N/A":<10} | {"N/A":<10}')
    print('='*80 + '\n')

if __name__ == "__main__":
    main()
