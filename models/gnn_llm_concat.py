import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import sys


class GNNLLMConcatNet(torch.nn.Module):
    """
    消融变体模型 2：GNN + LLM Late Fusion (简单拼接)
    
    代表了当前领域的传统做法：
    使用 GNN 提取分子特征，独立使用 LLM（ESM）提取蛋白特征，
    然后在预测头 (Prediction Head) 之前将两者粗暴拼接。
    
    用于验证我们提出的 Mechanism-level Fusion (LGGI) 相对 Feature-level Fusion 的优越性。
    """
    def __init__(self, n_output=1, num_features_xd=78, protein_dim=320, 
                 hidden_dim=128, output_dim=128, dropout=0.2):
        super(GNNLLMConcatNet, self).__init__()

        self.n_output = n_output
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # ==================== 药物分子图处理分支 (纯 GNN) ====================
        from torch_geometric.nn import GCNConv
        self.gnn1 = GCNConv(num_features_xd, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.gnn2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.gnn3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # ==================== 图级读出层 (Graph Readout) ====================
        self.fc_g1 = nn.Linear(hidden_dim * 2, output_dim)

        # ==================== 蛋白质处理分支 ====================
        # (可选) 对全局池化后的蛋白特征做一层映射
        self.fc_p1 = nn.Linear(protein_dim, output_dim)

        # ==================== 预测头 MLP (Late Fusion) ====================
        # 输入维度 = 药物图特征(output_dim) + 蛋白特征(output_dim)
        self.fc1 = nn.Linear(output_dim * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, self.n_output)

    def forward(self, data):
        # x shape: [num_nodes, 78] (RDKit 原子特征)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # protein_tokens shape: [batch_size, K=32, 320]
        protein_tokens = data.protein_feat

        # --- 1. Drug Pathway ---
        # 3 层 GNN，无蛋白引导
        x = self.gnn1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.gnn2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.gnn3(x, edge_index)
        x = self.bn3(x)
        x = self.relu(x)

        # 图池化 -> [batch_size, hidden_dim * 2]
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # 降维 -> [batch_size, output_dim]
        drug_feat = self.fc_g1(x)
        drug_feat = self.relu(drug_feat)
        drug_feat = self.dropout(drug_feat)

        # --- 2. Protein Pathway ---
        # 将残基级 token 退化为全局特征 (模拟传统做法)
        # [batch_size, 32, 320] -> [batch_size, 320]
        prot_feat_global = protein_tokens.mean(dim=1)
        
        # 映射 -> [batch_size, output_dim]
        prot_feat = self.fc_p1(prot_feat_global)
        prot_feat = self.relu(prot_feat)
        prot_feat = self.dropout(prot_feat)

        # --- 3. Late Fusion ---
        # 暴力拼接 concat -> [batch_size, output_dim * 2]
        xc = torch.cat((drug_feat, prot_feat), dim=1)

        # 预测头
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)

        out = self.out(xc)
        return out
