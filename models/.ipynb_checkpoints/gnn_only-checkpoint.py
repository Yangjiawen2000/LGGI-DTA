import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import sys
sys.path.append('..')
from models.gcn import GCNConv  # or maybe GINConv, but standard MP isn't exactly LGGILayer. Let's build a standard non-guided layer
                  
                  
class GNNOnlyLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNOnlyLayer, self).__init__()
        from torch_geometric.nn import GCNConv
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.bn(x)
        x = self.relu(x)
        return x


class GNNOnlyNet(torch.nn.Module):
    """
    消融变体模型 1：纯 GNN 模型 (GNN-Only)
    
    仅使用分子的结构信息进行亲和力预测，相当于切断了所有的 ESM 蛋白语义特征。
    用于验证"仅仅对分子图建模是否足够，引入蛋白信息的必要性"。
    """
    def __init__(self, n_output=1, num_features_xd=78, hidden_dim=128, output_dim=128, dropout=0.2):
        super(GNNOnlyNet, self).__init__()

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

        # ==================== 预测头 MLP (Prediction Head) ====================
        # 纯 GNN 结构预测，无蛋白特征参与
        self.fc1 = nn.Linear(output_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, self.n_output)

    def forward(self, data):
        # x shape: [num_nodes, 78] (RDKit 原子特征)
        x, edge_index, batch = data.x, data.edge_index, data.batch

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

        # 图池化 
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # 降维
        x = self.fc_g1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # 预测头 (仅依赖图特征)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        out = self.out(x)
        return out
