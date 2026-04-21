import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import sys
sys.path.append('..')
from lggi_layer import LGGILayer


class LGGIDTANet(torch.nn.Module):
    """
    LGGI-DTA: LLM-Guided Graph Interaction for Drug-Target Affinity Prediction
    
    与 GraphDTA 的本质区别在于：
    1. 蛋白特征不再是旁路独立处理后简单拼接，而是在图卷积底层的每一步 Message Passing
       中就以门控信号的形式深度参与了药物分子图的特征聚合。
    2. 因此最终的预测头 (Prediction Head) 仅需接收图池化后的药物表征即可，蛋白质信息
       已经在底层被充分"注入"到了药物表征中。
    """
    def __init__(self, n_output=1, num_features_xd=78, protein_dim=320,
                 hidden_dim=128, output_dim=128, dropout=0.2):
        """
        参数:
            n_output (int): 输出维度，回归任务为 1
            num_features_xd (int): 药物分子图的原始节点特征维度 (RDKit 提取的原子特征)
            protein_dim (int): ESM 大语言模型提取的蛋白质特征维度
            hidden_dim (int): LGGI 层的隐藏层维度
            output_dim (int): 图池化后全连接层的输出维度
            dropout (float): Dropout 概率
        """
        super(LGGIDTANet, self).__init__()

        self.n_output = n_output
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # ==================== 药物分子图处理分支 (LGGI Layers) ====================
        # 第 1 层 LGGI：将原始原子特征 (78维) 映射到隐藏空间 (hidden_dim)，
        #              同时注入蛋白质门控信号
        self.lggi1 = LGGILayer(in_channels=num_features_xd,
                               out_channels=hidden_dim,
                               protein_dim=protein_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # 第 2 层 LGGI：在隐藏空间内继续深度交互
        self.lggi2 = LGGILayer(in_channels=hidden_dim,
                               out_channels=hidden_dim,
                               protein_dim=protein_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 第 3 层 LGGI：最终一轮的蛋白引导信息传递
        self.lggi3 = LGGILayer(in_channels=hidden_dim,
                               out_channels=hidden_dim,
                               protein_dim=protein_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # ==================== 图级读出层 (Graph Readout) ====================
        # 将全局最大池化和全局平均池化拼接，捕获更丰富的图级表征
        # 池化后维度: hidden_dim * 2
        self.fc_g1 = nn.Linear(hidden_dim * 2, output_dim)

        # ==================== 预测头 MLP (Prediction Head) ====================
        # 注意：不再需要拼接蛋白特征！蛋白信息已经在 LGGI 层底层渗透进药物表征。
        self.fc1 = nn.Linear(output_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, self.n_output)

    def forward(self, data):
        """
        前向传播：
        从 PyG 的 Batch 对象中解包所有需要的属性，驱动三层 LGGI 机制融合，
        最终通过图级池化和 MLP 输出亲和力预测值。
        
        参数:
            data: PyG Batch 对象，包含以下属性：
                data.x            - 节点特征        shape: [num_nodes, 78]
                data.edge_index   - 边索引          shape: [2, num_edges]
                data.batch        - 节点所属图索引   shape: [num_nodes]
                data.protein_feat - ESM 蛋白质特征   shape: [batch_size, 320]
        
        返回:
            out: 亲和力预测值  shape: [batch_size, 1]
        """
        # ========== 第一步：从 Batch 对象中解包所有输入 ==========
        # x shape: [num_nodes, 78] (RDKit 原子特征)
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # protein_tokens shape: [batch_size, K=32, 320] (ESM 残基级蛋白 token)
        protein_tokens = data.protein_feat

        # ========== 第二步：三层 LGGI 机制级融合 ==========
        # 每一层都将 protein_tokens 作为门控条件注入到图消息传递中

        # LGGI Layer 1: [num_nodes, 78] -> [num_nodes, hidden_dim]
        x = self.lggi1(x, edge_index, protein_tokens, batch)
        x = self.bn1(x)
        x = self.relu(x)

        # LGGI Layer 2: [num_nodes, hidden_dim] -> [num_nodes, hidden_dim]
        x = self.lggi2(x, edge_index, protein_tokens, batch)
        x = self.bn2(x)
        x = self.relu(x)
        
        # LGGI Layer 3: [num_nodes, hidden_dim] -> [num_nodes, hidden_dim]
        x = self.lggi3(x, edge_index, protein_tokens, batch)
        x = self.bn3(x)
        x = self.relu(x)

        # ========== 第三步：图级读出 (Graph-level Readout) ==========
        # 使用 Global Max Pooling + Global Mean Pooling 双通道读出
        # gmp(x, batch) shape: [batch_size, hidden_dim]
        # gap(x, batch) shape: [batch_size, hidden_dim]
        # x shape（拼接后）: [batch_size, hidden_dim * 2]
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # 全连接降维
        # x shape: [batch_size, output_dim]
        x = self.fc_g1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # ========== 第四步：预测头 MLP ==========
        # 【核心变革】此处直接使用这些已经被蛋白特征深度调制过的药物图表征
        # 完全移除了原版 GraphDTA 中 torch.cat((drug_x, target_x)) 的暴力拼接操作
        
        # x shape: [batch_size, 512]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # x shape: [batch_size, 256]
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # out shape: [batch_size, 1]
        out = self.out(x)
        return out
