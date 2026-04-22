import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing


class LGGILayer(MessagePassing):
    """
    LLM-Guided Graph Interaction Layer (LGGI) — v2 残基级 Cross-Attention
    
    论文核心突破：利用预训练蛋白语言模型 (Protein LLM) 提取的残基级特征 (Residue Tokens)，
    通过 Atom-Residue Cross-Attention 机制为每个原子节点生成个性化的蛋白引导信号 (Guidance
    Signal)，然后以门控方式调制底层 Message Passing 中邻居节点传递的信息。
    
    与 v1 的关键区别：
    - v1: 全局蛋白特征 [batch_size, 320] → broadcast 到所有原子（所有原子看到相同的蛋白信号）
    - v2: 残基级 token [batch_size, K, 320] → 每个原子通过 attention 选择性地关注不同残基位点
          （不同原子看到的蛋白引导信号不同，实现了真正的 atom-residue 交互）
    """
    def __init__(self, in_channels, out_channels, protein_dim=320,
                 num_heads=4, return_attention=False, **kwargs):
        """
        参数:
            in_channels (int): 输入节点特征维度
            out_channels (int): 输出节点特征维度
            protein_dim (int): ESM 蛋白 token 的特征维度
            num_heads (int): Cross-Attention 的头数
            return_attention (bool): 是否返回 gate_score（用于可解释性可视化）
        """
        super(LGGILayer, self).__init__(aggr='add', **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.return_attention = return_attention
        self._last_gate_score = None  # P2: 缓存门控分数
        
        # ====== 节点特征投影 ======
        self.lin_x = nn.Linear(in_channels, out_channels)
        
        # ====== Atom-Residue Cross-Attention ======
        # Query: 原子节点特征 → [num_nodes, out_channels]
        # Key/Value: 残基级蛋白 token → [num_nodes, K, out_channels]
        self.W_q = nn.Linear(out_channels, out_channels)  # atom → query
        self.W_k = nn.Linear(protein_dim, out_channels)   # residue → key
        self.W_v = nn.Linear(protein_dim, out_channels)   # residue → value
        
        # attention 输出后的投影层
        self.attn_proj = nn.Linear(out_channels, out_channels)
        # Cross-Attention 残差后的层归一化
        self.attn_norm = nn.LayerNorm(out_channels)
        
        # ====== 门控信号生成网络 ======
        # 输入: 拼接的 [邻居节点特征, cross-attention 生成的蛋白引导信号]
        self.gate_nn = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()  # 压缩到 [0,1] 产生通道级门控
        )
        
        # ====== 残差更新层 ======
        self.update_net = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.SiLU()
        )
        
        # attention 缩放因子
        self.scale = (out_channels // num_heads) ** -0.5

    def _cross_attention(self, x_proj, protein_tokens, batch):
        """
        Atom-Residue Multi-Head Cross-Attention：
        每个原子节点并行通过多个头关注蛋白质的不同残基区域。
        
        参数:
            x_proj (Tensor): 投影后的原子特征。
                             Shape: [num_nodes, out_channels]
            protein_tokens (Tensor): 残基级蛋白 token。
                                     Shape: [batch_size, K, protein_dim]
            batch (LongTensor): 节点所属图索引。
                                Shape: [num_nodes]
        
        返回:
            guidance (Tensor): 每个原子独有的蛋白引导信号。
                               Shape: [num_nodes, out_channels]
        """
        num_nodes = x_proj.size(0)
        K = protein_tokens.size(1)  # 蛋白 token 数量 (=32)
        H = self.num_heads
        D = self.out_channels
        head_dim = D // H
        
        # 第一步：将图级蛋白 token 广播到对应的原子节点
        # protein_per_node shape: [num_nodes, K, protein_dim]
        protein_per_node = protein_tokens[batch]
        
        # 第二步：计算 Q / K / V 并切分为多头
        # Q: [num_nodes, D] -> [num_nodes, H, 1, head_dim]
        Q = self.W_q(x_proj).view(num_nodes, H, 1, head_dim)
        
        # K_mat/V_mat: [num_nodes, K, D] -> [num_nodes, H, K, head_dim]
        K_mat = self.W_k(protein_per_node).view(num_nodes, K, H, head_dim).transpose(1, 2)
        V_mat = self.W_v(protein_per_node).view(num_nodes, K, H, head_dim).transpose(1, 2)
        
        # 第三步：计算多头注意力分数
        # attn_scores = (Q @ K^T) / sqrt(dk) -> [num_nodes, H, 1, K]
        attn_scores = torch.matmul(Q, K_mat.transpose(-1, -2)) * self.scale
        
        # attn_weights shape: [num_nodes, H, 1, K]
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 第四步：加权聚合 Value 并拼接多头
        # context: [num_nodes, H, 1, head_dim]
        context = torch.matmul(attn_weights, V_mat)
        
        # Reshape & Concat: [num_nodes, D]
        context = context.squeeze(2).reshape(num_nodes, D)
        
        # 第五步：最终投影
        guidance = self.attn_proj(context)
        
        # P2: 缓存 attention_weights (包含所有头) 用于可解释性分析
        if self.return_attention:
            # 缓存为 [num_nodes, H, K]
            self._last_attn_weights = attn_weights.squeeze(2).detach()
        
        return guidance

    def forward(self, x, edge_index, protein_tokens, batch):
        """
        前向传播：Cross-Attention 生成引导信号 → 注入 Message Passing 门控。
        
        参数:
            x (Tensor): 分子图节点特征。
                        Shape: [num_nodes, in_channels]
            edge_index (LongTensor): 边索引。
                                     Shape: [2, num_edges]
            protein_tokens (Tensor): 残基级蛋白 token。
                                     Shape: [batch_size, K, protein_dim]
            batch (LongTensor): 节点所属图索引。
                                Shape: [num_nodes]
                                
        返回:
            Tensor: 更新后的节点特征。
                    Shape: [num_nodes, out_channels]
        """
        # 1) 投射节点特征到统一隐空间，确保与 attention 输出维度对齐
        # x_proj: [num_nodes, out_channels]
        x_proj = self.lin_x(x)

        # 2) Atom-Residue Cross-Attention 生成蛋白引导特征
        # attn_out(guidance): [num_nodes, out_channels]
        attn_out = self._cross_attention(x_proj, protein_tokens, batch)

        # 3) 严格执行 Residual + LayerNorm
        #    x_attn = LayerNorm(x_proj + attn_out)
        #    这里使用 x_proj 做残差支路，避免第一层 in_channels != out_channels 时形状不匹配
        x_attn = self.attn_norm(x_proj + attn_out)

        # 4) 在 PyG 中进行消息传递：
        #    - x=x_attn 会在 message() 中映射为 x_j，shape: [num_edges, out_channels]
        #    - g=attn_out 会在 message() 中映射为 g_i，shape: [num_edges, out_channels]
        out = self.propagate(edge_index, x=x_attn, g=attn_out)

        # 5) 消息聚合后的更新 + 残差
        # final: [num_nodes, out_channels]
        return self.update_net(out) + x_attn

    def message(self, x_j, g_i):
        """
        底层信息生成与门控调制 (Mechanism-level Fusion)。
        
        每个中心节点 i 拥有从 Cross-Attention 获得的个性化蛋白引导信号 g_i，
        该信号与邻居节点 j 的特征 x_j 共同决定信息传递的门控强度。
        
        参数:
            x_j (Tensor): 邻居节点 j 的特征（已投影）。
                          Shape: [num_edges, out_channels]
            g_i (Tensor): 中心节点 i 的蛋白引导信号（来自 Cross-Attention）。
                          Shape: [num_edges, out_channels]
                          
        返回:
            Tensor: 门控调制后的边消息。
                    Shape: [num_edges, out_channels]
        """
        # 1. 拼接邻居原子特征与该原子位置的蛋白引导信号
        # cross_feat shape: [num_edges, out_channels * 2]
        cross_feat = torch.cat([x_j, g_i], dim=-1)
        
        # 2. 生成通道级门控分数（Gating Signal）
        # gate_score shape: [num_edges, out_channels]
        gate_score = self.gate_nn(cross_feat)
        
        # P2: 缓存 gate_score 用于可解释性分析
        if self.return_attention:
            self._last_gate_score = gate_score.detach()
        
        # 3. Hadamard 门控：特征级别的信息放大或抑制
        # modulated_msg shape: [num_edges, out_channels]
        modulated_msg = x_j * gate_score
        
        return modulated_msg
