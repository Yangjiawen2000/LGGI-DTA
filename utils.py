import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch

# --- ESM Cache and Global Variables ---
ESM_CACHE = {}
_esm_model = None
_esm_tokenizer = None
_esm_device = None

# 残基级蛋白 token 数量（通过 adaptive pooling 压缩至固定长度）
NUM_PROTEIN_TOKENS = 32

def get_esm_embedding(sequence):
    """
    使用预训练 ESM 提取残基级蛋白特征 (带缓存优化)
    
    通过 Adaptive Average Pooling 将变长的残基序列压缩为固定 K 个 token，
    既保留了残基级的空间分辨率，又确保了 PyG Batch 拼接的维度一致性。
    
    输入: sequence (str) — 原始氨基酸序列
    输出: protein_tokens (torch.FloatTensor) shape: [1, K, 320]
          其中 K = NUM_PROTEIN_TOKENS = 32
    """
    global _esm_model, _esm_tokenizer, _esm_device, ESM_CACHE
    if sequence in ESM_CACHE:
        return ESM_CACHE[sequence]
        
    if _esm_model is None:
        from transformers import EsmTokenizer, EsmModel
        # 检测设备优先分配给Apple Silicon(mps)，再降级为cuda或cpu
        _esm_device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Loading ESM model facebook/esm2_t6_8M_UR50D to device: {_esm_device} ...")
        _esm_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        _esm_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(_esm_device)
        _esm_model.eval() # 开启推理模式
        
    # 截断支持到最大1024长度
    inputs = _esm_tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024).to(_esm_device)
    
    with torch.no_grad():
        outputs = _esm_model(**inputs)
        
    # ESM 输出最后一层的隐藏状态
    # last_hidden_state shape: [1, seq_len, hidden_dim=320]
    last_hidden = outputs.last_hidden_state  # [1, seq_len, 320]
    
    # 去除 BOS/EOS 特殊 token，只保留真实残基的 embedding
    # last_hidden[:, 1:-1, :] shape: [1, seq_len-2, 320]
    residue_feats = last_hidden[:, 1:-1, :]
    
    # 解决 MPS (Apple Silicon) 会报错: Adaptive pool MPS: input sizes must be divisible by output sizes
    # 将其转移到 CPU 进行池化
    residue_feats_cpu = residue_feats.cpu()
    
    # 使用 Adaptive Average Pooling 将变长残基序列压缩到固定 K 个 token
    # 转置: [1, seq_len-2, 320] → [1, 320, seq_len-2] (AdaptiveAvgPool1d 作用于最后一维)
    residue_feats_t = residue_feats_cpu.transpose(1, 2)  # [1, 320, seq_len-2]
    pooled = torch.nn.functional.adaptive_avg_pool1d(residue_feats_t, NUM_PROTEIN_TOKENS)  # [1, 320, K]
    # 转置回来: [1, 320, K] → [1, K, 320]
    protein_tokens = pooled.transpose(1, 2)  # 已经是 CPU 张量，shape: [1, K, 320]
    
    # 存入全局唯一序列字典缓存
    ESM_CACHE[sequence] = protein_tokens
    return protein_tokens

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt=None, xt_seq=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, xt_seq, y,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # XT_SEQ - list of raw protein sequences
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, xt_seq, y, smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES and Protein to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            target_seq = xt_seq[i]
            labels = y[i]
            
            # 获取 ESM 特征 shape: [320]
            protein_feat = get_esm_embedding(target_seq)
            
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]),
                                protein_feat=protein_feat) # [NEW] 植入提取出的大语言模型高维特征
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci