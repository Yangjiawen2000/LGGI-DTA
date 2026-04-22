import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx

# Functions from create_data.py
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

# --- ESM Cache and Global Variables ---
ESM_CACHE = {}
_esm_model = None
_esm_tokenizer = None
_esm_device = None

# 统一 ESM 序列长度配置：
# - tokenizer 最大长度至少 1000（含特殊 token），这里设为 1002 以保留 1000 个残基位点
# - 输出静态特征默认固定为 [1000, 320]，便于离线缓存和稳定复用
ESM_RESIDUE_MAX_LEN = 1000
ESM_TOKENIZER_MAX_LEN = ESM_RESIDUE_MAX_LEN + 2

def get_esm_embedding(sequence):
    """
    使用预训练 ESM 提取残基级蛋白特征 (带缓存优化)。

    设计目标：
    1) tokenizer 的 max_length >= 1000；
    2) 使用 no_grad 禁用反向图，降低显存压力；
    3) 在 CPU 上完成特征截断/补零与缓存，降低 OOM 风险；
    4) 返回静态特征矩阵，shape 为 [1000, 320]（或可截断为实际长度）。

    输入: sequence (str) — 原始氨基酸序列
    输出: protein_tokens (torch.FloatTensor) shape: [ESM_RESIDUE_MAX_LEN, 320]
    """
    global _esm_model, _esm_tokenizer, _esm_device, ESM_CACHE
    if sequence in ESM_CACHE:
        return ESM_CACHE[sequence]

    if _esm_model is None:
        from transformers import EsmTokenizer, EsmModel
        # 设备选择：优先使用 GPU（CUDA），不可用时回退到 CPU
        _esm_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Loading ESM model facebook/esm2_t6_8M_UR50D to device: {_esm_device} ...")
        _esm_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        _esm_model = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D").to(_esm_device)
        _esm_model.eval()  # 开启推理模式

    # tokenizer 最大长度至少为 1000（这里使用 1002: 1000 residue + BOS/EOS）
    inputs = _esm_tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=ESM_TOKENIZER_MAX_LEN
    )

    # 推理阶段禁用梯度，降低显存占用
    with torch.no_grad():
        try:
            inputs = {k: v.to(_esm_device) for k, v in inputs.items()}
            outputs = _esm_model(**inputs)
        except RuntimeError as e:
            # 超长序列导致 CUDA OOM 时，自动回退到 CPU 推理
            if 'out of memory' in str(e).lower() and _esm_device.type == 'cuda':
                print("CUDA OOM while extracting ESM features. Falling back to CPU inference.")
                _esm_device = torch.device('cpu')
                _esm_model = _esm_model.to(_esm_device)
                inputs = {k: v.to(_esm_device) for k, v in inputs.items()}
                outputs = _esm_model(**inputs)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise

    # ESM 输出最后一层隐藏状态: [1, seq_len, 320]
    # 去除 BOS/EOS，仅保留真实残基: [res_len, 320]
    residue_feats = outputs.last_hidden_state[:, 1:-1, :].squeeze(0).detach().cpu().float()

    # 固定长度静态缓存: [1000, 320]（短序列补零，超长序列截断）
    protein_tokens = torch.zeros((ESM_RESIDUE_MAX_LEN, residue_feats.size(-1)), dtype=torch.float32)
    valid_len = min(residue_feats.size(0), ESM_RESIDUE_MAX_LEN)
    if valid_len > 0:
        protein_tokens[:valid_len] = residue_feats[:valid_len]

    # 主动释放中间变量，进一步降低峰值内存
    del outputs, residue_feats, inputs

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
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

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
    # Load data from CSV and process
    def process(self):
        # Load data from CSV
        df = pd.read_csv(os.path.join(self.root, self.dataset + '.csv'))
        xd = df['compound_iso_smiles'].tolist()
        xt_seq = df['target_sequence'].tolist()
        y = df['affinity'].tolist()
        
        # Encode targets (simple index for now)
        all_prots = list(set(xt_seq))
        xt = [all_prots.index(seq) for seq in xt_seq]
        
        # Build smile_graph
        smile_graph = {}
        for smiles in set(xd):
            smile_graph[smiles] = smile_to_graph(smiles)
        
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES and Protein to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            target_seq = xt_seq[i]
            labels = y[i]
            
            # 获取静态 ESM 特征并在 CPU 上保存
            # get_esm_embedding 返回 [1000, 320]；这里扩一维为 [1, 1000, 320]，
            # 便于 PyG Batch 后形成 [batch_size, 1000, 320]。
            protein_feat = get_esm_embedding(target_seq).cpu().contiguous().unsqueeze(0)
            
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]),
                                protein_feat=protein_feat) # [NEW] 静态蛋白特征（CPU缓存）
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