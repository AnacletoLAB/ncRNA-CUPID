'''
train_data_path = 'db_strat/classification_aug_types_training.pt'
save_model_path = 'interaction_classifier.pt'
'''
import pickle
import torch
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
class InteractionDataset(Dataset):
    def __init__(self,
                 couples_path: str,
                 pooling_mode: str = 'bom',
                 k: int = 100,
                 stride: int = 20,
                 db_path: str = None,
                 existing_db_path: bool = False):
        self.pooling_mode = pooling_mode
        self.k = k
        self.stride = stride
        if existing_db_path and db_path and os.path.exists(db_path):
            try:
                data = torch.load(db_path, map_location='cpu')
                self.embA_list = data['embA']
                self.embB_list = data['embB']
                self.labels = data['labels']
                print(f"Loaded preprocessed dataset from {db_path}")
                return
            except Exception as e:
                print(f"Could not load dataset from {db_path}: {e}")
                sys.exit(1)
        else:
            with open(couples_path, 'rb') as f:
                raw = pickle.load(f)
            self.embA_list, self.embB_list, self.labels = [], [], []
            for i, item in enumerate(tqdm(raw, desc="BoM pooling")):
                #if i >= 10000:  # testing cutoff
                #    break
    
                if 'true' in item:
                    label = 1.0
                    embA = item['true'][0][0].detach().cpu()
                    embB = item['true'][1][0].detach().cpu()
                else:
                    label = 0.0
                    embA = item['negative'][0][0].detach().cpu()
                    embB = item['negative'][1][0].detach().cpu()
    
                if pooling_mode == 'bom':
                    embA_p = self._bom_pool(embA)
                    embB_p = self._bom_pool(embB)
                else:
                    raise ValueError(f"Unsupported pooling mode: {pooling_mode}")
    
                self.embA_list.append(embA_p)
                self.embB_list.append(embB_p)
                self.labels.append(label)

        if db_path:
            torch.save({
                'embA': self.embA_list,
                'embB': self.embB_list,
                'labels': torch.tensor(self.labels, dtype=torch.float)
            }, db_path)
            print(f"Preprocessed dataset saved to {db_path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embA_list[idx], self.embB_list[idx], self.labels[idx]

    def _bom_pool(self, emb: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = emb.squeeze(0).detach().cpu()
            if x.size(0) < self.k:
                pooled = x.mean(dim=0, keepdim=True)
            else:
                windows = x.unfold(0, self.k, self.stride)
                windows = windows.transpose(1, 2)
                pooled = windows.mean(dim=1)

            assert pooled.size(1) == 1024, f"Pooled shape wrong: {pooled.shape}"
            return pooled

class CrossAttention(nn.Module):
    def __init__(self, dim: int, inner_dim: int = 256, outer_dim: int = 1024):
        super().__init__()
        self.qnet = nn.Linear(dim, inner_dim)
        self.knet = nn.Linear(dim, inner_dim)
        self.vnet = nn.Linear(dim, outer_dim)
        self.inner_dim = inner_dim

    def forward(self, a: list[torch.Tensor], b: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        # concatenate all windows
        cat_a = torch.cat(a, dim=0)  # [N_a, dim]
        cat_b = torch.cat(b, dim=0)
        qa = self.qnet(cat_a)        # [N_a, inner]
        ka = self.knet(cat_a)
        va = self.vnet(cat_a)        # [N_a, outer]
        qb = self.qnet(cat_b)
        kb = self.knet(cat_b)
        vb = self.vnet(cat_b)

        idx_a = idx_b = 0
        emb_a = []
        emb_b = []
        for a_i, b_i in zip(a, b):
            len_a, len_b = a_i.size(0), b_i.size(0)
            qa_i = qa[idx_a: idx_a + len_a]
            ka_i = ka[idx_a: idx_a + len_a]
            va_i = va[idx_a: idx_a + len_a]
            qb_i = qb[idx_b: idx_b + len_b]
            kb_i = kb[idx_b: idx_b + len_b]
            vb_i = vb[idx_b: idx_b + len_b]

            # A -> B
            attn_a = F.softmax((qa_i / (self.inner_dim ** 0.5)) @ kb_i.transpose(0, 1), dim=-1)
            cross_a = attn_a @ vb_i
            # B -> A
            attn_b = F.softmax((qb_i / (self.inner_dim ** 0.5)) @ ka_i.transpose(0, 1), dim=-1)
            cross_b = attn_b @ va_i

            emb_a.append(cross_a.mean(dim=0, keepdim=True))  # [1, outer]
            emb_b.append(cross_b.mean(dim=0, keepdim=True))

            idx_a += len_a
            idx_b += len_b

        return emb_a, emb_b


class RNACrossAttentionClassifier(nn.Module):
    def __init__(self,
                 embedding_dim: int = 1024,
                 inner_dim: int = 256,
                 outer_dim: int = 1024,
                 hidden_dim: int = 512):
        super().__init__()
        self.cross_attn = CrossAttention(embedding_dim, inner_dim, outer_dim)
        self.fc = nn.Sequential(
            nn.Linear(2 * outer_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embA: torch.Tensor, embB: torch.Tensor) -> torch.Tensor:
        # embA: [nA, D], embB: [nB, D]
        emb_a_list, emb_b_list = self.cross_attn([embA], [embB])
        vA = emb_a_list[0].squeeze(0)  # [outer]
        vB = emb_b_list[0].squeeze(0)
        h = torch.cat([vA, vB], dim=-1)
        return self.fc(h).squeeze(-1)  # scalar logits
