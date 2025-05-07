'''
train_data_path = 'db_strat/classification_aug_types_training.p'
save_model_path = 'interaction_classifier.pt'
'''
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class InteractionDataset(Dataset):
    def __init__(self,
                 couples_path: str,
                 pooling_mode: str = 'bom',
                 k: int = 100,
                 stride: int = 20):
        """
        couples_path: path to pickle of raw embeddings
        pooling_mode: only 'bom' supported here
        k: window size
        stride: window stride
        """
        with open(couples_path, 'rb') as f:
            raw = pickle.load(f)

        self.pooling_mode = pooling_mode
        self.k = k
        self.stride = stride
        self.items = []

        for item in raw:
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

            self.items.append((embA_p, embB_p, torch.tensor(label)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def _bom_pool(self, emb: torch.Tensor) -> torch.Tensor:
        """
        emb: Tensor of shape [1, seq_len, D]
        returns: Tensor [n_windows, D]
        """
        with torch.no_grad():
            x = emb.squeeze(0)           # [seq_len, D]
        if x.size(0) > 2:
            x = x[1:-1]              # strip special tokens
        # short sequence: global avg
        if x.size(0) < self.k:
            return x.mean(dim=0, keepdim=True)
        # unfold into windows: [n_win, k, D]
        windows = x.unfold(0, self.k, self.stride)
        return windows.mean(dim=1)


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
