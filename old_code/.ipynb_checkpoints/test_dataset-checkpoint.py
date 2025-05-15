from torch.utils.data import Dataset
import h5py
import pickle
import torch
import os
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import numpy as np
from torch.nn.functional import scaled_dot_product_attention
import torch.nn.functional as F
from tqdm import tqdm

class InteractionTestDataset(Dataset):
    def __init__(self,
                 couples_path: str,
                 pooling_mode: str = 'bom',
                 k: int = 10000,
                 stride: int = 10000):
        self.pooling_mode = pooling_mode
        self.k = k
        self.stride = stride
        self.embA_list = []
        self.embB_list = []
        self.labels = []
        self.types = []

        with open(couples_path, 'rb') as f:
            raw = pickle.load(f)

        # TEST
        #counter = 0

        for sample in tqdm(raw, desc="BoM pooling (test)"):
            
            # TEST
            #counter += 1
            # TEST
            #if counter > 40000:
                # TEST
                #print('Testin 40 k')
                #break
                
            if 'true' in sample:
                embA = sample['true'][0][0].detach().cpu()
                embB = sample['true'][0][1].detach().cpu()
                label = 1.0
                itype = sample['true'][1]
            else:
                embA = sample['negative'][0][0].detach().cpu()
                embB = sample['negative'][0][1].detach().cpu()
                label = 0.0
                itype = sample['negative'][1]

            if pooling_mode == 'bom':
                embA_p = self._bom_pool(embA)
                embB_p = self._bom_pool(embB)
            else:
                raise ValueError(f"Unsupported pooling mode: {pooling_mode}")

            self.embA_list.append(embA_p)
            self.embB_list.append(embB_p)
            self.labels.append(label)
            self.types.append(itype)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embA_list[idx], self.embB_list[idx], self.labels[idx], self.types[idx]

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


class InteractionTestDataset_new(Dataset):
    def __init__(self,
                 couples_path: str,
                 pooling_mode: str = 'bom',
                 k: int = 10000,
                 stride: int = 10000,
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
                self.types = data['types']
                print(f"Loaded preprocessed dataset from {db_path}")
                return
            except Exception as e:
                print(f"Could not load dataset from {db_path}: {e}")
                sys.exit(1)
        else:
            with open(couples_path, 'rb') as f:
                raw = pickle.load(f)
            self.embA_list, self.embB_list, self.labels, self.types = [], [], [], []
            for i, item in enumerate(tqdm(raw, desc="BoM pooling")):
                #if i >= 10000:  # testing cutoff
                #    break
                '''
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
                '''

                if 'true' in sample:
                    embA = sample['true'][0][0].detach().cpu()
                    embB = sample['true'][0][1].detach().cpu()
                    label = 1.0
                    itype = sample['true'][1]
                else:
                    embA = sample['negative'][0][0].detach().cpu()
                    embB = sample['negative'][0][1].detach().cpu()
                    label = 0.0
                    itype = sample['negative'][1]
    
                if pooling_mode == 'bom':
                    embA_p = self._bom_pool(embA)
                    embB_p = self._bom_pool(embB)
                else:
                    raise ValueError(f"Unsupported pooling mode: {pooling_mode}")
    
                self.embA_list.append(embA_p)
                self.embB_list.append(embB_p)
                self.labels.append(label)
                self.types.append(itype)

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
