import os
import pickle
import h5py
import torch
import numpy as np

from torch.utils.data import Dataset


class InteractionDataset(Dataset):
    """
    In-memory pooling of raw pickle embeddings.
    """
    def __init__(self, couples_path: str, k: int = 100, stride: int = 20):
        self.k = k
        self.stride = stride
        with open(couples_path, 'rb') as f:
            raw = pickle.load(f)
        self.embA_list, self.embB_list, self.labels = [], [], []
        #for item in raw[:100000]:  # limit for quick experiments
        for item in raw:
            label = 1.0 if 'true' in item else 0.0
            embA = item.get('true', item.get('negative'))[0][0].detach().cpu()
            embB = item.get('true', item.get('negative'))[1][0].detach().cpu()
            self.embA_list.append(self._bom_pool(embA))
            self.embB_list.append(self._bom_pool(embB))
            self.labels.append(label)

    def _bom_pool(self, emb: torch.Tensor) -> torch.Tensor:
        x = emb.squeeze(0)
        if x.size(0) < self.k:
            return x.mean(dim=0, keepdim=True)
        windows = x.unfold(0, self.k, self.stride).transpose(1, 2)
        return windows.mean(dim=1)

    def __len__(self):
        return len(self.labels)


def build_h5_from_pickle(couples_path: str, h5_path: str, k: int = 100, stride: int = 20):
    """
    Pools raw embeddings then writes HDF5 datasets:
      - embA: (N, max_w, D)
      - embB: (N, max_w, D)
      - labels, lenA, lenB
    """
    ds = InteractionDataset(couples_path, k=k, stride=stride)
    N = len(ds)
    D = ds.embA_list[0].size(1)
    max_w = max(
        max(t.size(0) for t in ds.embA_list),
        max(t.size(0) for t in ds.embB_list)
    )

    arrA = np.zeros((N, max_w, D), dtype=np.float32)
    arrB = np.zeros((N, max_w, D), dtype=np.float32)
    labels = np.zeros((N,), dtype=np.int8)
    lenA = np.zeros((N,), dtype=np.int32)
    lenB = np.zeros((N,), dtype=np.int32)

    for i, (a, b, lbl) in enumerate(zip(ds.embA_list, ds.embB_list, ds.labels)):
        wA, wB = a.size(0), b.size(0)
        arrA[i, :wA] = a.numpy()
        arrB[i, :wB] = b.numpy()
        labels[i] = int(lbl)
        lenA[i] = wA
        lenB[i] = wB

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    with h5py.File(h5_path, 'w') as h5:
        h5.create_dataset('embA', data=arrA, compression='lzf')
        h5.create_dataset('embB', data=arrB, compression='lzf')
        h5.create_dataset('labels', data=labels)
        h5.create_dataset('lenA', data=lenA)
        h5.create_dataset('lenB', data=lenB)

