
## dataset_hug.py
import os
import pickle
import torch
import numpy as np
from datasets import Dataset, load_from_disk, load_dataset, Features, Value, Array2D
from typing import Union


def _bom_pool(emb: torch.Tensor, k: int, stride: int) -> np.ndarray:
    x = emb.squeeze(0)
    if x.size(0) < k:
        return x.mean(dim=0, keepdim=True).cpu().numpy()
    windows = x.unfold(0, k, stride).transpose(1, 2)
    return windows.mean(dim=1).cpu().numpy()


from datasets import concatenate_datasets

def build_arrow_from_pickle(
    pickle_path: str,
    arrow_path: str,
    k: int = 100,
    stride: int = 20,
    max_examples: Union[int, None] = None
):
    with open(pickle_path, 'rb') as f:
        raw = pickle.load(f)

    data = []
    for idx, item in enumerate(raw):
        if max_examples and idx >= max_examples:
            break
        label = 1 if 'true' in item else 0
        seq = item.get('true', item.get('negative'))
        embA_raw = seq[0][0].detach().cpu()
        embB_raw = seq[1][0].detach().cpu()
        pooledA = _bom_pool(embA_raw, k, stride)
        pooledB = _bom_pool(embB_raw, k, stride)
        data.append({
            'embA': pooledA.astype('float32'),
            'embB': pooledB.astype('float32'),
            'lenA': pooledA.shape[0],
            'lenB': pooledB.shape[0],
            'label': int(label),
        })

    if not data:
        raise ValueError("No data processed from pickle file.")

    D = data[0]['embA'].shape[1]
    features = Features({
        'embA': Array2D(dtype='float32', shape=(None, D)),
        'embB': Array2D(dtype='float32', shape=(None, D)),
        'lenA': Value('int32'),
        'lenB': Value('int32'),
        'label': Value('int8'),
    })

    # Break into safe chunks to avoid Arrow offset overflow
    chunks = []
    batch_size = 100  # adjust if needed
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        chunk_ds = Dataset.from_list(batch, features=features)
        chunks.append(chunk_ds)

    ds = concatenate_datasets(chunks)
    os.makedirs(arrow_path, exist_ok=True)
    ds.save_to_disk(arrow_path)
    print(f"Arrow dataset saved to {arrow_path} ({len(data)} examples)")


def load_arrow_dataset(
    arrow_path: str,
    streaming: bool = False
):
    if streaming:
        import glob
        shards = sorted(glob.glob(os.path.join(arrow_path, "*.arrow")))
        return load_dataset(
            'arrow', data_files={'train': shards}, split='train', streaming=True
        )
    return load_from_disk(arrow_path)

