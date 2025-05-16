import os
import torch
import numpy as np
import pickle
from typing import List, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score
from datasets import Dataset, Features, Value, Array2D, concatenate_datasets, load_from_disk
from transformers import Trainer, TrainingArguments

from model_hug import RNACrossAttentionHF, RNACrossAttentionConfig

def _bom_pool(emb: torch.Tensor, k: int, stride: int) -> np.ndarray:
    x = emb.squeeze(0)
    if x.size(0) < k:
        return x.mean(dim=0, keepdim=True).cpu().numpy()
    windows = x.unfold(0, k, stride).transpose(1, 2)
    return windows.mean(dim=1).cpu().numpy()

def build_test_arrow_from_pickle(
    pickle_path: str,
    arrow_path: str,
    k: int = 100,
    stride: int = 20,
    max_examples: Optional[int] = None
) -> List[Tuple[str, int]]:
    with open(pickle_path, 'rb') as f:
        raw = pickle.load(f)

    data = []
    types = []
    for idx, item in enumerate(raw):
        if max_examples is not None and idx >= max_examples:
            break
        if 'true' in item:
            pair, type_str = item['true']
            label = 1
        else:
            pair, type_str = item['negative']
            label = 0

        embA_raw = pair[0][0].detach().cpu()
        embB_raw = pair[1][0].detach().cpu()
        pooledA = _bom_pool(embA_raw, k, stride)
        pooledB = _bom_pool(embB_raw, k, stride)

        data.append({
            'embA': pooledA.astype('float32'),
            'embB': pooledB.astype('float32'),
            'lenA': pooledA.shape[0],
            'lenB': pooledB.shape[0],
            'label': int(label),
        })
        types.append((type_str, label))

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

    chunks = []
    batch_size = 100
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        chunk_ds = Dataset.from_list(batch, features=features)
        chunks.append(chunk_ds)

    ds = concatenate_datasets(chunks)
    os.makedirs(arrow_path, exist_ok=True)
    ds.save_to_disk(arrow_path)

    # Set format to return dicts (like training dataset)
    ds.set_format(type='torch', columns=['embA', 'embB', 'lenA', 'lenB', 'label'], output_all_columns=True)

    # Save types
    type_path = os.path.join(arrow_path, "types.pkl")
    with open(type_path, "wb") as f:
        pickle.dump(types, f)

    print(f"Test Arrow dataset saved to {arrow_path} ({len(data)} examples)")
    return types

def load_test_arrow_dataset(arrow_path: str):
    ds = load_from_disk(arrow_path)
    ds.set_format(type='torch', columns=['embA', 'embB', 'lenA', 'lenB', 'label'], output_all_columns=True)
    return ds

def load_test_types(arrow_path: str) -> List[Tuple[str, int]]:
    type_path = os.path.join(arrow_path, "types.pkl")
    with open(type_path, "rb") as f:
        return pickle.load(f)

def run_test_evaluation(trainer: Trainer, test_dataset, types: List[Tuple[str, int]]):
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions.squeeze()
    labels = predictions.label_ids
    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    # Overall metrics
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)

    print(f"\n[Overall Test Set]\nAUROC: {auroc:.4f}\nAUPRC: {auprc:.4f}")

    # Per-type metrics
    type_to_scores = {}
    print(types[0])
    for i, (t, y) in enumerate(types):
        if t not in type_to_scores:
            type_to_scores[t] = {'probs': [], 'labels': []}
        type_to_scores[t]['probs'].append(probs[i])
        type_to_scores[t]['labels'].append(y)

    print("\n[Per-Type Metrics]")
    for t, data in type_to_scores.items():
        y_true = np.array(data['labels'])
        y_pred = np.array(data['probs'])
        if len(np.unique(y_true)) > 1:
            t_auroc = roc_auc_score(y_true, y_pred)
            t_auprc = average_precision_score(y_true, y_pred)
            print(f'{t[0]}->{t[1]}\{label:20} | AUROC: {t_auroc:.4f} | AUPRC: {t_auprc:.4f}')
        else:
            print(f"{t[0]}->{t[1]} | AUROC: N/A    | AUPRC: N/A (single class)")

def run_test(
    model_dir: str,
    test_pickle_path: str,
    test_arrow_path: str,
    k: int = 128,
    stride: int = 64,
    batch_size: int = 1024,
    max_examples: Optional[int]=None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    from train_hug import collate_fn

    print('model loading')
    config = RNACrossAttentionConfig.from_pretrained(model_dir)
    model = RNACrossAttentionHF.from_pretrained(model_dir, config=config).to(device)

    print('dataset loading')
    if os.path.exists(os.path.join(test_arrow_path, "dataset_info.json")) and \
       os.path.exists(os.path.join(test_arrow_path, "types.pkl")):
        test_dataset = load_test_arrow_dataset(test_arrow_path)
        types = load_test_types(test_arrow_path)
    else:
        print('dataset not found. constructing dataset')
        types = build_test_arrow_from_pickle(test_pickle_path, test_arrow_path, k, stride, max_examples)
        test_dataset = load_test_arrow_dataset(test_arrow_path)

    test_args = TrainingArguments(
        output_dir="./test_results",
        per_device_eval_batch_size=batch_size,
        dataloader_drop_last=False,
        report_to="none",
        do_train=False,
        do_eval=True,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=test_args,
        compute_metrics=None,
        data_collator=collate_fn
    )

    print('test start')
    run_test_evaluation(trainer, test_dataset, types)

if __name__ == "__main__":
    run_test(
        model_dir="hf_rna_cross_full_128_64/best_model",
        test_pickle_path="db_strat/classification_aug_types_test.p",
        test_arrow_path="data/test_dataset_full"
    )
