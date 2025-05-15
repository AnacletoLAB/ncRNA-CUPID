import os
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
import h5py
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from sklearn.metrics import roc_auc_score, average_precision_score

from model_hug import RNACrossAttentionHF, RNACrossAttentionConfig
from dataset_hug import build_h5_from_pickle

import warnings
warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0"
)
warnings.filterwarnings(
    "ignore",
    message="Could not estimate the number of tokens"
)

class SaveBestModelCallback(TrainerCallback):
    """
    Callback to save the best model based on validation AUPRC.
    """
    def __init__(self):
        self.best_score = -float('inf')

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = state.log_history[-1]
        current = metrics.get('eval_auprc')
        if current is not None and current > self.best_score:
            self.best_score = current
            save_path = os.path.join(args.output_dir, 'best_model')
            kwargs['model'].save_pretrained(save_path)
            print(f"New best AUPRC: {current:.4f}. Saved model to {save_path}")


def load_tensor_dataset(h5_path: str) -> TensorDataset:
    with h5py.File(h5_path, 'r') as h5:
        arrA = h5['embA'][:]
        arrB = h5['embB'][:]
        labels = h5['labels'][:]
        lenA = h5['lenA'][:]
        lenB = h5['lenB'][:]

    As = torch.from_numpy(arrA)
    Bs = torch.from_numpy(arrB)
    lenAs = torch.from_numpy(lenA)
    lenBs = torch.from_numpy(lenB)
    labs = torch.tensor(labels, dtype=torch.float)

    return TensorDataset(As, Bs, lenAs, lenBs, labs)


def collate_fn(batch):
    As, Bs, lAs, lBs, labs = zip(*batch)
    As = torch.stack(As, dim=0); Bs = torch.stack(Bs, dim=0)
    lAs = torch.stack(lAs, dim=0); lBs = torch.stack(lBs, dim=0)
    max_w = As.size(1); idxs = torch.arange(max_w)
    maskA = idxs.unsqueeze(0) >= lAs.unsqueeze(1)
    maskB = idxs.unsqueeze(0) >= lBs.unsqueeze(1)
    labels = torch.tensor(labs, dtype=torch.float)
    return {'A': As, 'B': Bs, 'maskA': maskA, 'maskB': maskB, 'labels': labels}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = labels.astype(int)
    return {
        'auroc': roc_auc_score(labels, probs),
        'auprc': average_precision_score(labels, probs)
    }


def main():
    seed = 3
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    train_pickle = 'db_strat/classification_aug_types_training.p'
    h5_path = 'data/pooled_dataset_hug_fullsize_128_64.h5'
    if not os.path.exists(h5_path):
        build_h5_from_pickle(train_pickle, h5_path, k=128, stride=64)

    full_ds = load_tensor_dataset(h5_path)
    n = len(full_ds)
    split = int(0.8 * n)
    train_ds, val_ds = random_split(full_ds, [split, n-split], generator=torch.Generator().manual_seed(seed))

    batch_size = 2048
    steps_per_epoch = max(1, split // batch_size)

    config = RNACrossAttentionConfig()
    model = RNACrossAttentionHF(config)

    training_args = TrainingArguments(
        output_dir='./hf_rna_cross',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=30,
        learning_rate=5e-4,
        weight_decay=5e-5,
        logging_dir='./logs',
        do_eval=True,
        eval_steps=steps_per_epoch,
        save_steps=steps_per_epoch,
        metric_for_best_model='auprc',  
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=3
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2), SaveBestModelCallback()]
    )

    trainer.train()
    print('Training done. Best model in hf_rna_cross/best_model')

if __name__ == '__main__':
    main()

