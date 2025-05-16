import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import roc_auc_score, average_precision_score
from dataset_hug import build_arrow_from_pickle, load_arrow_dataset
from model_hug import RNACrossAttentionHF, RNACrossAttentionConfig
import gc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------- GPU/Memory utilities ----------------
def free_memory():
    """
    Frees up unused GPU and CPU memory.
    Call this whenever you’ve deleted large tensors/models
    and want to reclaim space immediately.
    """
    gc.collect()
    torch.cuda.empty_cache()

# ---------------- Metrics & Callbacks ----------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = labels.astype(int)
    return {
        'auroc': roc_auc_score(labels, probs),
        'auprc': average_precision_score(labels, probs)
    }

class SaveBestModelCallback(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience: int = 2):
        super().__init__(early_stopping_patience=early_stopping_patience)
        self.best_score = -float('inf')

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        free_memory()
        if metrics is None:
            return control
        current = metrics.get('eval_auprc')
        if current and current > self.best_score:
            self.best_score = current
            save_path = os.path.join(args.output_dir, 'best_model')
            kwargs['model'].save_pretrained(save_path)
            print(f"New best AUPRC: {current:.4f}. Model saved to {save_path}")
        return control

# ---------------- Collation ----------------
def collate_fn(batch):
    # Batch: list of dicts containing torch.Tensors
    As = [item['embA'] for item in batch]
    Bs = [item['embB'] for item in batch]
    lensA = torch.tensor([item['lenA'].item() for item in batch], dtype=torch.long)
    lensB = torch.tensor([item['lenB'].item() for item in batch], dtype=torch.long)
    labels = torch.tensor([item['label'].item() for item in batch], dtype=torch.float)

    Bsz = len(As)
    maxA, maxB = int(lensA.max()), int(lensB.max())
    D = As[0].size(1)

    padA = torch.zeros((Bsz, maxA, D), dtype=torch.float)
    padB = torch.zeros((Bsz, maxB, D), dtype=torch.float)
    maskA = torch.ones((Bsz, maxA), dtype=torch.bool)
    maskB = torch.ones((Bsz, maxB), dtype=torch.bool)

    for i, (a, b) in enumerate(zip(As, Bs)):
        La, Lb = a.size(0), b.size(0)
        padA[i, :La] = a
        padB[i, :Lb] = b
        maskA[i, :La] = False
        maskB[i, :Lb] = False

    return {'A': padA, 'B': padB, 'maskA': maskA, 'maskB': maskB, 'labels': labels}

# ---------------- Balanced Sampler ----------------
class BalancedBatchSampler:
    """
    Sampler yielding batches with a fixed ratio of negatives (without replacement)
    and positives (with replacement).
    """
    def __init__(self, labels, batch_size, neg_batch_ratio=0.7):
        # labels: numpy array or tensor
        self.labels = labels.numpy() if torch.is_tensor(labels) else np.array(labels)
        self.batch_size = batch_size
        self.neg_batch_ratio = neg_batch_ratio

        # indices per class
        self.neg_indices = np.where(self.labels == 0)[0]
        self.pos_indices = np.where(self.labels == 1)[0]

        # counts per batch
        self.num_neg = int(batch_size * neg_batch_ratio)
        self.num_pos = batch_size - self.num_neg

        # number of batches per epoch
        full = len(self.neg_indices) // self.num_neg
        rem = len(self.neg_indices) % self.num_neg
        self.num_batches = full + (1 if rem > 0 else 0)
        self.leftover = rem

    def __iter__(self):
        neg_perm = np.random.permutation(self.neg_indices)
        start = 0
        for i in range(self.num_batches):
            # negatives for this batch
            if i == self.num_batches - 1 and self.leftover:
                neg_count = self.leftover
            else:
                neg_count = self.num_neg
            neg_batch = neg_perm[start:start+neg_count]
            start += neg_count

            # positives with replacement
            pos_batch = np.random.choice(self.pos_indices, self.batch_size - neg_count, replace=True)

            batch = np.concatenate([neg_batch, pos_batch])
            np.random.shuffle(batch)
            yield batch.tolist()

    def __len__(self):
        return self.num_batches

# ---------------- Custom Trainer ----------------
class BalancedTrainer(Trainer):
    def get_train_dataloader(self):
        # Extract labels array from HuggingFace Dataset
        labels = np.array(self.train_dataset['label'])
        sampler = BalancedBatchSampler(
            labels=labels,
            batch_size=self.args.per_device_train_batch_size,
            neg_batch_ratio=0.7
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# ---------------- Main script ----------------
def main(
    max_examples: int = None,
    k: int = 128,
    stride: int = 64,
    test_ratio: float = 0.2
):
    seed = 3
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    pickle_file = 'db_strat/classification_aug_types_training.p'
    arrow_dir = 'data/arrow_dataset_full'

    if not os.path.exists(arrow_dir):
        print('Building dataset')
        build_arrow_from_pickle(pickle_file, arrow_dir, k=k, stride=stride, max_examples=max_examples)
    free_memory()

    print('Loading dataset')
    ds = load_arrow_dataset(arrow_dir, streaming=False)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    free_memory()

    print('Splitting dataset')
    split = ds.train_test_split(test_size=test_ratio, seed=seed)
    train_ds, eval_ds = split['train'], split['test']
    free_memory()

    # format for torch
    cols = ['embA', 'embB', 'lenA', 'lenB', 'label']
    train_ds.set_format(type='torch', columns=cols)
    eval_ds.set_format(type='torch', columns=cols)

    # model config
    config = RNACrossAttentionConfig()
    model = RNACrossAttentionHF(config)

    total = len(train_ds)
    batch_size, epochs = 2048, 30
    steps_per_epoch = (total + batch_size - 1) // batch_size
    max_steps = steps_per_epoch * epochs

    training_args = TrainingArguments(
        output_dir='./hf_rna_cross_fullbalanced__128_64',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-4,
        weight_decay=5e-5,
        logging_dir='./logs',
        do_eval=True,
        eval_strategy='steps',
        eval_steps=steps_per_epoch,
        save_strategy='steps',
        save_steps=steps_per_epoch,
        metric_for_best_model='eval_auprc',
        load_best_model_at_end=True,
        remove_unused_columns=False,
        save_total_limit=3,
        max_steps=max_steps,
        dataloader_num_workers=6,
        dataloader_pin_memory=True,
    )

    trainer = BalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[SaveBestModelCallback()]
    )

    print('Begin Training')
    free_memory()
    trainer.train()
    print(f'Training complete. Best model saved in {training_args.output_dir}/best_model')

if __name__ == '__main__':
    main()
