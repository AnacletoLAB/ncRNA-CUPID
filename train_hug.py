## train_hug.py
import os
import random
import numpy as np
import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import roc_auc_score, average_precision_score
from dataset_hug import build_arrow_from_pickle, load_arrow_dataset
from model_hug import RNACrossAttentionHF, RNACrossAttentionConfig
import gc


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def free_memory():
    """
    Frees up unused GPU and CPU memory.
    Call this whenever you’ve deleted large tensors/models
    and want to reclaim space immediately.
    """
    # 1. Delete any Python references to large tensors/models
    #    For example, if you’ve just finished with `outputs` or `loss`:
    # del outputs, loss

    # 2. Run garbage collection on Python side
    gc.collect()

    # 3. Release unreferenced CUDA memory
    torch.cuda.empty_cache()

    # 4. (Optional) synchronize to ensure all kernels have finished
    #if torch.cuda.is_available():
    #    torch.cuda.synchronize()

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


def collate_fn(batch):
    # Batch is list of dicts with torch.Tensor fields
    As = [item['embA'] for item in batch]
    Bs = [item['embB'] for item in batch]
    lensA = torch.tensor([item['lenA'].item() for item in batch], dtype=torch.long)
    lensB = torch.tensor([item['lenB'].item() for item in batch], dtype=torch.long)
    labels = torch.tensor([item['label'].item() for item in batch], dtype=torch.float)

    Bsz = len(As)
    maxA, maxB = lensA.max().item(), lensB.max().item()
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

    # Load full dataset in-memory and split
    print('Loading dataset')
    ds = load_arrow_dataset(arrow_dir, streaming=False)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    free_memory()

    print('Splitting dataset')
    split = ds.train_test_split(test_size=test_ratio, seed=seed)
    train_ds, eval_ds = split['train'], split['test']

    free_memory()

    # Ensure columns available as torch tensors
    cols = ['embA', 'embB', 'lenA', 'lenB', 'label']
    train_ds.set_format(type='torch', columns=cols)
    eval_ds.set_format(type='torch', columns=cols)

    # Initialize model and training args
    config = RNACrossAttentionConfig()
    model = RNACrossAttentionHF(config)

    total = len(train_ds)
    batch_size, epochs = 2048, 30
    steps = ((total + batch_size - 1) // batch_size) * epochs
    steps_per_epoch = ((total + batch_size - 1) // batch_size)

    training_args = TrainingArguments(
        output_dir='./hf_rna_cross_full_128_64',
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
        max_steps=steps
    )

    trainer = Trainer(
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
    print(f'Training complete. Best model in {output_dir}')


if __name__ == '__main__':
    #main(max_examples=100)
    main()


