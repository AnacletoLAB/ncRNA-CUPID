import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from model_data_cross_att import InteractionDataset, RNACrossAttentionClassifier
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def collate_fn(batch):
    embAs, embBs, labels = zip(*batch)
    return list(embAs), list(embBs), torch.stack(labels)


def train(
    data_path: str,
    k: int = 100,
    stride: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    epochs: int = 50,
    val_split: float = 0.1,
    patience: int = 5,
    device: str = 'cuda'
):
    # reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # dataset
    print('loading dataset now...')
    dataset = InteractionDataset(data_path, pooling_mode='bom', k=k, stride=stride)
    n_val = int(len(dataset) * val_split)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    print('Done. now preparing loaders...')

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,         # spawn 8 worker processes
        pin_memory=True,       # faster host→GPU copies
        prefetch_factor=2     # each worker prefetches 2 batches
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True
    )

    # model, loss, optimizer
    model = RNACrossAttentionClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    print('Model initialized, begin training...')
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        counter = 0
        for embAs, embBs, labels in train_loader:
            optimizer.zero_grad()
            logits = torch.stack([
                model(a.to(device), b.to(device)) for a, b in zip(embAs, embBs)
            ])
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            counter +=1
            print('counter iterations: ', counter)

        avg_train_loss = total_loss / len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for embAs, embBs, labels in val_loader:
                logits = torch.stack([
                    model(a.to(device), b.to(device)) for a, b in zip(embAs, embBs)
                ])
                loss = criterion(logits, labels.to(device))
                val_loss += loss.item() * len(labels)
        avg_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # save checkpoint
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

    # load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train RNA-RNA interaction model')
    parser.add_argument('--data', type=str, required=True, help='path to pickle data')
    parser.add_argument('--k', type=int, default=100)
    parser.add_argument('--stride', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--val_split', type=float, default=0.8)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    train(
        data_path=args.data,
        k=args.k,
        stride=args.stride,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        val_split=args.val_split,
        patience=args.patience,
        device=args.device
    )
