import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from model_data_cross_att_old import RNACrossAttentionClassifier
from test_dataset import InteractionTestDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset
test_dataset = InteractionTestDataset(
    couples_path='db_strat/classification_aug_types_test.p',
    pooling_mode='bom',
    k = 256,
    stride = 128
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load model
model = RNACrossAttentionClassifier()
model.load_state_dict(torch.load('ckpt/best_model_256_128_ep10.pt', map_location=device))
model.to(device)
model.eval()

# Prediction loop
y_true_all = []
y_prob_all = []
types_all = []

with torch.no_grad():
    #counter = 0
    for embA, embB, label, itype in test_loader:
        #counter += 1
        #if counter > 39900:
            #break
        
        embA = embA[0].to(device)
        embB = embB[0].to(device)
        label = float(label[0])
        itype = itype[0]

        logits = model(embA, embB)
        prob = torch.sigmoid(logits).item()

        y_true_all.append(label)
        y_prob_all.append(prob)
        types_all.append(itype)

# Global metrics
auroc_global = roc_auc_score(y_true_all, y_prob_all)
auprc_global = average_precision_score(y_true_all, y_prob_all)

print("=== Global Metrics ===")
print(f"AUROC : {auroc_global:.4f}")
print(f"AUPRC : {auprc_global:.4f}")


# Per-type metrics
# TODO
