import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from model import Model  # 기존 구조 재사용
import matplotlib.pyplot as plt
import umap.umap_ as umap

VIDEO_FEATURE_PATH = 'my_video_npy'  # .npy 경로
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ====== Custom Dataset ======
class NPYDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label_dir in os.listdir(root_dir):
            full_path = os.path.join(root_dir, label_dir)
            if not os.path.isdir(full_path):
                continue
            label = 0 if label_dir.lower() == 'normal' else 1
            for file in os.listdir(full_path):
                if file.endswith('.npy'):
                    self.samples.append((os.path.join(full_path, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        x = np.load(path).astype(np.float32)
        return torch.from_numpy(x), label

# ====== Test Function (Same as test.py) ======
def test(dataloader, model, device='cuda', name='my_test', main=False):
    model.to(device).eval()
    preds, labels, feats = [], [], []

    with torch.no_grad():
        for x, label in dataloader:
            x = x.to(device)
            scores, feat = model(x)
            score = torch.sigmoid(scores).squeeze().item()  # float 값
            preds.append(score)
            labels.append(label.item())
            feats.append(feat.squeeze().cpu().numpy())
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(labels, preds)
    pr_auc = auc(recall, precision)

    print(f'ROC AUC: {roc_auc:.4f}')
    print(f'PR AUC: {pr_auc:.4f}')

    if main:
        feats = np.array(feats)
        labels = np.array(labels)
        reducer = umap.UMAP()
        reduced_feats = reducer.fit_transform(feats)
        plt.figure()
        plt.scatter(reduced_feats[labels == 0, 0], reduced_feats[labels == 0, 1], c='blue', label='Normal')
        plt.scatter(reduced_feats[labels == 1, 0], reduced_feats[labels == 1, 1], c='red', label='Anomaly')
        plt.legend()
        plt.title("UMAP Embedding")
        plt.savefig(f"{name}_embed.png")

    return roc_auc, pr_auc

# ====== Main ======
if __name__ == '__main__':
    dataset = NPYDataset(VIDEO_FEATURE_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load("saved_models/888tiny.pkl"))
    
    test(dataloader, model, device=DEVICE, name="my_test", main=True)
