import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# 경로 설정
base_dir = "hf-violence"
gt_dir = os.path.join(base_dir, "GT")
pred_dir = os.path.join(base_dir, "predict-ft")

# CSV 파일 목록
gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".csv")])
pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".csv")])

all_gt = []
all_pred = []

for file_name in gt_files:
    if file_name in pred_files:
        gt_path = os.path.join(gt_dir, file_name)
        pred_path = os.path.join(pred_dir, file_name)
        
        # CSV 읽기
        gt_df = pd.read_csv(gt_path)
        pred_df = pd.read_csv(pred_path)
        
        # frame 기준으로 merge
        merged = pd.merge(gt_df, pred_df, on="frame", suffixes=("_gt", "_pred"))
        
        all_gt.extend(merged["violence_gt"].tolist())
        all_pred.extend(merged["violence_pred"].tolist())

# 전체 metric 계산
precision = precision_score(all_gt, all_pred, zero_division=0)
recall = recall_score(all_gt, all_pred, zero_division=0)
accuracy = accuracy_score(all_gt, all_pred)
f1 = f1_score(all_gt, all_pred, zero_division=0)

print("=== Overall Metrics ===")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"F1-score:  {f1:.4f}")
