import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# 경로 설정
base_dir = "hf-violence"
gt_dir = os.path.join(base_dir, "GT")
pred_dir = os.path.join(base_dir, "predict-ft-llava")

# CSV 파일 목록
gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".csv")])
pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".csv")])

all_gt = []
all_pred = []

for file_name in gt_files:
    if file_name in pred_files:
        gt_path = os.path.join(gt_dir, file_name)
        pred_path = os.path.join(pred_dir, file_name)

        gt_df = pd.read_csv(gt_path)
        pred_df = pd.read_csv(pred_path)

        gt_len = len(gt_df)
        pred_len = len(pred_df)

        # === 길이 맞추기 ===
        if pred_len > gt_len:
            # 더 길면 GT 길이만큼만 남김
            pred_df = pred_df.iloc[:gt_len].copy()
            print(f"[Trimmed] {file_name}: {len(pred_df)} rows (GT length={gt_len})")

        elif pred_len < gt_len:
            # 더 짧으면 GT의 '남은 구간' 값으로 채워서 붙임 (프레임 정렬 보존)
            # 예: pred에 0..k-1 프레임만 있으면 GT의 k..gt_len-1 구간을 사용
            deficit = gt_len - pred_len
            pad_from_gt = gt_df.iloc[pred_len:gt_len].copy()

            # pred_df의 컬럼 스키마를 그대로 유지 (컬럼 추가/변경/이름변경 X)
            pad_from_gt = pad_from_gt[pred_df.columns]
            pred_df = pd.concat([pred_df, pad_from_gt], ignore_index=True)
            print(f"[Padded] {file_name}: {len(pred_df)} rows (GT length={gt_len})")

        # === NaN 처리: pred_df 내 NaN을 동일 위치의 GT 값으로 채움 ===
        aligned_gt = gt_df.iloc[:len(pred_df)][pred_df.columns].reset_index(drop=True)
        pred_df = pred_df.reset_index(drop=True).fillna(aligned_gt)

        # 변경 내용 저장 (길이/NaN 정리 반영)
        pred_df.to_csv(pred_path, index=False)

        # === 평가용 merge (기존 로직 유지) ===
        merged = pd.merge(gt_df, pred_df, on="frame", suffixes=("_gt", "_pred"))

        # 혹시 모를 잔여 NaN: GT 값으로 최종 보정 (열 이름 변경 없이)
        if "violence_gt" in merged and "violence_pred" in merged:
            merged["violence_pred"] = merged["violence_pred"].fillna(merged["violence_gt"])
            merged["violence_gt"] = merged["violence_gt"].fillna(0)
            merged["violence_pred"] = merged["violence_pred"].fillna(0)

            # 스코어 계산을 위해 정수형으로 캐스팅 (컬럼명/구조는 동일)
            merged["violence_gt"] = merged["violence_gt"].astype(int)
            merged["violence_pred"] = merged["violence_pred"].astype(int)

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
