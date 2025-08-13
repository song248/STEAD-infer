#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
hf_eval3.py
- GT:        {BASE}/{GT_DIR}/{name}.csv           (columns: frame, violence)
- PRED(raw): {BASE}/{PRED_DIR}/{name}.csv         (columns: frame, prob[, violence])
- SAVE(real):{BASE}/{PRED_DIR}-real/{name}.csv    (columns: frame, violence)
- REPORT:    {BASE}/{PRED_DIR}-real/report.csv    (per-file best threshold & metrics)

동작:
1) BASE/GT_DIR과 BASE/PRED_DIR에서 공통 파일명을 찾는다.
2) 각 파일에 대해:
   - GT와 PRED를 frame 기준으로 정합(LEFT JOIN on GT frame).
   - PRED의 prob NaN은 해당 위치 GT로 채움(이전 스크립트의 보정 규칙 유지).
   - 해당 파일에서 F1이 최대가 되는 threshold(τ*) 탐색.
   - τ*로 (prob >= τ*) → violence(0/1) 생성.
   - 결과를 SAVE_DIR = f"{PRED_DIR}-real" 에 frame,violence 두 컬럼으로 저장.
   - 파일별 best threshold와 성능(P/R/A/F1)을 기록.
3) 전체 파일을 합산한 전역 성능(Precision/Recall/Accuracy/F1) 출력.
4) 파일별 요약을 SAVE_DIR/report.csv 로 저장.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# ===== Defaults =====
DEFAULT_BASE = "hf-violence"
DEFAULT_GT_DIR = "GT"
DEFAULT_PRED_DIR = "predict-ft"

# ===== Utils =====
def load_csv_normalized(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 기본 검증
    lower = {c.lower(): c for c in df.columns}
    if "frame" not in lower:
        raise ValueError(f"'frame' column missing in {path}")
    return df

def best_threshold_for_file(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    파일 단위 F1 최대 임계값 탐색.
    - 후보: y_prob의 고유값 + 보강 격자(필요 시).
    - 동률이면 0.5에 더 가까운 쪽 선호.
    """
    y_prob = y_prob.astype(float, copy=True)
    nan_mask = np.isnan(y_prob)
    if nan_mask.any():
        y_prob[nan_mask] = 0.0

    uniq = np.unique(y_prob)
    candidates = uniq.tolist()
    if len(candidates) < 3:
        # 값이 너무 단조로우면 격자 보강
        candidates = sorted(set(candidates) | set(np.linspace(0.0, 1.0, 101).tolist()))

    best_tau = 0.5
    best_f1 = -1.0
    y_true_u8 = y_true.astype(np.uint8)

    for tau in candidates:
        y_pred = (y_prob >= tau).astype(np.uint8)
        f1 = f1_score(y_true_u8, y_pred, zero_division=0)
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and abs(tau - 0.5) < abs(best_tau - 0.5)):
            best_f1 = f1
            best_tau = float(tau)
    return best_tau

# ===== Main Eval → Save Real Labels + Report =====
def main():
    parser = argparse.ArgumentParser(description="Per-file thresholding to produce real (binary) labels and report.")
    parser.add_argument("--base_dir", default=DEFAULT_BASE, help="Root directory (contains GT and PRED dirs).")
    parser.add_argument("--gt_dir", default=DEFAULT_GT_DIR, help="Ground truth directory under base_dir.")
    parser.add_argument("--pred_dir", default=DEFAULT_PRED_DIR, help="Pred(probs) directory under base_dir.")
    # save_dir는 명시 안 하면 pred_dir + '-real' 자동 생성
    parser.add_argument("--save_dir", default=None, help="Output directory under base_dir for real labels and report.csv.")
    args = parser.parse_args()

    base_dir = args.base_dir
    gt_dir = os.path.join(base_dir, args.gt_dir)
    pred_dir = os.path.join(base_dir, args.pred_dir)
    save_dir_name = args.save_dir if args.save_dir else f"{args.pred_dir}-real"
    save_dir = os.path.join(base_dir, save_dir_name)

    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"GT dir not found: {gt_dir}")
    if not os.path.isdir(pred_dir):
        raise FileNotFoundError(f"Pred dir not found: {pred_dir}")

    os.makedirs(save_dir, exist_ok=True)

    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".csv")])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".csv")])
    common = [f for f in gt_files if f in pred_files]

    if not common:
        print("No common CSV filenames between GT and PRED directories.")
        print(f"GT:   {gt_dir}")
        print(f"PRED: {pred_dir}")
        return

    print(f"GT dir:    {gt_dir}")
    print(f"PRED dir:  {pred_dir}")
    print(f"SAVE dir:  {save_dir}  (auto from pred_dir + '-real')" if args.save_dir is None else f"SAVE dir:  {save_dir} (from --save_dir)")

    saved_count = 0
    per_file_rows = []  # report rows

    # 전체 합산용 버퍼(기존 hf_eval2 스타일)
    all_gt = []
    all_pred = []

    for file_name in common:
        gt_path = os.path.join(gt_dir, file_name)
        pred_path = os.path.join(pred_dir, file_name)

        # Load
        gt_df = load_csv_normalized(gt_path)
        pred_df = load_csv_normalized(pred_path)

        # Column names (원래 케이스 유지)
        gt_frame_col = [c for c in gt_df.columns if c.lower() == "frame"][0]
        gt_label_col = [c for c in gt_df.columns if c.lower() == "violence"][0]  # GT는 violence 필수
        pred_frame_col = [c for c in pred_df.columns if c.lower() == "frame"][0]

        # Pred에는 prob 필수
        prob_cols = [c for c in pred_df.columns if c.lower() == "prob"]
        if not prob_cols:
            print(f"[Skip] {file_name}: Pred missing 'prob' column")
            continue
        pred_prob_col = prob_cols[0]

        # Align by frame (LEFT JOIN on GT frames only)
        merged = pd.merge(
            gt_df[[gt_frame_col, gt_label_col]],
            pred_df[[pred_frame_col, pred_prob_col]],
            left_on=gt_frame_col,
            right_on=pred_frame_col,
            how="left",
            suffixes=("_gt", "_pred"),
        )

        # Canonicalize
        merged = merged.rename(columns={
            gt_frame_col: "frame",
            gt_label_col: "violence_gt",
            pred_prob_col: "prob",
        })
        if pred_frame_col in merged.columns and pred_frame_col != "frame":
            merged = merged.drop(columns=[pred_frame_col])

        # Fill NaNs: prob ← GT( float )
        merged["prob"] = merged["prob"].astype(float)
        nan_mask = merged["prob"].isna()
        if nan_mask.any():
            merged.loc[nan_mask, "prob"] = merged.loc[nan_mask, "violence_gt"].astype(float)

        # Types / bounds
        merged["violence_gt"] = merged["violence_gt"].fillna(0).astype(int)
        merged["prob"] = merged["prob"].clip(0.0, 1.0)

        y_true = merged["violence_gt"].to_numpy()
        y_prob = merged["prob"].to_numpy()

        # Find best τ* for this file
        tau_star = best_threshold_for_file(y_true, y_prob)

        # Make real (binary) predictions with τ*
        y_pred = (y_prob >= tau_star).astype(np.int32)

        # Save: frame, violence (2 cols), same filename, to SAVE_DIR
        out_df = pd.DataFrame({
            "frame": merged["frame"].astype(int).to_numpy(),
            "violence": y_pred,
        })
        out_path = os.path.join(save_dir, file_name)
        out_df.to_csv(out_path, index=False)
        saved_count += 1

        # Per-file metrics (for report)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        a = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        per_file_rows.append({
            "file": file_name,
            "best_threshold": tau_star,
            "precision": p,
            "recall": r,
            "accuracy": a,
            "f1": f1,
        })

        # Aggregate for overall metrics
        all_gt.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

        print(f"[Saved] {file_name}: best τ={tau_star:.6f}  P={p:.4f}  R={r:.4f}  A={a:.4f}  F1={f1:.4f}  → {out_path}")

    # Save per-file report
    report_path = os.path.join(save_dir, "report.csv")
    if per_file_rows:
        pd.DataFrame(per_file_rows).to_csv(report_path, index=False)
        print(f"\nPer-file report saved to: {report_path}")
    else:
        print("\nNo per-file rows to report (nothing saved).")

    # Overall metrics (global, by concatenating all frames)
    if all_gt and all_pred:
        precision = precision_score(all_gt, all_pred, zero_division=0)
        recall = recall_score(all_gt, all_pred, zero_division=0)
        accuracy = accuracy_score(all_gt, all_pred)
        f1 = f1_score(all_gt, all_pred, zero_division=0)

        print("\n=== Overall Metrics (with per-file best thresholds) ===")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"F1-score:  {f1:.4f}")
    else:
        print("\nNo samples aggregated for overall metrics. Check your GT/Pred CSVs.")

    print(f"\nDone. {saved_count} file(s) written to {save_dir}.")

if __name__ == "__main__":
    main()
