import os
import math
import csv
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from pytorchvideo.data.encoded_video import EncodedVideo
from model import Model

# ===== CONFIG =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "saved_models/888tiny.pkl"

video_root_folder = "eval_video/normal"  # 처리할 영상 폴더
segment_score_folder = "segment_scores"  # 구간별 점수 CSV 저장 폴더
os.makedirs(segment_score_folder, exist_ok=True)

output_csv = "sliding_result.csv"
threshold = 0.5
valid_exts = [".mp4", ".avi"]

# ===== X3D 설정 =====
model_name = 'x3d_l'
transform_params = {
    "num_frames": 15,
    "sampling_rate": 4,
    "side_size": 320,
    "crop_size": 320,
    "frames_per_second": 30,
}
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]

# ===== Normalize 함수 =====
def normalize_video_tensor(video_tensor, mean, std):
    mean = torch.tensor(mean, device=video_tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=video_tensor.device).view(1, -1, 1, 1)
    return (video_tensor - mean) / std

# ===== 고정 간격 샘플링 =====
def fixed_interval_sample(frames, interval, num_samples):
    total_needed = interval * num_samples
    if frames.shape[0] < total_needed:
        pad = frames[-1:].repeat(total_needed - frames.shape[0], 1, 1, 1)
        frames = torch.cat([frames, pad], dim=0)
    indices = torch.arange(0, total_needed, step=interval)
    return frames[indices]

# ===== 전처리 함수 =====
def preprocess_frames(frames):
    frames = frames.permute(0, 3, 1, 2) / 255.0  # (T, H, W, C) → (T, C, H, W)
    frames = normalize_video_tensor(frames, mean, std)
    frames = torch.nn.functional.interpolate(
        frames, size=(transform_params["side_size"], transform_params["side_size"]),
        mode="bilinear", align_corners=False
    )
    crop = transform_params["crop_size"]
    center = transform_params["side_size"] // 2
    frames = frames[:, :, center - crop//2:center + crop//2,
                          center - crop//2:center + crop//2]
    frames = frames.permute(1, 0, 2, 3)  # (T, C, H, W) → (C, T, H, W)
    return frames

# ===== X3D 모델 로딩 =====
x3d = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
x3d = nn.Sequential(*list(x3d.blocks[:5]))  # classification head 제거
x3d = x3d.to(device).eval()

# ===== 이상 판단 함수 =====
def predict_anomaly(feature, model):
    with torch.no_grad():
        score, _ = model(feature.unsqueeze(0).to(device))  # (1, C, T, H, W)
        prob = torch.sigmoid(score).item()
    return prob

# ===== 슬라이딩 인퍼런스 =====
def stream_anomaly_inference(video_path, csv_save_path, model):
    video = EncodedVideo.from_path(video_path)
    fps = transform_params["frames_per_second"]
    duration = float(video.duration)
    window_sec = (transform_params["num_frames"] * transform_params["sampling_rate"]) / fps
    stride_sec = 1.0

    total_steps = max(1, math.ceil((duration - window_sec) / stride_sec) + 1)
    window_scores = []

    with open(csv_save_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["start_sec", "end_sec", "score"])

        for step in range(total_steps):
            start = step * stride_sec
            end = start + window_sec
            try:
                clip = video.get_clip(start_sec=start, end_sec=end)
            except:
                break
            frames = clip["video"]
            if frames.shape[0] == 3:
                frames = frames.permute(1, 2, 3, 0)
            elif frames.shape[-1] != 3:
                raise ValueError(f"Unsupported frame shape: {frames.shape}")

            frames = fixed_interval_sample(frames, transform_params["sampling_rate"], transform_params["num_frames"])
            frames = preprocess_frames(frames).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = x3d(frames).squeeze(0)
                score = predict_anomaly(feature, model)
                label = "Abnormal" if score > threshold else "Normal"
                # print(f"[{start:.1f}s ~ {end:.1f}s]  Score: {score:.4f}  → {label}")

                window_scores.append(score)
                writer.writerow([round(start,2), round(end,2), round(score,4)])

    # video_score = max(window_scores) if window_scores else 0.0
    video_score = float(np.mean(window_scores)) if window_scores else 0.0
    video_pred = int(video_score >= threshold)
    return video_score, video_pred

# ===== MAIN =====
if __name__ == "__main__":
    model = Model(ff_mult=1, dims=(32, 32), depths=(1, 1)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []
    folder_name = os.path.basename(video_root_folder.lower())

    video_files = [
        fname for fname in os.listdir(video_root_folder)
        if os.path.splitext(fname)[-1].lower() in valid_exts
    ]

    for fname in tqdm(video_files, desc="🔍 Processing videos"):
        video_path = os.path.join(video_root_folder, fname)
        csv_save_path = os.path.join(segment_score_folder, fname.split('.')[0] + "_scores.csv")

        # ===== GT 결정 =====
        if folder_name == "normal":
            gt = 0
        elif folder_name in ["v_easy", "v_hard"]:
            gt = 1
        elif folder_name == "total":
            lower_name = fname.lower()
            if lower_name.startswith("normal"):
                gt = 0
            elif lower_name.startswith("violence") or lower_name.startswith("abnormal"):
                gt = 1
            else:
                print(f"⚠️ Skipping (Unknown label in 'total'): {fname}")
                continue
        else:
            print(f"⚠️ Unknown folder: {video_root_folder}, skipping {fname}")
            continue

        try:
            video_score, video_pred = stream_anomaly_inference(video_path, csv_save_path, model)
            results.append({"name": fname, "p": video_score, "pred": video_pred, "GT": gt})
        except Exception as e:
            print(f"🚨 Error - {fname}: {e}")
            continue

    if len(results) == 0:
        print("⚠️ No valid videos processed. Check folder or file naming.")
        exit()

    # 결과 저장
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # ===== Metrics 출력 =====
    y_true = df["GT"]
    y_pred = df["pred"]
    print("\n📊 Evaluation Metrics:")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"\n✅ Video-level result saved to: {output_csv}")
    print(f"✅ Segment CSV saved in folder: {segment_score_folder}")
