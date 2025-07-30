import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import Model
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torch import nn
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# ===== CONFIG =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "saved_models/888tiny.pkl"
video_root_folder = "eval_video/v_easy"
threshold = 0.5
output_csv = "v_easy-result.csv"
valid_exts = [".mp4", ".avi"]

# ===== X3D 설정 =====
model_name = 'x3d_l'
transform_params = {
    "num_frames": 16,
    "sampling_rate": 5,
    "side_size": 320,
    "crop_size": 320,
    "frames_per_second": 30,
}
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"]) / transform_params["frames_per_second"]

# ===== Transform 및 모델 준비 =====
def temporal_subsample(frames: torch.Tensor, num_samples: int):
    t = frames.shape[0]
    if t < num_samples:
        pad = frames[-1:].repeat(num_samples - t, 1, 1, 1)
        frames = torch.cat([frames, pad], dim=0)
        t = frames.shape[0]
    indices = torch.linspace(0, t - 1, num_samples).long()
    return frames[indices]

def normalize_video_tensor(video_tensor, mean, std):
    mean = torch.tensor(mean, device=video_tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=video_tensor.device).view(1, -1, 1, 1)
    return (video_tensor - mean) / std

transform = Compose([
    Lambda(lambda x: temporal_subsample(x, transform_params["num_frames"])),
    Lambda(lambda x: x.permute(0, 3, 1, 2) / 255.0),
    Lambda(lambda x: normalize_video_tensor(x, mean, std)),
    Lambda(lambda x: torch.nn.functional.interpolate(
        x, size=(transform_params["side_size"], transform_params["side_size"]),
        mode="bilinear", align_corners=False)),
    Lambda(lambda x: x[:, :,
                       transform_params["side_size"]//2 - transform_params["crop_size"]//2:
                       transform_params["side_size"]//2 + transform_params["crop_size"]//2,
                       transform_params["side_size"]//2 - transform_params["crop_size"]//2:
                       transform_params["side_size"]//2 + transform_params["crop_size"]//2]),
    Lambda(lambda x: x.permute(1, 0, 2, 3)),
])

x3d = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
x3d = nn.Sequential(*list(x3d.blocks[:5])).to(device).eval()

def extract_x3d_feature(path, fname=""):
    try:
        video = EncodedVideo.from_path(path)
        clip = video.get_clip(start_sec=0, end_sec=clip_duration)
    except Exception as e:
        print(f"⚠️ get_clip() failed: {fname} - {e}")
        raise

    frames = clip["video"]
    if frames.shape[0] == 3:
        frames = frames.permute(1, 2, 3, 0)
    elif frames.shape[-1] != 3:
        raise ValueError(f"Unsupported frame shape: {frames.shape}")
    frames = transform(frames).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = x3d(frames).squeeze(0)
    return feat
def predict_anomaly(feature, model):
    with torch.no_grad():
        score, _ = model(feature.unsqueeze(0))
        prob = torch.sigmoid(score).item()
    return prob

# ===== MAIN LOOP =====
if __name__ == "__main__":
    model = Model(ff_mult=1, dims=(32, 32), depths=(1, 1)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []

    video_files = [
        fname for fname in os.listdir(video_root_folder)
        if os.path.splitext(fname)[-1].lower() in valid_exts
    ]

    for fname in tqdm(video_files, desc="🔍 Processing videos"):
        video_path = os.path.join(video_root_folder, fname)
        try:
            feature = extract_x3d_feature(video_path, fname=fname)  # ← 파일명 같이 넘김
            p = predict_anomaly(feature, model)
            pred = int(p >= threshold)

            parent_folder = os.path.basename(os.path.dirname(video_path))
            gt = 0 if parent_folder == "normal" else 1

            results.append({"name": fname, "p": p, "pred": pred, "GT": gt})
        except Exception as e:
            print(f"🚨 Error - {fname}: {e}")
            continue

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)

    # ===== METRICS 출력 =====
    y_true = df["GT"]
    y_pred = df["pred"]
    print("\n📊 Evaluation Metrics (normal only):")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"\n✅ Saved to: {output_csv}")
