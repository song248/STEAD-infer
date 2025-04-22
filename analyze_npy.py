import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from model import Model
import glob

# ====== 설정 ======
NPY_DIR = 'my_video_npy'
OUTPUT_CSV_DIR = 'output/csv'
OUTPUT_GRAPH_DIR = 'output/graph'
MODEL_PATH = 'saved_models/888tiny.pkl'
CLIP_LEN = 16
THRESHOLD = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)

# ====== 모델 로딩 ======
def load_model():
    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# ====== 클립 점수 → 프레임 확률 분배 ======
def distribute_clip_scores(clip_scores, total_frames, clip_len=16):
    frame_scores = np.zeros(total_frames)
    counts = np.zeros(total_frames)

    for i, score in enumerate(clip_scores):
        start = i * clip_len
        end = min(start + clip_len, total_frames)
        frame_scores[start:end] += score
        counts[start:end] += 1

    counts[counts == 0] = 1  # division safeguard
    return frame_scores / counts

# ====== 분석 함수 ======
def analyze_npy(model, npy_path):
    video_name = os.path.basename(npy_path).replace('.npy', '')
    features = np.load(npy_path)  # shape: (C, T, H, W)

    # 모델 입력 형태로 변환: (1, C, T, H, W)
    input_tensor = torch.from_numpy(features).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        scores, _ = model(input_tensor)
        clip_score = torch.sigmoid(scores).squeeze().item()

    total_frames = features.shape[1] * CLIP_LEN  # T * CLIP_LEN
    clip_scores = [clip_score]  # 모델은 영상 단위 1점 반환
    frame_probs = distribute_clip_scores(clip_scores, total_frames, CLIP_LEN)
    frame_preds = (frame_probs >= THRESHOLD).astype(int)

    # ==== CSV 저장 ====
    df = pd.DataFrame({'frame': np.arange(total_frames), 'violence': frame_preds})
    csv_path = os.path.join(OUTPUT_CSV_DIR, f'{video_name}.csv')
    df.to_csv(csv_path, index=False)

    # ==== 그래프 저장 ====
    plt.figure(figsize=(12, 4))
    plt.plot(frame_probs, label='Anomaly Probability')
    plt.axhline(THRESHOLD, color='red', linestyle='--', label='Threshold')
    plt.xlabel('Frame')
    plt.ylabel('Anomaly Probability')
    plt.title(video_name)
    plt.legend()
    plt.tight_layout()
    graph_path = os.path.join(OUTPUT_GRAPH_DIR, f'{video_name}.png')
    plt.savefig(graph_path)
    plt.close()

    print(f"✅ Processed: {video_name} – CSV + Graph saved.")

# ====== 메인 함수 ======
def main():
    model = load_model()
    npy_files = glob.glob(os.path.join(NPY_DIR, '*.npy'))

    print(f"📁 Found {len(npy_files)} .npy files")
    for npy_path in npy_files:
        analyze_npy(model, npy_path)

if __name__ == '__main__':
    main()
