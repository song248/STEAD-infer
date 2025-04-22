import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from feat_extractor import X3DExtractor
from model import STEAD
import pickle

# --------- 설정 ---------
VIDEO_DIR = './violence'
OUTPUT_DIR = './output'
MODEL_LOCATION = 'saved_models/'
MODEL_NAME = '888tiny'
MODEL_EXTENSION = '.pkl'
MODEL_PATH = os.path.join(MODEL_LOCATION, MODEL_NAME + MODEL_EXTENSION)

CLIP_LEN = 16
THRESHOLD = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESIZE_SHAPE = (160, 160)
# -------------------------

os.makedirs(os.path.join(OUTPUT_DIR, 'graphs'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'csvs'), exist_ok=True)


def extract_frames(video_path, resize=(160, 160)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def frames_to_clips(frames, clip_len=16):
    clips = []
    for i in range(0, len(frames) - clip_len + 1, clip_len):
        clip = frames[i:i + clip_len]
        clips.append(np.stack(clip, axis=0))  # (T, H, W, C)
    return clips


def interpolate_scores(clip_scores, total_frames, clip_len=16):
    frame_scores = np.zeros(total_frames)
    counts = np.zeros(total_frames)
    for i, score in enumerate(clip_scores):
        start = i * clip_len
        end = min(start + clip_len, total_frames)
        frame_scores[start:end] += score
        counts[start:end] += 1
    counts[counts == 0] = 1
    return frame_scores / counts


def save_graph(probs, video_name):
    plt.figure(figsize=(12, 4))
    plt.plot(probs, label='Anomaly Probability')
    plt.axhline(y=THRESHOLD, color='r', linestyle='--', label='Threshold = 0.5')
    plt.xlabel('Frame')
    plt.ylabel('Anomaly Probability')
    plt.title(f'{video_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'graphs', f'{video_name}.png'))
    plt.close()


def save_csv(probs, video_name):
    preds = (probs >= THRESHOLD).astype(int)
    df = pd.DataFrame({
        'frame': np.arange(len(probs)),
        'violence': preds
    })
    df.to_csv(os.path.join(OUTPUT_DIR, 'csvs', f'{video_name}.csv'), index=False)


@torch.no_grad()
def process_video(video_path, x3d, stead):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames = extract_frames(video_path, RESIZE_SHAPE)
    total_frames = len(frames)
    clips = frames_to_clips(frames, CLIP_LEN)

    clip_scores = []
    for clip in clips:
        clip_tensor = torch.tensor(clip).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        clip_tensor = clip_tensor.to(DEVICE)
        features = x3d(clip_tensor)  # shape: (1, C, T', H', W')
        score = torch.sigmoid(stead(features)).item()  # anomaly score (0~1)
        clip_scores.append(score)

    frame_probs = interpolate_scores(clip_scores, total_frames, CLIP_LEN)
    save_graph(frame_probs, video_name)
    save_csv(frame_probs, video_name)


def main():
    print("🚀 Loading models...")
    x3d = X3DExtractor().to(DEVICE).eval()

    with open(MODEL_PATH, 'rb') as f:
        stead = pickle.load(f)
    stead = stead.to(DEVICE).eval()

    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    print(f"📂 Found {len(video_files)} video(s) in '{VIDEO_DIR}'. Processing...")

    for vf in tqdm(video_files):
        video_path = os.path.join(VIDEO_DIR, vf)
        process_video(video_path, x3d, stead)

    print(f"✅ Done! Results saved in: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
