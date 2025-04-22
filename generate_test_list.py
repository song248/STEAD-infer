import os

video_dir = 'violence'  # 영상이 저장된 상위 폴더
output_txt = 'violence_Test.txt'

with open(output_txt, 'w') as f:
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith('.mp4'):
                # 상대 경로로 기록
                rel_path = os.path.join(root, file).replace('\\', '/')
                f.write(f"{rel_path}\n")

print(f"✅ {output_txt} 파일 생성 완료! 총 {sum(1 for line in open(output_txt))}개 항목 저장됨.")
