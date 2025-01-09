import cv2
from groundingdino.util.inference import load_model, load_image, predict, annotate
import os

# 모델 로딩
model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/work/GroundingDINO/weights/groundingdino_swint_ogc.pth")

# 비디오 경로와 출력 디렉토리 설정
VIDEO_PATH = "/home/work/GroundingDINO/video/tuktuk_01.mp4"
OUTPUT_DIR = "/home/work/GroundingDINO/video/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TEXT_PROMPT = "car . bus . truck . motorcycle . tuktuk" #indonesia traffic measurement pjt
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# 비디오 파일 열기
cap = cv2.VideoCapture(VIDEO_PATH)

# 프레임 번호 초기화
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 비디오 끝

    # 프레임 전처리 및 객체 탐지 수행
    image_source, image = load_image(frame)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )

    # 객체 탐지 결과를 이미지에 주석 추가
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

    # 결과를 저장할 파일 경로 설정
    output_filename = os.path.join(OUTPUT_DIR, f"frame_{frame_num:04d}.jpg")
    
    # 주석이 달린 프레임 저장
    cv2.imwrite(output_filename, annotated_frame)

    # 프레임 번호 증가
    frame_num += 1

# 비디오 파일 닫기
cap.release()
print(f"프레임 단위 객체 탐지 완료! 출력 폴더: {OUTPUT_DIR}")
