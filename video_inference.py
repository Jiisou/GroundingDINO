import cv2
from groundingdino.util.inference import load_model, load_image, predict, annotate
import os
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm


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

    # numpy.ndarray를 PIL 이미지로 변환
    image_source = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV에서 BGR을 RGB로 변환
    image_pil = Image.fromarray(image_source)  # numpy.ndarray에서 PIL.Image로 변환

    # PIL 이미지를 Torch Tensor로 변환
    transform = transforms.ToTensor()  # ToTensor 변환 객체 생성
    image_tensor = transform(image_pil)  # PIL.Image를 Torch Tensor로 변환

    # image_tensor는 이제 Torch Tensor 형태로, device로 이동 가능합니다.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_tensor = image_tensor.to(device)

    # 객체 탐지 수행
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
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