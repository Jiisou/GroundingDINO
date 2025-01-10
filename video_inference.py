import cv2
from groundingdino.util.inference import load_model, predict, annotate
import os
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm

# 모델 로딩
model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/work/GroundingDINO/weights/groundingdino_swint_ogc.pth")

# 비디오 경로와 출력 비디오 설정
INPUT_VIDEO_PATH = "/home/work/GroundingDINO/video/tuktuk_01.mp4"

filename = INPUT_VIDEO_PATH.split("/")[-1] #cat_dog.jpeg
vid_name = filename.split(".")[0]
OUTPUT_VIDEO_PATH = f"/home/work/GroundingDINO/video/results/{vid_name}_infer.mp4"

# 비디오 파일 열기
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

# 비디오의 총 프레임 수와 프레임 크기 가져오기
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 비디오 출력 파일 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 비디오 코덱
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 30.0, (frame_width, frame_height))  # 초당 30프레임

TEXT_PROMPT = "car . bus . truck . motorcycle . tuktuk"  # indonesia traffic measurement pjt
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

# 프레임 번호 초기화
frame_num = 0

# tqdm을 사용하여 진행 상황 표시
for frame_num in tqdm(range(total_frames), desc="Processing frames", ncols=100):
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

    # 주석이 달린 프레임을 비디오 출력에 추가
    out.write(annotated_frame)

# 비디오 파일 닫기
cap.release()
out.release()

print(f"비디오 객체 탐지 완료! 출력 비디오 파일: {OUTPUT_VIDEO_PATH}")
