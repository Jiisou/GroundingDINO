import cv2
from groundingdino.util.inference import load_model, predict
from PIL import Image
from torchvision import transforms
import torch
import time
from groundingdino.util.misc import nested_tensor_from_tensor_list

def process_video_inference_time(model, input_path, text_prompt, box_threshold, text_threshold):
    # 비디오 파일 열기
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Info - Total Frames: {total_frames}")

    inference_times = []

    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # # 이미지 변환 및 Nested Tensor로 변환
        # image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # transform = transforms.ToTensor()
        # image_tensor = transform(image_pil).unsqueeze(0)  # (1, C, H, W)
        # samples = nested_tensor_from_tensor_list([image_tensor])

        # numpy.ndarray를 PIL 이미지로 변환
        image_source = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV에서 BGR을 RGB로 변환
        image_pil = Image.fromarray(image_source)  # numpy.ndarray에서 PIL.Image로 변환
        # PIL 이미지를 Torch Tensor로 변환
        transform = transforms.ToTensor()  # ToTensor 변환 객체 생성
        image_tensor = transform(image_pil)  # PIL.Image를 Torch Tensor로 변환
        # image_tensor는 이제 Torch Tensor 형태로, device로 이동 가능합니다.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sample_tensor = image_tensor.to(device)

        # 추론 시간 측정
        start_time = time.time()
        _ = predict(
            model=model,
            image=sample_tensor, 
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        end_time = time.time()

        inference_times.append(end_time - start_time)

    cap.release()

    # 결과 계산
    total_time = sum(inference_times)
    avg_time_per_frame = total_time / len(inference_times) if inference_times else 0
    fps = 1 / avg_time_per_frame if avg_time_per_frame > 0 else 0

    print(f"\nTotal Inference Time: {total_time:.2f} seconds")
    print(f"Average Time Per Frame: {avg_time_per_frame:.4f} seconds")
    print(f"Estimated FPS: {fps:.2f}")


def main():
    # 모델 로딩
    model = load_model(
        "groundingdino/config/GroundingDINO_SwinT_OGC.py", 
        "/home/work/GroundingDINO/weights/groundingdino_swint_ogc.pth"
    )
    
    # 설정값
    INPUT_VIDEO_PATH = "/home/work/GroundingDINO/video/target_v1/upperview_30s.mp4"
    TEXT_PROMPT = "car . bus . truck . motorcycle . tuktuk"
    BOX_THRESHOLD = 0.30
    TEXT_THRESHOLD = 0.30
    
    # 추론 시간 측정
    process_video_inference_time(
        model=model,
        input_path=INPUT_VIDEO_PATH,
        text_prompt=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

if __name__ == "__main__":
    main()
