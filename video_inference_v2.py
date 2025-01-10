import cv2
from groundingdino.util.inference import load_model, predict, annotate
import os
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm

def process_video(model, input_path, output_path, text_prompt, box_threshold, text_threshold):
    # 비디오 파일 열기
    cap = cv2.VideoCapture(input_path)
    
    # 비디오의 총 프레임 수와 프레임 크기 가져오기
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 비디오 출력 파일 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))
    
    # tqdm을 사용하여 진행 상황 표시
    for frame_num in tqdm(range(total_frames), desc=f"Processing {os.path.basename(input_path)}", ncols=100):
        ret, frame = cap.read()
        if not ret:
            break
            
        # numpy.ndarray를 PIL 이미지로 변환
        image_source = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_source)
        
        # PIL 이미지를 Torch Tensor로 변환
        transform = transforms.ToTensor()
        image_tensor = transform(image_pil)
        
        # device 설정
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_tensor = image_tensor.to(device)
        
        # 객체 탐지 수행
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        # 객체 탐지 결과를 이미지에 주석 추가
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        
        # 주석이 달린 프레임을 비디오 출력에 추가
        out.write(annotated_frame)
    
    # 비디오 파일 닫기
    cap.release()
    out.release()

def main():
    # 모델 로딩
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                      "/home/work/GroundingDINO/weights/groundingdino_swint_ogc.pth")
    
    # 설정값
    INPUT_VIDEO_DIR = "/home/work/GroundingDINO/video/targets"
    OUTPUT_VIDEO_DIR = "/home/work/GroundingDINO/video/results_v2/"
    TEXT_PROMPT = "car . bus . truck . motorcycle . tuktuk"
    BOX_THRESHOLD = 0.30
    TEXT_THRESHOLD = 0.30
    
    # 출력 디렉토리가 없으면 생성
    os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)
    
    # 지원하는 비디오 확장자
    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')
    
    # 입력 디렉토리의 모든 비디오 파일 처리
    video_files = [f for f in os.listdir(INPUT_VIDEO_DIR) 
                  if os.path.isfile(os.path.join(INPUT_VIDEO_DIR, f)) 
                  and f.lower().endswith(VIDEO_EXTENSIONS)]
    
    print(f"Found {len(video_files)} video files to process")
    
    # 각 비디오 파일 처리
    for video_file in video_files:
        input_path = os.path.join(INPUT_VIDEO_DIR, video_file)
        vid_name = os.path.splitext(video_file)[0]
        output_path = os.path.join(OUTPUT_VIDEO_DIR, f"{vid_name}_infer.mp4")
        
        try:
            process_video(
                model=model,
                input_path=input_path,
                output_path=output_path,
                text_prompt=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD
            )
            print(f"Successfully processed: {video_file} -> {output_path}")
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            continue
    
    print("All videos processed!")

if __name__ == "__main__":
    main()