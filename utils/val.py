import os
from ultralytics import YOLO

def main():
    # 모델 초기화
    model = YOLO('../data/runs/detect/train/weights/best.pt')

    # 테스트 데이터에 대한 성능 평가
    results = model.val(data='data.yaml', imgsz=(512, 512), split='test')

if __name__ == "__main__":
    main()