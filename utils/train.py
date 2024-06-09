import os
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer

class CustomTrainer(BaseTrainer):
    def epoch_end(self):
        super().epoch_end()  # 기본 epoch_end 메서드를 호출하여 기본 동작을 유지
        weight_path = f"weights/epoch_{self.epoch}.pt"
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)  # 디렉토리가 없으면 생성
        self.model.save(weight_path)
        print(f"Saved weights for epoch {self.epoch} at {weight_path}")

def main():
    # 모델 초기화
    model = YOLO('../data/yolov8n.pt')

    # 데이터와 학습 설정
    data = 'data.yaml'
    epochs = 100
    patience = 5
    batch = 128
    imgsz = (512, 512)
    device = 0

    # 사용자 정의 트레이너 설정
    model.trainer = CustomTrainer

    # 학습
    model.train(data=data, epochs=epochs, patience=patience, batch=batch, imgsz=imgsz, device=device)

if __name__ == "__main__":
    main()
