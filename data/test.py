# import os
# from ultralytics import YOLO

# def main():
#     # 모델 초기화
#     best_model = YOLO('runs/detect/train/weights/best.pt')

#     # test
#     results = best_model.predict(source='test/images', save=True, imgsz=(512, 512))

#     # 결과 출력
#     for result in results:
#         print(f"Image: {result.path}")
#         for box in result.boxes:
#             print(f"Class: {box.cls}, Confidence: {box.conf}, Bbox: {box.xyxy}")

# if __name__ == "__main__":
#     main()
import os
from ultralytics import YOLO

def main():
    # 모델 초기화
    model = YOLO('runs/detect/train/weights/best.pt')

    # 테스트 데이터에 대한 성능 평가
    results = model.val(data='data.yaml', imgsz=(512, 512), split='test')

if __name__ == "__main__":
    main()
