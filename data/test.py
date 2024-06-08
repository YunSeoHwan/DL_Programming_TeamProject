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
import cv2
import numpy as np

def draw_bounding_boxes(image, boxes):
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, f'Class: {class_id}, Conf: {confidence:.2f}', 
                    (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                    (0, 255, 0), 2)
    
    return image

def main():
    # 모델 초기화
    best_model = YOLO('runs/detect/train/weights/best.pt')
    
    # 예측할 이미지 수 설정
    num_images = 4
    
    # test 이미지 경로
    image_dir = 'test/images'
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')][:num_images]

    for image_file in image_files:
        # 이미지 예측
        results = best_model.predict(source=image_file, save=False, imgsz=(512, 512))

        for result in results:
            print(f"Image: {result.path}")
            for box in result.boxes:
                print(f"Class: {box.cls}, Confidence: {box.conf}, Bbox: {box.xyxy}")

            # 이미지 로드
            image = cv2.imread(result.path)
            
            # 바운딩 박스 시각화
            image_with_boxes = draw_bounding_boxes(image, result.boxes)
            
            # 시각화 결과 저장
            output_path = result.path.replace('test/images', 'test/output')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image_with_boxes)

if __name__ == "__main__":
    main()