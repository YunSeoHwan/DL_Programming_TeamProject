# data_loader.py

import os
import xml.etree.ElementTree as ET

class_name_to_id = {
    'bundle of ropes': 0,
    # 필요한 경우 다른 클래스도 추가
}

def parse_annotation(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        class_id = class_name_to_id.get(name, -1)  # 클래스 이름을 정수로 변환
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax, class_id])
    
    return boxes

def load_dataset_paths(image_dir, annotation_dir):
    image_paths = []
    labels = []

    annotation_files = os.listdir(annotation_dir)
    for annotation_file in annotation_files:
        annotation_path = os.path.join(annotation_dir, annotation_file)
        image_file = annotation_file.replace('.xml', '.jpg')
        image_path = os.path.join(image_dir, image_file)

        if os.path.exists(image_path):
            boxes = parse_annotation(annotation_path)
            image_paths.append(image_path)
            labels.append(boxes)
    
    return image_paths, labels
