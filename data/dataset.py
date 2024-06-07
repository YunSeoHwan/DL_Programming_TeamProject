import os
import xml.etree.ElementTree as ET
from PIL import Image
from typing import Type
import torch
import torchvision.transforms as tr
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir, bbox_dir, transform):
        self.image_dir = image_dir
        self.bbox_dir = bbox_dir
        self.data_images = [file for file in os.listdir(image_dir) if file.endswith('.jpg')]
        self.transform = transform

        # 레이블 매핑 딕셔너리
        self.label_map = {
            'tire': 0,
            'spring fish trap': 1,
            'circular fish trap': 1,
            'rectangular fish trap': 1,
            'eel fish trap': 1,
            'fish net': 2,
            'wood': 3,
            'rope': 4,
            'bundle of ropes': 4
        }

    def parse_bbox_xml(self, xml_file, image_width, image_height):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            label = self.label_map.get(name, -1)  # 레이블 매핑, 없으면 -1
            
            if label == -1:
                continue

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text) / image_width
            ymin = float(bbox.find('ymin').text) / image_height
            xmax = float(bbox.find('xmax').text) / image_width
            ymax = float(bbox.find('ymax').text) / image_height

            # 바운딩 박스의 중심점과 너비, 높이 계산
            x_center = (xmin + xmax) / 2
            y_center = (ymin + ymax) / 2
            width = xmax - xmin
            height = ymax - ymin

            boxes.append([x_center, y_center, width, height])
            labels.append(label)

        # 객체가 1개만 있는 경우만 반환
        if len(boxes) == 1 and len(labels) == 1:
            return boxes, labels
        else:
            return None, None

    def __getitem__(self, idx):
        img_file = self.data_images[idx]
        img_path = os.path.join(self.image_dir, img_file)
        image = Image.open(img_path).convert("RGB")
        image_width, image_height = image.size

        # xml 파일 경로 찾기
        xml_file = os.path.join(self.bbox_dir, img_file.replace('.jpg', '.xml'))

        boxes, labels = self.parse_bbox_xml(xml_file, image_width, image_height)
        
        # boxes 또는 labels가 None이면 다음 데이터로 넘어감
        if boxes is None or labels is None:
            return self.__getitem__((idx + 1) % len(self.data_images))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        image = self.transform(image)

        return image, labels, boxes

    def __len__(self):
        return len(self.data_images)

def make_dataset(image_dir: str,
                 bbox_dir: str,
                 transform=tr.Compose([tr.Resize((512, 512)), 
                                       tr.ToTensor(),                                       
                                       tr.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                       ])
                 ) -> Type[torch.utils.data.Dataset]:
    """
    Make pytorch Dataset for given task.
    Read the image using the PIL library and return it as an np.array.

    Args:
        image_dir (str): dataset directory
        bbox_dir (str) : dataset directory
        transform (torchvision.transforms) pytorch image transforms  

    Returns:
        torch.Dataset: pytorch Dataset
    """
        
    dataset = CustomDataset(image_dir=image_dir,
                            bbox_dir=bbox_dir,
                            transform=transform)
            
    return dataset

def collate_fn(batch):
    images, labels, boxes = zip(*batch)
    images = torch.stack(images, 0)
    return images, labels, boxes

# 저장할 경로
image_dir_train = 'train/images/'
bbox_dir_train = 'train/bbox/'

image_dir_val = 'validation/images/'
bbox_dir_val = 'validation/bbox/'

image_dir_test = 'test/images/'
bbox_dir_test = 'test/bbox/'

# 라벨 파일 생성 및 저장
def save_label_file(image_file, labels, boxes, output_dir):
    label_file = image_file.replace('.jpg', '.txt')
    with open(os.path.join(output_dir, label_file), 'w') as f:
        for label, box in zip(labels, boxes):
            x_center, y_center, width, height = box
            line = f"{label} {x_center} {y_center} {width} {height}\n"
            f.write(line)

def save_label_files(dataset, image_dir, bbox_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx in range(len(dataset)):
        image, labels, boxes = dataset[idx]
        image_file = dataset.data_images[idx]
        save_label_file(image_file, labels, boxes, output_dir)

# train set
train_dataset = make_dataset(image_dir_train, bbox_dir_train)
save_label_files(train_dataset, image_dir_train, bbox_dir_train, 'train/labels')

# validation set
val_dataset = make_dataset(image_dir_val, bbox_dir_val)
save_label_files(val_dataset, image_dir_val, bbox_dir_val, 'validation/labels')

# test set
test_dataset = make_dataset(image_dir_test, bbox_dir_test)
save_label_files(test_dataset, image_dir_test, bbox_dir_test, 'test/labels')