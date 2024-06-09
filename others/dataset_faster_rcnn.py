import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import xml.etree.ElementTree as ET
from PIL import ImageOps

class CustomDataset(Dataset):
    def __init__(self, img_dir, annotations_dir, transform=None):
        self.img_dir = img_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        annotation_path = os.path.join(self.annotations_dir, self.image_files[idx].replace('.jpg', '.xml'))
        image = Image.open(img_path).convert("RGB")
        orig_width, orig_height = image.size

        # 이미지 패딩 적용: 이미지 중앙에 위치하도록 패딩 추가
        max_dim = max(orig_width, orig_height)
        padding_left = (max_dim - orig_width) // 2
        padding_top = (max_dim - orig_height) // 2
        padding_right = max_dim - orig_width - padding_left
        padding_bottom = max_dim - orig_height - padding_top
        image = ImageOps.expand(image, (padding_left, padding_top, padding_right, padding_bottom), fill='black')

        # Resize 이미지를 512x512로 조정
        image = image.resize((512, 512))

        if self.transform:
            image = self.transform(image)

        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # 패딩을 고려한 좌표 변환
        scale = 512 / max_dim  # 원본 최대 크기에 대한 새 크기의 비율
        boxes = []
        labels = []
        for obj in root.findall('object'):
            original_label = obj.find('name').text.lower()
            label = self.simplify_label(original_label)
            bbox = obj.find('bndbox')
            xmin = (float(bbox.find('xmin').text) + padding_left) * scale
            ymin = (float(bbox.find('ymin').text) + padding_top) * scale
            xmax = (float(bbox.find('xmax').text) + padding_left) * scale
            ymax = (float(bbox.find('ymax').text) + padding_top) * scale
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor([self.label_to_index(label) for label in labels], dtype=torch.int64)

        targets = {'boxes': boxes, 'labels': labels}
        return {'image': image, 'targets': targets}

    def simplify_label(self, label):
        label = label.lower()
        if 'tire' in label:
            return 'tire'
        elif 'spring fish trap' in label or 'circular fish trap' in label or 'rectangular fish trap' in label or 'eel fish trap' in label or 'trap' in label:
            return 'trap'
        elif 'fish net' in label or 'net' in label:
            return 'fish net'
        elif 'wood' in label:
            return 'wood'
        elif 'rope' in label or 'bundle of ropes' in label:
            return 'rope'
        elif 'other objects' in label or 'othe objects' in label or 'other objets' in label:
            return 'other objects'

    def label_to_index(self, label):
        label_dict = {'tire': 0, 'trap': 1, 'fish net': 2, 'wood': 3, 'rope': 4, 'other objects':5}
        return label_dict[label]
    
def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])  
    targets = []
    for item in batch:
        target = {
            'boxes': item['targets']['boxes'],  
            'labels': item['targets']['labels']  
        }
        targets.append(target)
    return {'image': images, 'targets': targets}
    
def get_data_loaders():
    transform = transforms.Compose([
    # (높이, 너비) 형식으로 이미지 크기를 명확히 조정
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    train_dataset = CustomDataset(img_dir='./data/train/images', annotations_dir='./data/train/bbox', transform=transform)
    val_dataset = CustomDataset(img_dir='./data/validation/images', annotations_dir='./data/validation/bbox', transform=transform)
    test_dataset = CustomDataset(img_dir='./data/test/images', annotations_dir='./data/test/bbox', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loaders()

