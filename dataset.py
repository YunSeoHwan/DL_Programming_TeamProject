# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class UnderwaterTrashDataset(Dataset):
    def __init__(self, image_paths, labels, image_shape):
        self.image_paths = image_paths
        self.labels = labels
        self.image_shape = image_shape
        self.transform = transforms.Compose([
            transforms.Resize((image_shape[0], image_shape[1])),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.float32)
        return image, label

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)

    max_boxes = max([label.size(0) for label in labels])
    padded_labels = []

    for label in labels:
        if label.size(0) < max_boxes:
            padded_label = torch.cat([label, torch.zeros((max_boxes - label.size(0), 5))], dim=0)
        else:
            padded_label = label
        padded_labels.append(padded_label)
    
    padded_labels = torch.stack(padded_labels)
    
    return images, padded_labels

def create_dataloader(image_paths, labels, image_shape, batch_size=32):
    dataset = UnderwaterTrashDataset(image_paths, labels, image_shape)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    return dataloader
