# preprocess.py

from tqdm import tqdm

def preprocess_labels(labels, image_shape):
    processed_labels = []
    for label in tqdm(labels, desc="Preprocessing Labels"):
        boxes = []
        for box in label:
            xmin, ymin, xmax, ymax, name = box
            xmin /= image_shape[1]
            xmax /= image_shape[1]
            ymin /= image_shape[0]
            ymax /= image_shape[0]
            boxes.append([xmin, ymin, xmax, ymax, name])
        processed_labels.append(boxes)
    return processed_labels
