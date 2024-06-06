import shutil
import os

def prepare_dataset(images, annotations, output_dir):
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Split data into train and val
    split_idx = int(0.8 * len(images))
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    train_annotations = annotations[:split_idx]
    val_annotations = annotations[split_idx:]
    
    # Copy train images and labels
    for image, annotation in zip(train_images, train_annotations):
        shutil.copy(image, train_images_dir)
        shutil.copy(annotation, train_labels_dir)
    
    # Copy val images and labels
    for image, annotation in zip(val_images, val_annotations):
        shutil.copy(image, val_images_dir)
        shutil.copy(annotation, val_labels_dir)

    print(f"Dataset prepared with {len(train_images)} training and {len(val_images)} validation images.")

def create_yaml(output_dir, classes):
    yaml_content = f"""
    train: {os.path.join(output_dir, 'train', 'images')}
    val: {os.path.join(output_dir, 'val', 'images')}

    nc: {len(classes)}
    names: {classes}
    """
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    print(f"YAML file created at {os.path.join(output_dir, 'data.yaml')}")
