import os
import xml.etree.ElementTree as ET

def load_images_and_annotations(image_dir, annotation_dir):
    images = []
    annotations = []
    
    print(f"Loading images and annotations from {image_dir} and {annotation_dir}")
    
    for annotation_file in os.listdir(annotation_dir):
        if annotation_file.endswith('.xml'):
            annotation_path = os.path.join(annotation_dir, annotation_file)
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            
            image_file = root.find('filename').text
            image_path = os.path.join(image_dir, image_file)
            
            if os.path.exists(image_path):
                images.append(image_path)
                annotations.append(annotation_path)
    
    print(f"Loaded {len(images)} images and {len(annotations)} annotations.")
    return images, annotations
