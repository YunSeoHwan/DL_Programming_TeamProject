import xml.etree.ElementTree as ET

CLASS_MAPPING = {
    "tire": "타이어",
    "spring fish trap": "통발류",
    "circular fish trap": "통발류",
    "rectangular fish trap": "통발류",
    "eel fish trap": "통발류",
    "fish net": "어망류",
    "wood": "나무",
    "rope": "로프",
    "bundle of ropes": "로프"
}

def convert_annotations(annotation_paths):
    converted_annotations = []
    
    print("Converting annotations to new class labels...")
    
    for annotation_path in annotation_paths:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in CLASS_MAPPING:
                obj.find('name').text = CLASS_MAPPING[name]
        
        converted_annotations.append(annotation_path)
    
    print(f"Converted {len(annotation_paths)} annotations.")
    return converted_annotations
