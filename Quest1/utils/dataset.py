# utils/dataset.py

import os
import xml.etree.ElementTree as ET
from torchvision import datasets

class ImageFolderWithXMLBBox(datasets.ImageFolder):
    def __init__(self, root, annot_root, transform=None, target_size=(224, 224)):
        super().__init__(root, transform=transform)
        self.annot_root = annot_root
        self.target_size = target_size

    def __getitem__(self, index):
        # 1. Image & Label Loading (Delegated to parent class)
        image, label = super().__getitem__(index)
        img_path, _ = self.samples[index]

        # 2. BBox Parsing (Delegated to a specialized method)
        bbox = self._get_scaled_bbox(img_path)

        return image, label, bbox

    def _get_scaled_bbox(self, img_path):
        """
        Private method to handle only the Annotation & BBox scaling logic.
        Separating this makes the code easier to debug and more modular.
        """
        # Path Mapping Logic
        rel_path = os.path.relpath(img_path, self.root)
        filename = os.path.splitext(os.path.basename(rel_path))[0]
        folder_name = os.path.dirname(rel_path)
        annot_path = os.path.join(self.annot_root, folder_name, filename)
        
        if not os.path.exists(annot_path) and os.path.exists(annot_path + ".xml"):
            annot_path += ".xml"

        # Default empty bbox
        bbox = [0.0, 0.0, 0.0, 0.0]

        if not os.path.exists(annot_path):
            return bbox

        try:
            tree = ET.parse(annot_path)
            root_xml = tree.getroot()
            
            # Extract Original Image Size
            size_tag = root_xml.find('size')
            orig_w = float(size_tag.find('width').text)
            orig_h = float(size_tag.find('height').text)
            
            # Extract Bounding Box
            obj = root_xml.find('object')
            if obj is not None:
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # Scaling Logic
                new_h, new_w = self.target_size
                bbox = [
                    xmin * (new_w / orig_w), 
                    ymin * (new_h / orig_h),
                    xmax * (new_w / orig_w), 
                    ymax * (new_h / orig_h)
                ]
        except Exception as e:
            print(f"[Warning] Parsing error at {annot_path}: {e}")
            
        return bbox