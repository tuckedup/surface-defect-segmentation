import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List
import json

class BaseDefectDataset(Dataset):
    def __init__(self, root: str, source_name: str, transforms=None):
        self.root = root
        self.source_name = source_name
        self.transforms = transforms
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Dataset root {self.root} for {self.source_name} not found.")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError

class NEUDETDataset(BaseDefectDataset):
    """NEU-DET: 6 classes [crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches]"""
    def __init__(self, root: str, transforms=None):
        super().__init__(root, "NEU-DET", transforms)
        self.img_dir = os.path.join(self.root, "IMAGES")
        self.anno_dir = os.path.join(self.root, "ANNOTATIONS")
        if not os.path.exists(self.img_dir):
             raise FileNotFoundError(f"NEU-DET image directory {self.img_dir} not found.")
        self.files = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # In a real scenario, parse XML here. 
        # For this skeleton, we assume masks are pre-generated or simplified.
        # Returning a dummy mask for the skeleton.
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Label is encoded in filename prefix for NEU
        label = fname.split('_')[0] 
        
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask, self.source_name, label

class DAGMDataset(BaseDefectDataset):
    """DAGM 2007: 10 classes (Class1 to Class10)"""
    def __init__(self, root: str, transforms=None):
        super().__init__(root, "DAGM 2007", transforms)
        self.samples = []
        for i in range(1, 11):
            class_dir = os.path.join(self.root, f"Class{i}")
            if os.path.exists(class_dir):
                for f in os.listdir(class_dir):
                    if f.endswith('.PNG') and not f.endswith('_label.PNG'):
                        self.samples.append((os.path.join(class_dir, f), f"Class{i}"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
        img_path, label = self.samples[idx]
        mask_path = img_path.replace('.PNG', '_label.PNG')
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask, self.source_name, label

class SeverstalDataset(BaseDefectDataset):
    """Severstal: 4 classes [1, 2, 3, 4]"""
    def __init__(self, root: str, transforms=None):
        super().__init__(root, "Severstal", transforms)
        self.img_dir = os.path.join(self.root, "train_images")
        csv_path = os.path.join(self.root, "train.csv")
        if not os.path.exists(self.img_dir) or not os.path.exists(csv_path):
            raise FileNotFoundError("Severstal images or train.csv not found.")
        
        # Simplified: list images. In reality, parse CSV for RLE.
        self.files = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        # Severstal uses class IDs 1-4.
        label = "mixed" # Severstal can have multiple defects per image
        
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask, self.source_name, label

class MVTecADDataset(BaseDefectDataset):
    """MVTec AD: Many objects, 'good' vs specific defect types"""
    def __init__(self, root: str, transforms=None):
        super().__init__(root, "MVTec AD", transforms)
        self.samples = []
        # root/object_name/test/defect_type/xxx.png
        for obj in os.listdir(self.root):
            obj_dir = os.path.join(self.root, obj)
            if os.path.isdir(obj_dir):
                test_dir = os.path.join(obj_dir, "test")
                if os.path.exists(test_dir):
                    for defect in os.listdir(test_dir):
                        if defect == "good": continue
                        defect_dir = os.path.join(test_dir, defect)
                        for f in os.listdir(defect_dir):
                            if f.endswith('.png'):
                                self.samples.append((os.path.join(defect_dir, f), f"{obj}_{defect}"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str, str]:
        img_path, label = self.samples[idx]
        # Mask is in ground_truth/defect/xxx_mask.png
        mask_path = img_path.replace("test", "ground_truth").replace(".png", "_mask.png")
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask, self.source_name, label
