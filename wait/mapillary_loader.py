import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

class MapillaryDataset(Dataset):
    def __init__(self, data_csv, img_dir, label_dir, transform=None):
        self.data = pd.read_csv(data_csv)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        class_label_name = os.path.join(self.label_dir, self.data.iloc[idx, 1])  # Classification label
        seg_label_name = os.path.join(self.label_dir, self.data.iloc[idx, 2])    # Segmentation label
        
        image = Image.open(img_name).convert("RGB")
        class_label = Image.open(class_label_name)
        seg_label = Image.open(seg_label_name)

        if self.transform:
            image = self.transform(image)

        # Return both classification and segmentation labels
        return image, {'classification': class_label, 'segmentation': seg_label}
