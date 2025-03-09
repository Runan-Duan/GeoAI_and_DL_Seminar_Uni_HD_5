import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MapillaryDataset(Dataset):
    def __init__(self, data_dir, image_size=(512, 1024), transform=None):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform
        self.images = [f for f in os.listdir(os.path.join(data_dir, "images")) if f.endswith((".jpg", ".png"))]
        self.labels = [f for f in os.listdir(os.path.join(data_dir, "labels")) if f.endswith(".png")]

        # Load label configuration
        with open(os.path.join(data_dir, "config_v2.0.json")) as config_file:
            self.config = json.load(config_file)
        self.labels_info = self.config['labels']

        # Define road-related labels and their corresponding IDs
        self.road_related_labels = [
            "construction--flat--road",
            "construction--flat--sidewalk",
            "construction--flat--crosswalk-plain",
            "construction--flat--bike-lane",
            "construction--flat--service-lane",
            "marking--discrete--crosswalk-zebra",
            "marking--discrete--arrow--left",
            "marking--discrete--arrow--right",
            "marking--discrete--arrow--straight",
            "marking--discrete--symbol--bicycle",
            "marking--discrete--stop-line",
            "marking--continuous--dashed",
            "marking--continuous--solid"
        ]

        # Create a mapping from label IDs to class IDs
        self.label_to_class = {}
        self.class_names = ["background"]  # Class 0 is for "no road"
        for label_id, label in enumerate(self.labels_info):
            if label["name"] in self.road_related_labels:
                self.label_to_class[label_id] = len(self.class_names)
                self.class_names.append(label["name"])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, "images", self.images[idx])
        label_path = os.path.join(self.data_dir, "labels", self.labels[idx])

        # Load image and label
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")  # Grayscale label

        # Resize image and label
        resize_transform = transforms.Resize(self.image_size, interpolation=Image.NEAREST)
        image = resize_transform(image)
        label = resize_transform(label)

        # Convert label to numpy array
        label_array = np.array(label)

        # Create multi-class segmentation mask
        segmentation_mask = np.zeros_like(label_array, dtype=np.uint8)
        for label_id, class_id in self.label_to_class.items():
            segmentation_mask[label_array == label_id] = class_id

        # Apply additional transforms (if any)
        if self.transform:
            image = self.transform(image)
            segmentation_mask = torch.tensor(segmentation_mask, dtype=torch.long)

        return image, segmentation_mask