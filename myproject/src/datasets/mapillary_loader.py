import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import logging

class MapillaryDataset(Dataset):
    def __init__(self, data_dir, image_size=(512, 1024), transform=None, is_testing=False):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transform
        self.is_testing = is_testing  # Flag to indicate testing mode (no labels)

        # List images
        self.images = [f for f in os.listdir(os.path.join(data_dir, "images")) if f.endswith((".jpg", ".png"))]

        # List labels (if not in testing mode)
        if not self.is_testing:
            self.labels = [f for f in os.listdir(os.path.join(data_dir, "labels")) if f.endswith(".png")]
            if len(self.images) != len(self.labels):
                raise ValueError("Number of images and labels do not match.")
        else:
            self.labels = None  # No labels in testing mode

        # Load label configuration
        config_path = os.path.join(data_dir, "config_v2.0.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        with open(config_path) as config_file:
            self.config = json.load(config_file)
        self.labels_info = self.config.get('labels', [])
        if not self.labels_info:
            raise ValueError("No labels found in config file.")

        # Define road-related labels (areas where a car can drive)
        self.road_related_labels = [
            "construction--flat--road",  # Major road
            "construction--flat--crosswalk-plain",  # Crosswalk
            "construction--flat--bike-lane",  # Bike lane (often drivable)
            "construction--flat--service-lane",  # Service lane
            "marking--discrete--crosswalk-zebra",  # Zebra crossing
            "marking--discrete--arrow--left",  # Arrows on the road
            "marking--discrete--arrow--right",
            "marking--discrete--arrow--straight",
            "marking--discrete--stop-line",  # Stop lines
            "marking--continuous--dashed",  # Lane markings
            "marking--continuous--solid",
            "construction--flat--driveway",  # Driveways
            "construction--flat--parking",  # Parking areas
            "construction--flat--parking-aisle",  # Parking aisles
            "construction--flat--pedestrian-area",  # Pedestrian areas (sometimes drivable)
            "construction--flat--rail-track",  # Rail tracks (sometimes drivable)
            "construction--flat--road-shoulder",  # Road shoulders
        ]

        # Create a mapping from label IDs to class IDs
        self.label_to_class = {label["id"]: 1 for label in self.labels_info if label["name"] in self.road_related_labels}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, "images", self.images[idx])

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load label (if not in testing mode)
        if not self.is_testing:
            label_path = os.path.join(self.data_dir, "labels", self.labels[idx])
            label = Image.open(label_path).convert("L")  # Grayscale label
        else:
            # Create a dummy mask for testing (all zeros)
            label = Image.new("L", image.size, 0)

        # Resize image and label
        resize_transform = transforms.Resize(self.image_size, interpolation=Image.NEAREST)
        image = resize_transform(image)
        label = resize_transform(label)

        # Convert label to numpy array
        label_array = np.array(label)

        # Create binary segmentation mask for all road-related areas
        segmentation_mask = np.zeros_like(label_array, dtype=np.uint8)
        for label_id, class_id in self.label_to_class.items():
            segmentation_mask[label_array == label_id] = class_id

        # Check if the mask is empty (no drivable area)
        if np.sum(segmentation_mask) == 0:
            logging.info(f"No drivable area found in image: {self.images[idx]}")

        # Apply additional transforms (if any)
        if self.transform:
            image = self.transform(image)
            segmentation_mask = torch.tensor(segmentation_mask, dtype=torch.float32)  # Binary mask

        return image, segmentation_mask