import os
import logging
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Class mappings
SURFACE_CLASSES = ['asphalt', 'concrete', 'paving_stones', 'sett', 'unpaved']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(SURFACE_CLASSES)}

class StreetSurfaceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            width, height = img.size

            img = img.crop((0.2 * width, 0.4 * height, 0.8 * width, height))

            if self.transform:
                img = self.transform(img)

            label = CLASS_TO_IDX[self.labels[idx]]
            return img, label

        except Exception as e:
            logging.error(f"Error processing {self.image_paths[idx]}: {e}")
            return None  # Signal failure


class StreetSurfaceVis:
    name = "StreetSurfaceVis"
    dims = (3, 384, 384)
    has_test_dataset = True

    def __init__(self, batch_size=128, num_workers=4, data_root="./data"):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self._init_transforms()

    def _init_transforms(self):
        self.transform_train = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        self.transform_val = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    @property
    def num_classes(self):
        return len(SURFACE_CLASSES)

    def prepare_data(self):
        csv_path = os.path.join(self.data_root, "streetSurfaceVis_v1_0.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found.")

    def setup(self, val_size=0.2, test_size=0.2, random_state=42):
        csv_path = os.path.join(self.data_root, "streetSurfaceVis_v1_0.csv")
        df = pd.read_csv(csv_path)

        # Drop original train/test split and re-split using stratification
        df = df[df["surface_type"].isin(SURFACE_CLASSES)]  # ensure valid classes

        # First, split off the test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["surface_type"],
            random_state=random_state
        )

        # Then split the remaining into train and validation
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            stratify=train_val_df["surface_type"],
            random_state=random_state
        )

        logging.info("Train Set Class Distribution:\n%s", train_df["surface_type"].value_counts())
        logging.info("Validation Set Class Distribution:\n%s", val_df["surface_type"].value_counts())
        logging.info("Test Set Class Distribution:\n%s", test_df["surface_type"].value_counts())

        self.train_set = self._create_dataset(train_df, self.transform_train)
        self.val_set = self._create_dataset(val_df, self.transform_val)
        self.test_set = self._create_dataset(test_df, self.transform_val)

    def _create_dataset(self, df, transform):
        image_paths = [os.path.join(self.data_root, "s_1024", f"{id}.jpg") for id in df["mapillary_image_id"]]
        labels = df["surface_type"].tolist()
        return StreetSurfaceDataset(image_paths=image_paths, labels=labels, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
