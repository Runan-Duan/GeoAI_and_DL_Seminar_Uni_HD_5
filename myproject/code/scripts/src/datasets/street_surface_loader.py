import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_streetsurfacevis(data_dir):
    """
    Load the StreetSurfaceVis dataset and split it into train and test sets based on the `train` column.
    """
    csv_path = os.path.join(data_dir, "streetSurfaceVis_v1_0.csv")
    try:
        df = pd.read_csv(csv_path)
        train_images, train_labels = [], []
        test_images, test_labels = [], []
        for _, row in df.iterrows():
            image_path = os.path.join(data_dir, "s_1024", f"{row['mapillary_image_id']}.jpg")
            if not os.path.exists(image_path):
                logging.warning(f"Image file not found: {image_path}")
                continue
            label = row["surface_type"]
            if row["train"]:
                train_images.append(image_path)
                train_labels.append(label)
            else:
                test_images.append(image_path)
                test_labels.append(label)
        return train_images, train_labels, test_images, test_labels
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

class SurfaceDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(set(labels))}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        try:
            # Open image and apply recommended cropping
            img = Image.open(image_path).convert("RGB")
            width, height = img.size
            img_cropped = img.crop((0.25 * width, 0.5 * height, 0.75 * width, height))

            # Apply additional transformations (if any)
            if self.transform:
                img_cropped = self.transform(img_cropped)

            label = self.labels[idx]
            return img_cropped, self.label_to_idx[label]
        except Exception as e:
            logging.error(f"Error loading image {image_path}: {e}")
            return None, None

def split_and_save_data(data_dir, processed_dir, val_size=0.2, random_state=42):
    """
    Split the dataset into train, validation, and test sets based on the `train` column.
    Save the splits to the processed directory.
    """
    csv_path = os.path.join(data_dir, "streetSurfaceVis_v1_0.csv")
    try:
        # Load the dataset
        df = pd.read_csv(csv_path)

        # Split into train and test based on the `train` column
        train_df = df[df["train"]]
        test_df = df[~df["train"]]

        # Further split the training data into train and validation sets
        train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state)

        # Save the splits to the processed directory
        os.makedirs(processed_dir, exist_ok=True)
        train_df.to_csv(os.path.join(processed_dir, "street_surface_train.csv"), index=False)
        val_df.to_csv(os.path.join(processed_dir, "street_surface_val.csv"), index=False)
        test_df.to_csv(os.path.join(processed_dir, "street_surface_test.csv"), index=False)

        logging.info(f"Data split and saved to {processed_dir}")
    except Exception as e:
        logging.error(f"Error splitting and saving data: {e}")
        raise