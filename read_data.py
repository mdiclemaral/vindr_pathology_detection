import os
import pandas as pd
import pydicom
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional

from PIL import Image
import numpy as np

def read_annotations(csv_path: str) -> pd.DataFrame:
    """Reads the finding_annotations.csv file."""
    df = pd.read_csv(csv_path)
    return df

def read_breast_level_annotations(csv_path: str) -> pd.DataFrame:
    """Reads the breast_level_annotations.csv file to get image metadata including height and width."""
    df = pd.read_csv(csv_path)
    return df[['study_id', 'series_id', 'image_id', 'laterality', 'view_position', 'height', 'width']]

def get_dicom_path(vindr_folder: str, study_id: str, image_id: str) -> Optional[str]:
    """Returns the path of the DICOM file given a study_id and image_id."""
    study_path = os.path.join(vindr_folder, '1.0.0', 'images', study_id)

    if not os.path.exists(study_path):
        return None  # Study folder missing
    
    dicom_path = os.path.join(study_path, f"{image_id}.dicom")

    if not os.path.exists(dicom_path):
        return None  # DICOM file missing
    return dicom_path

def load_dicom(dicom_path: str) -> Optional[np.ndarray]:
    os.makedirs('im_jpg', exist_ok=True)
    """Loads a DICOM file and converts it to a NumPy array."""
    if dicom_path and os.path.exists(dicom_path):

        image_id = dicom_path.split("/")[-1].split(".")[0]
        dicom = pydicom.dcmread(dicom_path)
        image_array = dicom.pixel_array
        image = Image.fromarray(image_array)
        image.save(f"im_jpg/{image_id}.jpg")
        return image
    return None

def get_image_dicom(vindr_folder: str, annotations: pd.DataFrame) -> Dict[str, Optional[np.ndarray]]:
    """Retrieves the DICOM images corresponding to the annotations."""
    dicom_dict = {}
    for _, row in annotations.iterrows():
        study_id, image_id = row['study_id'], row['image_id']
        dicom_path = get_dicom_path(vindr_folder, study_id, image_id)
        if dicom_path: 
            dicom_dict[image_id] = load_dicom(dicom_path) if dicom_path else None
            # print(f"Study ID: {study_id}, Image ID: {image_id}, DICOM Path: {dicom_path}")

    return dicom_dict

def split_train_test(annotations: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the annotations into train and test sets based on the 'split' column."""
    train_df = annotations[annotations['split'] == 'training']
    test_df = annotations[annotations['split'] == 'test']
    return train_df, test_df

def prepare_training_data(annotations: pd.DataFrame, metadata: pd.DataFrame, dicom_dict: dict) -> List[Tuple[str, List[float], List[str], int, int]]:
    """Prepares training data with image_id, bounding box coordinates, pathology labels, image height, and width."""
    training_data = {}
    metadata_dict = metadata.set_index('image_id').to_dict(orient='index')
    for _, row in annotations.iterrows():
        image_id = row['image_id']
        if image_id in dicom_dict:
            bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            labels = eval(row['finding_categories']) if isinstance(row['finding_categories'], str) else []
            birads = row['breast_birads'].split(' ')[-1] if isinstance(row['breast_birads'], str) else 0
            height = metadata_dict.get(image_id, {}).get('height', 0)
            width = metadata_dict.get(image_id, {}).get('width', 0)

            training_data[image_id] = [bbox, labels, birads, (height, width)]
    print(training_data)
    
    return training_data

def prepare_training_data_yolo(annotations: pd.DataFrame, metadata: pd.DataFrame, dicom_dict: dict, output_dir: str, output_name: str) -> None:
    """Prepares training data with image_id, bounding box coordinates, pathology labels, image height, and width."""
    training_data = {}
    metadata_dict = metadata.set_index('image_id').to_dict(orient='index')

    label_mapping = {}
    label_id = 0
    

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, output_name) 

    with open(output_file, "w") as f:
        
        for _, row in annotations.iterrows():
            image_id = row['image_id']
            # if image_id in dicom_dict:
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            labels = eval(row['finding_categories']) if isinstance(row['finding_categories'], str) else []
            birads = row['breast_birads'].split(' ')[-1] if isinstance(row['breast_birads'], str) else 0
            img_height = metadata_dict.get(image_id, {}).get('height', 0)
            img_width = metadata_dict.get(image_id, {}).get('width', 0)

            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            for label in labels:
                if not label in label_mapping: 
                    label_mapping[label] = label_id
                    label_id += 1
                class_id = label_mapping[label]
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        label_map_file = os.path.join(output_dir, "label_map.txt")
        with open(label_map_file, "w") as f:
            for label, class_id in label_mapping.items():
                f.write(f"{label} {class_id}\n")
            
    
# Example usage
vindr_folder = "vindr"
annotations_csv = os.path.join(vindr_folder, "1.0.0", "finding_annotations.csv")
breast_annotations_csv = os.path.join(vindr_folder, "1.0.0", "breast-level_annotations.csv")
annotations_df = read_annotations(annotations_csv)
breast_annotations_df = read_breast_level_annotations(breast_annotations_csv)
train_df, test_df = split_train_test(annotations_df)
dicom_data_train = get_image_dicom(vindr_folder, train_df)
dicom_data_test = get_image_dicom(vindr_folder, test_df)
train_data = prepare_training_data_yolo(train_df, breast_annotations_df, dicom_data_train, "data_yolo", "train.txt")
test_data = prepare_training_data_yolo(test_df, breast_annotations_df, dicom_data_test, "data_yolo", "test.txt")

# print(dicom_data_train)
# print(test_data)