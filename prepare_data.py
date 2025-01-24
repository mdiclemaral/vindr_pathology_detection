import os
import argparse
import pandas as pd
import pydicom
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import gc 

label_mapping = {}

def read_annotations(csv_path: str) -> pd.DataFrame:
    """
    Reads the finding_annotations.csv file.
    """
    return pd.read_csv(csv_path)

def read_breast_level_annotations(csv_path: str) -> pd.DataFrame:
    """
    Reads the breast_level_annotations.csv file to get image metadata including height and width.
    """
    return pd.read_csv(csv_path)[['study_id', 'series_id', 'image_id', 'laterality', 'view_position', 'height', 'width']]

def get_dicom_path(vindr_folder: str, study_id: str, image_id: str) -> Optional[str]:
    """
    Returns the path of the DICOM file given a study_id and image_id.
    """
    study_path = os.path.join(vindr_folder, '1.0.0', 'images', study_id)

    if not os.path.exists(study_path):
        return None  # Study folder missing
    
    dicom_path = os.path.join(study_path, f"{image_id}.dicom")

    if not os.path.exists(dicom_path):
        return None  # DICOM file missing
    return dicom_path


def load_dicom(dicom_path: str, output_d, set_name: str) -> None:
    """Loads a DICOM file and converts it to a NumPy array and jpeg."""

    output_dir = os.path.join(output_d, set_name, "images")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if dicom_path and os.path.exists(dicom_path):
        image_id = dicom_path.split("/")[-1].split(".")[0]
        dicom = pydicom.dcmread(dicom_path)
        image_array = dicom.pixel_array.astype(float)
        rescaled_image = (np.maximum(image_array, 0) / image_array.max()) * 255.0 # float pixels
        image = np.uint8(rescaled_image)
        image = Image.fromarray(image)

        output_file = os.path.join(output_dir, f"{image_id}.jpg")
        image.save(output_file) # save image as jpeg

        del image, rescaled_image, dicom, image_array, dicom_path, output_file
        gc.collect()


def get_image_dicom(vindr_folder: str, annotations: pd.DataFrame, output_d: str, set_name: str) -> List[str]:
    """
    Retrieves the DICOM images corresponding to the annotations.
    """

    dicom_name_list = []
    for _, row in annotations.iterrows():
        study_id, image_id = row['study_id'], row['image_id']
        dicom_path = get_dicom_path(vindr_folder, study_id, image_id)
        if dicom_path: 
            dicom_name_list.append(image_id)
            load_dicom(dicom_path, output_d, set_name) 

    return dicom_name_list

def expand_labels(annotations: pd.DataFrame) -> pd.DataFrame:
    """
    Expands rows where 'finding_categories' contains multiple labels 
    into separate rows, keeping other columns the same.
    """
    annotations = annotations.copy()

    # Ensure the column is treated as a list if it's not already
    annotations['finding_categories'] = annotations['finding_categories'].apply(
        lambda x: x if isinstance(x, list) else eval(x) if isinstance(x, str) and x.startswith("[") else [x]
    )

    # Explode labels into multiple rows
    annotations = annotations.explode('finding_categories').reset_index(drop=True)

    return annotations
def split_train_test(annotations: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the annotations into train, validation, and test sets while balancing labels in validation and test sets.
    """
    # Expand labels before splitting
    annotations = expand_labels(annotations)

    # Split into training and test
    train_df = annotations[annotations['split'] == 'training']
    test_df = annotations[annotations['split'] == 'test']

    return train_df, test_df

def split_train_val(train_df: pd.DataFrame, val_size: float = 0.01) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the annotations into train and validation sets while balancing labels in the validation set.
    """

    # Stratify test_df to create balanced validation and test sets
    if not train_df.empty:
        train_df, val_df = train_test_split(
            train_df, 
            test_size=val_size, 
            stratify=train_df['finding_categories'],  # Ensures label balance
            random_state=42
        )
    else:
        val_df = pd.DataFrame()  # Empty if no test data

    return train_df, val_df

def write_yaml(labels: List[str], output_file: str, output_dir: str) -> None:

    """
    Writes the labels to a YAML file.
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:

        f.write("names:\n")
        for label in labels:
            f.write(f"  - {label}\n")
        f.write(f"nc:\n")
        f.write(f"  - {len(labels)}\n")
        f.write(f"test: {output_dir}/test/images\n")
        f.write(f"train: {output_dir}/train/images\n")
        f.write(f"val: {output_dir}/val/images\n")

def prepare_training_data_yolo(annotations: pd.DataFrame, metadata: pd.DataFrame, dicom_image_id_list: list, output_d: str, set_name: str) -> pd.DataFrame:  
    """
    Prepares training data with image_id, bounding box coordinates, pathology labels, image height, and width.
    """

    metadata_dict = metadata.set_index('image_id').to_dict(orient='index')
    label_id = 0

    
    output_dir = os.path.join(output_d,  set_name, "labels")
    os.makedirs(output_dir, exist_ok=True)

    for _, row in annotations.iterrows():
        image_id = row['image_id']
        output_file = os.path.join(output_dir, f"{image_id}.txt")

        if image_id in dicom_image_id_list:
            with open(output_file, "w") as f:    
                xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                label = row['finding_categories']
                birads = row['breast_birads'].split(' ')[-1] if isinstance(row['breast_birads'], str) else 0
                img_height = metadata_dict.get(image_id, {}).get('height', 0)
                img_width = metadata_dict.get(image_id, {}).get('width', 0)

                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                if not label in label_mapping: 
                    label_mapping[label] = label_id
                    label_id += 1
                class_id = label_mapping[label]
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
    write_yaml(list(label_mapping.keys()), os.path.join(output_d, "data.yaml"), output_d)

    # Filter annotations to only keep rows where image_id exists in dicom_dict
    filtered_annotations = annotations[annotations['image_id'].isin(dicom_image_id_list)]
    return filtered_annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vindr_folder", type=str, help="Path to the VinDr dataset folder")
    parser.add_argument("new_dataset_path", type=str, help="Path to output YOLO dataset")
    args = parser.parse_args()

    annotations_csv = os.path.join(args.vindr_folder, "1.0.0", "finding_annotations.csv")
    breast_annotations_csv = os.path.join(args.vindr_folder, "1.0.0", "breast-level_annotations.csv")
    
    annotations_df = read_annotations(annotations_csv)
    breast_annotations_df = read_breast_level_annotations(breast_annotations_csv)

    train_df, test_df = split_train_test(annotations_df)
    
    dicom_data_train = get_image_dicom(args.vindr_folder, train_df, args.new_dataset_path, "train")
    train_df = prepare_training_data_yolo(train_df, breast_annotations_df, dicom_data_train, args.new_dataset_path, "train")
    train_df, val_df = split_train_val(train_df)

    dicom_data_test = get_image_dicom(args.vindr_folder, test_df, args.new_dataset_path, "test")
    test_df = prepare_training_data_yolo(test_df, breast_annotations_df, dicom_data_test, args.new_dataset_path, "test")
    
    dicom_data_val = get_image_dicom(args.vindr_folder, val_df, args.new_dataset_path, "val")
    val_df = prepare_training_data_yolo(val_df, breast_annotations_df, dicom_data_val, args.new_dataset_path, "val")
