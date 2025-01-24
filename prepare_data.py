import os
import argparse
import pandas as pd
import pydicom
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional

label_mapping = {}

def read_annotations(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def read_breast_level_annotations(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)[['study_id', 'series_id', 'image_id', 'laterality', 'view_position', 'height', 'width']]

def get_dicom_path(vindr_folder: str, study_id: str, image_id: str) -> Optional[str]:
    study_path = os.path.join(vindr_folder, '1.0.0', 'images', study_id)
    dicom_path = os.path.join(study_path, f"{image_id}.dicom")
    return dicom_path if os.path.exists(dicom_path) else None

def load_dicom(dicom_path: str, output_d: str, set_name: str) -> Optional[np.ndarray]:
    output_dir = os.path.join(output_d, set_name, "images")
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(dicom_path):
        image_id = os.path.basename(dicom_path).split(".")[0]
        dicom = pydicom.dcmread(dicom_path)
        image_array = dicom.pixel_array.astype(float)
        rescaled_image = (np.maximum(image_array, 0) / image_array.max()) * 255.0
        image = Image.fromarray(np.uint8(rescaled_image))
        image.save(os.path.join(output_dir, f"{image_id}.jpg"))
        return image_array
    return None

def get_image_dicom(vindr_folder: str, annotations: pd.DataFrame, output_d: str, set_name: str) -> Dict[str, Optional[np.ndarray]]:
    return {row['image_id']: load_dicom(get_dicom_path(vindr_folder, row['study_id'], row['image_id']), output_d, set_name)
            for _, row in annotations.iterrows() if get_dicom_path(vindr_folder, row['study_id'], row['image_id'])}

def expand_labels(annotations: pd.DataFrame) -> pd.DataFrame:
    annotations['finding_categories'] = annotations['finding_categories'].apply(
        lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else [x])
    return annotations.explode('finding_categories').reset_index(drop=True)

def split_train_test(annotations: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    annotations = expand_labels(annotations)
    return annotations[annotations['split'] == 'training'], annotations[annotations['split'] == 'test']

def split_train_val(train_df: pd.DataFrame, val_size: float = 0.01) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(train_df, test_size=val_size, stratify=train_df['finding_categories'], random_state=42) if not train_df.empty else (train_df, pd.DataFrame())

def write_yaml(labels: List[str], output_file: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        f.write("names:\n")
        f.writelines([f"  - {label}\n" for label in labels])
        f.write(f"nc:\n  - {len(labels)}\n")
        f.write(f"test: {output_dir}/test/images\ntrain: {output_dir}/train/images\nval: {output_dir}/val/images\n")

def prepare_training_data_yolo(annotations: pd.DataFrame, metadata: pd.DataFrame, dicom_dict: dict, output_d: str, set_name: str) -> pd.DataFrame:
    metadata_dict = metadata.set_index('image_id').to_dict(orient='index')
    output_dir = os.path.join(output_d, set_name, "labels")
    os.makedirs(output_dir, exist_ok=True)
    
    label_id = 0
    for _, row in annotations.iterrows():
        image_id = row['image_id']
        if image_id in dicom_dict:
            with open(os.path.join(output_dir, f"{image_id}.txt"), "w") as f:
                xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
                label = row['finding_categories']
                img_height, img_width = metadata_dict.get(image_id, {}).get('height', 0), metadata_dict.get(image_id, {}).get('width', 0)
                
                x_center, y_center = ((xmin + xmax) / 2) / img_width, ((ymin + ymax) / 2) / img_height
                width, height = (xmax - xmin) / img_width, (ymax - ymin) / img_height
                
                if label not in label_mapping:
                    label_mapping[label] = label_id
                    label_id += 1
                f.write(f"{label_mapping[label]} {x_center} {y_center} {width} {height}\n")
    
    write_yaml(list(label_mapping.keys()), os.path.join(output_d, "data.yaml"), output_d)
    return annotations[annotations['image_id'].isin(dicom_dict)]

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
