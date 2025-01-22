import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
import os
import cv2
import numpy as np
from typing import List, Tuple, Dict

class YOLODataset(Dataset):
    def __init__(self, annotations: List[Tuple[str, List[float], List[str], int, int]], image_data: Dict[str, np.ndarray], transform=None):
        self.annotations = annotations
        self.image_data = image_data
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        image_id, bbox, labels, height, width = self.annotations[idx]
        image = self.image_data.get(image_id, None)
        if image is None:
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
        
        if self.transform:
            image = self.transform(image)
        
        return image, bbox, labels

def train_yolo(train_loader, num_epochs=10, model_save_path="yolo_vindr.pt"):
    """Trains a YOLO model using the provided training data loader."""
    model = YOLO("yolov8n.pt")  # Load pre-trained YOLOv8 model
    model.train(data=train_loader, epochs=num_epochs)
    model.save(model_save_path)
    return model

# # Example usage:
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((640, 640))])
train_dataset = YOLODataset(train_data, dicom_data_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
trained_model = train_yolo(train_loader)