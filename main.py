import os
import read_data as rd
import train_yolo as t_yolo
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Example usage
vindr_folder = "vindr"
annotations_csv = os.path.join(vindr_folder, "1.0.0", "finding_annotations.csv")
breast_annotations_csv = os.path.join(vindr_folder, "1.0.0", "breast-level_annotations.csv")
annotations_df = rd.read_annotations(annotations_csv)
breast_annotations_df = rd.read_breast_level_annotations(breast_annotations_csv)
train_df, test_df = rd.split_train_test(annotations_df)
dicom_data_train = rd.get_image_dicom(vindr_folder, train_df)
dicom_data_test = rd.get_image_dicom(vindr_folder, test_df)
train_data = rd.prepare_training_data(train_df, breast_annotations_df,  dicom_data_train)
test_data = rd.prepare_training_data(test_df, breast_annotations_df, dicom_data_test)

# train_data = test_data
# print(train_data)
# print(len(train_data))

transform = t_yolo.transforms.Compose([transforms.ToTensor(), transforms.Resize((640, 640))])
train_dataset = t_yolo.YOLODataset(train_data, dicom_data_train, transform=transform)
# print(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
trained_model = t_yolo.train_yolo(train_loader)
