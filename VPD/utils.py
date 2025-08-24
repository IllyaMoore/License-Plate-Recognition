import torch
from torchvision import transforms
import os
import glob
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image


def multi_glob(pattern_base):
    return glob.glob(os.path.join(pattern_base, "*"))     

def collect_image_paths(base_folder, ext="*.jpg"):
    image_paths = []
    for folder in glob.glob(os.path.join(base_folder, "*/")):
        image_paths.extend(glob.glob(os.path.join(folder, ext)))
    return image_paths

def create_dataloader(trainds, testds):
    
    trainloader = torch.utils.data.DataLoader(trainds, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testds, batch_size=64, shuffle=False)
    
    return trainloader, testloader
    

class GenerateDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None, split_data=False, test_size=0.2, random_state=42):
        if split_data:
            if labels is not None:
                self.train_paths, self.test_paths, self.train_labels, self.test_labels = train_test_split(
                    paths, labels, test_size=test_size, random_state=random_state
                )
            else:
                self.train_paths, self.test_paths = train_test_split(
                    paths, test_size=test_size, random_state=random_state
                )
                self.train_labels, self.test_labels = None, None
        else:
            self.paths = paths
            self.labels = labels
        
        self.split_data = split_data
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform

    def get_train_dataset(self):
        if not self.split_data:
            raise ValueError("Дані не були розділені. Встановіть split_data=True при ініціалізації")
        return GenerateDataset(self.train_paths, self.train_labels, self.transform, split_data=False)
    
    def get_test_dataset(self):
        if not self.split_data:
            raise ValueError("Дані не були розділені. Встановіть split_data=True при ініціалізації")
        return GenerateDataset(self.test_paths, self.test_labels, self.transform, split_data=False)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        return img
    
    
def transform_sample(pth):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(pth).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image
    
