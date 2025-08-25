import torch
from torchvision import transforms
import os
import cv2
import glob
import re
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

provNum, alphaNum, adNum = 38, 25, 35
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def multi_glob(pattern_base):
    return glob.glob(os.path.join(pattern_base, "*"))     

def collect_image_paths(base_folder, ext="*.jpg"):
    image_paths = []
    for folder in glob.glob(os.path.join(base_folder, "*/")):
        image_paths.extend(glob.glob(os.path.join(folder, ext)))
    return image_paths

def get_imgs_resolutions(path):
    s = set()
    for i in collect_image_paths(path):
        s.add(Image.open(i).size)
    return s

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
            raise ValueError("ERROR_01")
        return GenerateDataset(self.train_paths, self.train_labels, self.transform, split_data=False)
    
    def get_test_dataset(self):
        if not self.split_data:
            raise ValueError("ERROR_02")
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
    
# boxes operations and creations 
def get_coordinates(coord_string):
    info = str(coord_string.split("\\")).split("-")
    return info[3]

def get_plate_number(coord_string):
    info = str(coord_string.split("\\")).split("-")
    plate = ''
    number = info[4].split("_")
    plate += provinces[int(number[0])]
    for ind in range(1, len(number)):
        plate += ads[int(number[ind])]
    return plate

def parse_coordinates(coord_string):
    numbers = re.findall(r'\d+', coord_string)
    coords = [int(x) for x in numbers]
    
    boxes = []
    for i in range(0, len(coords), 4):
        if i + 3 < len(coords):
            x1, y1, x2, y2 = coords[i:i+4]
            boxes.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'width': abs(x2 - x1),
                'height': abs(y2 - y1)
            })
    
    return boxes

def draw_boxes(image_path, boxes):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for box in boxes:
        x1, y1 = box['x1'], box['y1']
        x2, y2 = box['x2'], box['y2']
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    plt.imshow(img)
    plt.axis("off")
    plt.show()

boxes = [
    {'x1': 405, 'y1': 571, 'x2': 235, 'y2': 574, 'width': 170, 'height': 3},
    {'x1': 231, 'y1': 523, 'x2': 403, 'y2': 522, 'width': 172, 'height': 1}
]