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

class boxes:
    
    def get_box_coordinates(result):
        
        boxes = result.boxes
        num = 0
        coordinates = {}
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()     
            conf = box.conf[0].item()                 
            cls = int(box.cls[0].item())              
            print(f"Coords: ({x1:.1f}, {y1:.1f}) - ({x2:.1f}, {y2:.1f}) | "
                    f"Class: {cls} | Confidence: {conf:.2f}")
            coordinates[num] = {x1}, {x2}, {y1}, {y2}
            num +=1
        
        return coordinates
    
    def get_test_box(coordinates):
        x1, x2, y1, y2 = coordinates[0]
        x1, x2, y1, y2 = float(list(x1)[0]), float(list(x2)[0]), float(list(y1)[0]), float(list(y2)[0])
        return x1, x2, y1, y2
    
    def cut_plate_numbers(coords, img):
        return img.crop(coords)
        
        