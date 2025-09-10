from ultralytics import YOLO
import easyocr
from PIL import Image
from torchvision import transforms
import os
import cv2
import numpy as np
import glob
import utils
from utils import boxes

try:
    from dotenv import load_dotenv
    
    load_dotenv() 
    
    SAMPLE = os.getenv(r'SAMPLE')
    
    print("SAMPLE_PATH =", SAMPLE)
except ModuleNotFoundError as e:
    print("Module not found:", e)
    
SAMPLE = os.getenv('SAMPLE')

if not SAMPLE:
    SAMPLE = input("img path: ")

MODEL_VPD = r'src\modelVPD\yolov8n.pt'


model = YOLO(MODEL_VPD)
img = Image.open(SAMPLE)
y, x = img.size
img = transforms.Resize((1080, 1920))(img)
result = model(img)


coordinates = boxes.get_box_coordinates(result[0])

if coordinates and len(coordinates) > 0:
    x1, x2, y1, y2 = boxes.get_test_box(coordinates)
else:
    print("No plates found")
    exit()

coords = [x1, y1, x2, y2]
cropped_license_plates = []
cropped_license_plates.append(np.array(boxes.cut_plate_numbers(coords, img)))



reader = easyocr.Reader(['en'])

for index, cropped_plate in enumerate(cropped_license_plates):
    cropped_plate_rgb = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)
    
    ocr_results = reader.readtext(cropped_plate_rgb)
    
    print(f"License plate {index} results:")
    plate_text = " ".join([res[1] for res in ocr_results])
    print(plate_text)
    
boxes.cut_plate_numbers(coords, img)