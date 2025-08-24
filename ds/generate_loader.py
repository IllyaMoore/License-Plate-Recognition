import sys
import os
current_dir = os.getcwd()
vpd_path = os.path.join(os.path.dirname(current_dir), 'VPD')
sys.path.append(vpd_path)
import glob
import utils
from utils import multi_glob
from utils import collect_image_paths
from utils import GenerateDataset
from utils import create_dataloader

path = r'B:\VLPR\ds\data'

def dataloaders():
    full_dataset = GenerateDataset(collect_image_paths(path), split_data=True, test_size=0.2)

    train_dataset = full_dataset.get_train_dataset()
    test_dataset = full_dataset.get_test_dataset()

    trainloader, testloader = create_dataloader(train_dataset, test_dataset)
    return trainloader, testloader