import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from PIL import Image
import torchvision
# torchvision.set_image_backend('accimage')
import os

class ImageNet(Dataset):

    
    def __init__(self, annotations, root_dir, preprocess, train=True):
        """
        ImageNet dataset

        args:
            annotations: for `train=True` annotations file path (annotations of foldername to label id)
                        for `train=False` file path for ground truth of validation data
            root_dir: path to images
            train: True - train/ False - validation
        """

        self.root_dir = root_dir
        self.src = []
        # Load for train
        if train:
            # Read folder to id dictionary
            with open(annotations, "r") as f:
                d = json.load(f)
            
            # List folders and inside folders
            for folder in os.listdir(root_dir):
                label = d[folder]
                for file in os.listdir(f'{root_dir}/{folder}'):
                    self.src.append([f'{folder}/{file}', label])
        # Load for validation
        else:
            # Load array of labels
            with open(annotations, 'r') as f:
                labels = json.load(f)
            # Load sorted files
            for i, file in enumerate(sorted(os.listdir(root_dir))):
                self.src.append([file, labels[i]])
        
        self.preprocess = preprocess

    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):

        file, label = self.src[idx]
        img = Image.open(f'{self.root_dir}/{file}').convert('RGB')
        # Preprocess
        if self.preprocess is not None:
            img = self.preprocess(img)
        
        return img, int(label)