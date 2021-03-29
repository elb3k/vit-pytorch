import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

import torchvision.transforms.functional as F


preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def getK(arr, k=16):

    out = []
    ratio = len(arr)/k

    for i in range(k):
        out.append(arr[int(i*ratio)])
    return out


def read_video(video, frames=16):

    imgs = []
    cap = cv2.VideoCapture(video)
    while True:
        status, img = cap.read()
        if not status:
            break
            
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    return getK(imgs, frames)
