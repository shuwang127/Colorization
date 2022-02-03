from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import os
from torchvision.transforms.functional import resize
from PIL import Image

datapath = './landscape_images/'

class ColorizeData(Dataset):
    def __init__(self, set_type='all'):
        # Initialize dataset, you may use a second dataset for validation if required
        imgs = []
        for root, _, fs in os.walk(datapath):
            for file in fs:
                if file.endswith('.jpg'):
                    imgpath = os.path.join(root, file)
                    imgs.append(imgpath)
        num_imgs = int(len(imgs))
        num_train = int(0.8 * num_imgs)
        if ('train' == set_type):
            self.imgs = imgs[:num_train]
        elif ('verify' == set_type):
            self.imgs = imgs[num_train:]
        else:
            self.imgs = imgs

        # Use the input transform to convert images to grayscale
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.Normalize((0.5), (0.5))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256,256)),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def __len__(self) -> int:
        # return Length of dataset
        return len(self.imgs)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        imgpath = self.imgs[index]
        img = Image.open(imgpath).convert('RGB')

        if self.input_transform is not None:
            input_img = self.input_transform(img)
        if self.target_transform is not None:
            target_img = self.target_transform(img)

        return (input_img.cuda(), target_img.cuda()) if torch.cuda.is_available() else (input_img, target_img)