import pandas as pd
from torch.utils import data
import numpy as np
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode

        self.__degrade()

        print("> Found %d images..." % (len(self.img_name)))

    
    def __degrade(self):
        path = os.path.join(self.root, f'{self.img_name[0]}.jpeg')
        img = Image.open(path)
        label = self.label[0]
        return self.__transform(img), label


    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __augmentation(self, img):
        augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((-10, 10)),
        ])

        return augmentation(img)

    def __transform(self, img):
        new_size = min(img.size)

    
        script = transforms.Compose([
            transforms.CenterCrop((new_size, new_size)),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        img = script(img)
        return img

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]                                                                                            
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        path = os.path.join(self.root, f'{self.img_name[index]}.jpeg')
        img = Image.open(path)

        label = self.label[index]

        try:
            img = self.__transform(img)
            if self.mode == 'train':
                img = self.__augmentation(img)
            return  img, label
        except :
            print(f'failed to read an image {self.img_name[index]}')
            return self.__degrade()