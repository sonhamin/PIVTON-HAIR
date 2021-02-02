from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, x_data1, x_data2, y_data, transform_face=None, transform_hair=None, transform_others=None):
        self.transform_face = transform_face
        self.transform_hair = transform_hair
        self.transform_others = transform_others

        self.data1 = np.transpose(x_data1, (0, 3, 1, 2))
        self.data2 = np.transpose(x_data2, (0, 3, 1, 2))
        self.targets = np.transpose(y_data, (0, 3, 1, 2))

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        face = self.data1[index]
        hair = self.data2[index]
        y = self.targets[index]

        if self.transform_face:
            seed = np.random.randint(2147483647)
            random.seed(seed)

            face = Image.fromarray(self.data1[index].astype(np.uint8).transpose(1, 2, 0))
            face = self.transform_face(face)
            
            random.seed(seed)
            
            y = Image.fromarray(self.targets[index].astype(np.uint8).transpose(1, 2, 0))      
            y = self.transform_face(y)
        
        if self.transform_hair:
            
            seed = np.random.randint(2147483647)
            random.seed(seed)            

            hair = Image.fromarray(self.data2[index].astype(np.uint8).transpose(1, 2, 0))
            hair = self.transform_hair(hair)

        

        return face, hair, y

    
    
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)    
    
    
    
from torchvision.models import vgg19
from collections import namedtuple

class Vgg19(torch.nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        features = list(vgg19(pretrained = True).features)[:36]
        self.features = nn.ModuleList(features).eval()
        
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {2,7,12,21,30}:
                results.append(x)
        return results    