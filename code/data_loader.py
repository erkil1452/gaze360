import torch.utils.data as data
from PIL import Image
import os
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
import re
import glob
import random
import cv2
import torch.nn as nn
import math
import random
import scipy.io as sio

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(source_path,file_name):
    images = []
    print(file_name)
    with open(file_name, 'r') as f:
        for line in f:
            line = line[:-1]
            line = line.replace("\t", " ")
            line = line.replace("  ", " ")
            split_lines = line.split(" ")
            if(len(split_lines)>3):
                frame_number = int(split_lines[0].split('/')[-1][:-4])
                lists_sources = []
                for j in range(-3,4):
                    name_frame = '/'.join(split_lines[0].split('/')[:-1]+['%0.6d.jpg'%(frame_number+j)])
                    name = '{0}/{1}'.format(source_path, name_frame)
                    lists_sources.append(name)

                gaze = np.zeros((3))
           
                gaze[0] = float(split_lines[1])
                gaze[1] = float(split_lines[2])
                gaze[2] = float(split_lines[3])
                item = (lists_sources,gaze)
                images.append(item)
    return images


def default_loader(path):
    try:
        im = Image.open(path).convert('RGB')
        return im
    except OSError:
        print(path)
        return Image.new("RGB", (512, 512), "white")




class ImagerLoader(data.Dataset):
    def __init__(self, source_path,file_name,
                transform=None, target_transform=None, loader=default_loader):

        imgs = make_dataset(source_path,file_name)

        self.source_path = source_path
        self.file_name = file_name

        self.imgs = imgs
        self.transform = transform
        self.target_transform = transform
        self.loader = loader


    def __getitem__(self, index):
        path_source,gaze = self.imgs[index]


        gaze_float = torch.Tensor(gaze)
        gaze_float = torch.FloatTensor(gaze_float)
        normalized_gaze = nn.functional.normalize(gaze_float.view(1,3)).view(3)

        source_video = torch.FloatTensor(7,3,224,224)
        for i,frame_path in enumerate(path_source):
            source_video[i,...] = self.transform(self.loader(frame_path))

        source_video = source_video.view(21,224,224)

        spherical_vector = torch.FloatTensor(2)
        spherical_vector[0] = math.atan2(normalized_gaze[0],-normalized_gaze[2])
        spherical_vector[1] = math.asin(normalized_gaze[1])
        return source_video,spherical_vector


        
    def __len__(self):
        return len(self.imgs)
