import os
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import itertools
from functools import lru_cache

from datasets import register

import ipdb

@register("coz_folder")
class RealImageFolder(Dataset):
    def __init__(self, root_path, first_k = None):
        self.max_focal_length = 140
        self.data = []
        self.transform = transforms.ToTensor()

        for dirname in os.listdir(root_path):
            dir_path = os.path.join(root_path, dirname)
            file_list = sorted(os.listdir(dir_path), key=lambda x: float(x.replace('.JPG', '')))
            file_list = [os.path.join(dir_path, filename) for filename in file_list]
            # ipdb.set_trace()
            
            if file_list:
                image_combination = list(itertools.combinations(file_list, 2))
                self.data.extend(image_combination)

    @lru_cache(maxsize = 100)
    def load_image(self, img_path):
        return Image.open(img_path).convert('RGB').copy()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        hr_path, lr_path = self.data[idx]
        hr_img = self.load_image(hr_path)
        lr_img = self.load_image(lr_path)

        hr_basename = os.path.basename(hr_path)
        focal_length = self.max_focal_length / float(os.path.splitext(hr_basename)[0])
        focal_length = round(focal_length, 2)
        ret = {
            "focal_length": focal_length,
            "lr_image": self.transform(lr_img),
            "hr_image": self.transform(hr_img),
        }

        return ret

if __name__ == "__main__":
    path = "data/coz_data/train_LR"
    dataset = RealImageFolder(path)
    ipdb.set_trace()
