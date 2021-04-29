import os
from PIL import Image
import torch.nn.functional as F

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import random

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class JBLDataset(Dataset):
    def __init__(self,cont_img_path,style_img_path,img_size, one_by_one=False, set_size=-1, num_gpus=1):
        self.num_gpus = num_gpus
        self.cont_img_path = cont_img_path
        self.style_img_path = style_img_path
        self.img_size = img_size
        self.cont_img_files = self.list_files(self.cont_img_path)
        if cont_img_path == style_img_path:
            self.style_img_files = self.cont_img_files
            self.only_one_dataset = True
        else:
            self.style_img_files = self.list_files(self.style_img_path)
            self.only_one_dataset = False
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size), Image.BICUBIC),
            transforms.ToTensor()
        ])
        self.one_by_one = one_by_one
        self.set_size = set_size

    def __len__(self):
        if self.set_size > 0:
            return self.set_size
        else:
            return len(self.cont_img_files)

    def get_img_with_rnd_security(self, file_list:list, idx:int, rng:tuple):
        while True:
            try:
                img = Image.open(file_list[idx]).convert('RGB')
                break
            except Exception as e:
                print("Img read error: {}, file: {}".format(e, self.cont_img_files[idx]))
                idx = random.randint(*rng)
        return img

    def __getitem__(self,idx):
        if self.one_by_one:
            idx = idx//self.num_gpus # TODO: set denominator according to GPUs
        if self.set_size > 0:
            idx = idx%len(self.cont_img_files)
        cont_img = self.get_img_with_rnd_security(self.cont_img_files, idx, (0, len(self)-1))
        cont_img = self.transform(cont_img)
        low_cont = resize(cont_img,cont_img.shape[-1]//2)
        if not self.only_one_dataset:
            if self.one_by_one:
                style_idx = idx
            else:
                style_idx = random.randint(0,len(self.style_img_files) - 1)
            style_img = self.get_img_with_rnd_security(self.style_img_files, style_idx, (0, len(self.style_img_files)-1))
            style_img = self.transform(style_img)
            low_style = resize(style_img, style_img.shape[-1]//2)
            return low_cont, cont_img, low_style, style_img
        else:
            return low_cont, cont_img

    def list_files(self, in_path, ext_filter = ["jpg", "bmp", "jpeg", "png"]):
        files = []
        for (dirpath, dirnames, filenames) in os.walk(in_path):
            valid_files = list(filter(lambda x:x.lower().split('.')[-1] in ext_filter, filenames))
            files.extend(filenames)
            break
        files = sorted([os.path.join(in_path, x) for x in files])
        return files

