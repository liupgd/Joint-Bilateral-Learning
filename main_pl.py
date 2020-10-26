import torch
from torch import nn
from torch.optim import Adam
import torchvision as T
import torchvision.transforms as Transformer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import *
from datasets import *
import VGG

class PLModel(pl.LightningModule):
    def __init__(self):
        super(PLModel, self).__init__()
        self.model = Model()
    
    def prepare_data(self):
        self.train_dataset = JBLDataset("/mnt/DataDisk/public/dataset/COCO/2017/images/test2017")