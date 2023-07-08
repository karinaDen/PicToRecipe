import torchvision
from torchvision import datasets, transforms, models
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np
import openai


class Net(nn.Module):
    def __init__(self, num_classes=101):
        super(Net, self).__init__()
        self.net = models.densenet201(pretrained=True,progress=True)
        self.net.trainable = False
        self.net.fc = nn.Sequential(nn.Linear(1000, 512),
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=0.4),
                                    nn.Linear(512, num_classes))

    def forward(self, x):
        return self.net(x)

