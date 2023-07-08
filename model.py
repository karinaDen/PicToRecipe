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

openai.api_key_path = "api_key.txt"



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


# extract content from the file by its absolute path
def extract_file_content(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content.split("\n")[:-1] 


def generate_recipe(num_servings, food, marketplace):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Write a recipe for {food} for {num_servings} people on english and find russian ingredients on {marketplace} with links?",
    temperature=0.3,
    max_tokens=120,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    return response

