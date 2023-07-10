import torchvision
import torch
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights
from torch import nn
from PIL import Image
import numpy as np
import openai

openai.api_key_path = 'api_key.txt'

def download_dataset():

    # Download the dataset
    !wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz

    # Unzip the dataset
    !tar -xzf food-101.tar.gz

    # Remove the tar file
    !rm food-101.tar.gz

    # Move the dataset to the data folder
    !mv food-101 da

class Net(nn.Module):
    def __init__(self, num_classes=101):
        super(Net, self).__init__()
        self.net = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.DEFAULT,
                                    progress=True)
        self.net.trainable = False
        self.net.fc = nn.Sequential(nn.Linear(2048, 1024),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.3),
                                    nn.Linear(1024, num_classes))

    def forward(self, x):
        return self.net(x)



# extract content from the file by its absolute path
def extract_file_content(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content.split("\n")[:-1] 


def generate_recipe(num_servings, food):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Write a recipe for" + food + "for" + str(num_servings) + "servings",
    temperature=0.3,
    max_tokens=750,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    
    return response["choices"][0]["text"]

