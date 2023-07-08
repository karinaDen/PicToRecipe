import streamlit as st
from PIL import Image,ImageFilter

import torchvision.transforms as transforms
from torchvision import *
from torch import *


from model import *

# extract content from the file by its absolute path
def extract_file_content(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content.split("\n")[:-1] 

ds_path = './data/food-101'
classes = extract_file_content(ds_path + '/meta/classes.txt')




st.write('''<style>
            body{
            text-align:center;
            background-color:#ACDDDE;

            }

            </style>''', unsafe_allow_html=True)



st.title('PicToRecipe')

transform = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


#loading the model
loaded_densenet201 = Net()
loaded_densenet201.load_state_dict(torch.load('densenet201.pt',map_location=torch.device('cpu')))
loaded_densenet201.eval()

st.text('model loaded using densenet201')




file_type = 'jpg'


uploaded_file = st.file_uploader("Choose a  file",type = file_type)


if uploaded_file != None:

    image = Image.open(uploaded_file)

    image = image.filter(ImageFilter.MedianFilter)

    st.image(image)

    # Convert uploaded image to PyTorch tensor using defined transforms,
    # and generate prediction using loaded model
    predicted_class_index = torch.argmax(loaded_densenet201(transform(image).unsqueeze(0)))
    predicted_class = classes[predicted_class_index]

    # Show predicted class name to user
    st.write('The food in the image is:', predicted_class)

    