import streamlit as st
from PIL import Image, ImageFilter
from torchvision import *
from torchvision import transforms
from torch import *
from model import *


classes = extract_file_content('classes.txt')

logo = Image.open('logo-no-background.png')
col1, col2, col3 = st.columns([1.25, 0.5, 1.25])
col2.image(logo, use_column_width=True)
st.markdown("<h1 style='font-size: 180%; text-align: center; color: #fc9551;'>Welcome to PicToRecipe!</h1>", unsafe_allow_html=True)

transform = transforms.Compose([
    transforms.Resize(240),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# loading the model
loaded_densenet201 = Net()
loaded_densenet201.load_state_dict(torch.load('densenet201.pt', map_location=torch.device('cpu')))
loaded_densenet201.eval()
st.text('densenet201 model loaded')


file_type = 'jpg'
uploaded_file = st.file_uploader("Choose a file", type=file_type)

if uploaded_file != None:

    image = Image.open(uploaded_file)

    image = image.filter(ImageFilter.MedianFilter)

    st.image(image)

    # Convert uploaded image to PyTorch tensor using defined transforms,
    # and generate prediction using a loaded model
    predicted_class_index = torch.argmax(loaded_densenet201(transform(image).unsqueeze(0)))
    predicted_class = classes[predicted_class_index]
    food = predicted_class.replace("_", " ")

    # Show the predicted class name to a user
    st.write('The food in the image is:', food)

    num_servings = st.number_input('How many servings do you need?', min_value=1, max_value=10, value=4, step=1)

    st.write(generate_recipe(num_servings, food))
    
