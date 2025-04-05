import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch_directml as dml
import pandas as pd
import PIL.Image as Image
import streamlit as st
import google.generativeai as genai
from ultralytics import YOLO
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY") #Add your API key

# Initialize Services
genai.configure(api_key=GEMINI_API_KEY)
text_model = genai.GenerativeModel('gemini-1.5-flash')

device = dml.device()
print(f"Using device: {device}")

# Load Dataset (for label mapping)
data = pd.read_csv("fathomnet_images2.csv")
unique_labels = ['Acanthascinae','Bathochordaeus' 'Lampocteis Cruentiventer' 'Nanomia']
label_to_idx = {'Acanthascinae':0,'Bathochordaeus': 1, 'Lampocteis Cruentiventer': 2, 'Nanomia': 3}
idx_to_label = {0:'Acanthascinae',1: 'Bathochordaeus', 2: 'Lampocteis Cruentiventer', 3: 'Nanomia'}
num_classes = 4

class CNNModel(nn.Module):
    def __init__(self, input_size=400):
        super(CNNModel, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0)

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size, input_size)
            x = self.conv1(dummy_input)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = self.pool(x)
            x = self.conv4(x)
            x = self.pool(x)
            flattened_size = x.view(1, -1).size(1)

        # Fully Connected Layers (Dynamically Initialized)
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.fc4 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 512).to(device)
            self.add_module("fc1", self.fc1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.softmax(x, dim=1)
        return x

#Loading custom model
model_custom = CNNModel().to(device)
model_custom.load_state_dict(torch.load("custom_fish_model1.pth", map_location=device))
model_custom.eval()
print("Model loaded successfully")

# Load resnet Model
model_resnet = models.resnet18(pretrained=False)  # Important: Do not load pretrained during inference loading.
model_resnet.fc = nn.Linear(model_resnet.fc.in_features, num_classes)
model_resnet.load_state_dict(torch.load("resnet_fish_model1.pth", map_location=device))
model_resnet.to(device)
model_resnet.eval()
print("Model loaded successfully")

# Load Yolov11 Model
model_yolo=YOLO("best_final_yolo.pt")

def predict_yolo(image):
    results = model_yolo(image) 
    for output in results:
            predicted=int(output.boxes.cls[0])
            return idx_to_label[predicted]


# Using gemini to get information about the species
def find_info(name):
    prompt = f"""Give Details about {name} marine species, 
    it should contain details like their habitat and life cycle,
    depth at which it is found,
    what do they cosume,
    and any extra information about {name}"""
    return text_model.generate_content(prompt).text

# Image preprocessing for resnet model
def preprocess_resnet(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

# Predict function for resnet model
def predict_resnet(image):
    image = preprocess_resnet(image).to(device)
    with torch.no_grad():
        outputs = model_resnet(image)
        predicted = torch.argmax(outputs, dim=1).item()
        return idx_to_label[predicted]

# Preprocessing images for custom model
def preprocess_custom(image):
        transform = transforms.Compose([transforms.Resize((400, 400)), transforms.ToTensor()])
        image = transform(image)
        return image.unsqueeze(0)

# Predict function for resnet model
def predict_custom(image):
        image = preprocess_custom(image).to(device)
        with torch.no_grad():
            outputs = model_custom(image)
            predicted = torch.argmax(outputs, dim=1)
            idx_to_label = {idx: label for label, idx in label_to_idx.items()}
            return idx_to_label[predicted.item()]

st.set_page_config(layout="wide") 
st.title("Marine Species Classification")

tab1, tab2, tab3 = st.tabs(["Custom CNN Model", "ResNet18 Model", "YOLO-V-11 Model"])

with tab1:
    col1,col2=st.columns(2)
    with col1:
        try:
            st.header("Custom CNN Model")
            st.write("This model is a custom Convolutional Neural Network designed for marine species classification.")
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"],key='custom_upload')
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_container_width=True)
                if st.button("Predict",key='custom_pred'):
                    prediction = predict_custom(image)
                    # information=find_info(prediction)
                    with col2:
                        st.header(f"Predicted Class: {prediction}")
                        information=find_info(prediction)
                        st.write(information)
        except:
            with col2:
                st.header("Not Able to Recognize😔")

with tab2:
    col3,col4=st.columns(2)
    with col3:
        try:
            st.header("ResNet18 Model")
            st.write("This model is a ResNet18 model, a pre-built model that has been fine-tuned for marine species classification.")
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"],key='resnet_upload')
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_container_width=True)
                if st.button("Predict",key='resnet_pred'):
                    prediction = predict_resnet(image)
                    with col4:
                        st.header(f"Predicted Class: {prediction}")
                        information=find_info(prediction)
                        st.write(information)
        except:
            with col4:
                st.header("Not Able to Recognize😔")


with tab3:
    col5,col6=st.columns(2)
    with col5:
        try:
            st.header("YOLO-V-11 Model")
            st.write("This model is a YOLO version 11 model, a pre-built model that has been fine-tuned for marine species classification.")
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"],key='yolo_upload')
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_container_width=True)
                if st.button("Predict",key='yolo_pred'):
                    prediction = predict_yolo(image)
                    # information=find_info(prediction)
                    with col6:
                        st.header(f"Predicted Class: {prediction}")
                        information=find_info(prediction)
                        st.write(information)
        except:
            with col6:
                st.header("Not Able to Recognize😔")