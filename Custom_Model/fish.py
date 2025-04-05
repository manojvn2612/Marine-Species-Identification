import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import PIL.Image as Image
import torch_directml as dml
from torch.utils.tensorboard import SummaryWriter

summary_writer = SummaryWriter()
device = dml.device()
print(f"Using device: {device}")

class FishDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, -1]
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Load Dataset
data = pd.read_csv("fathomnet_images.csv")
unique_labels = data["label"].unique()
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
data["label"] = data["label"].map(label_to_idx)
num_classes = len(unique_labels)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = FishDataset(data, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train Model
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss, correct_train, total_train = 0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
    
    train_acc = correct_train / total_train * 100
    summary_writer.add_scalar('Loss/Train', train_loss, epoch)
    summary_writer.add_scalar('Accuracy/Train', train_acc, epoch)

    model.eval()
    val_loss, correct_val, total_val = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)
    
    val_acc = correct_val / total_val * 100
    summary_writer.add_scalar('Loss/Validation', val_loss, epoch)
    summary_writer.add_scalar('Accuracy/Validation', val_acc, epoch)
    
    print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

torch.save(model.state_dict(), "resnet_fish_model.pth")
print("Model saved successfully")

model.load_state_dict(torch.load("resnet_fish_model.pth", map_location=device))
model.eval()
print("Model loaded successfully")

def preprocess(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict(image_path):
    image = preprocess(image_path).to(device)
    with torch.no_grad():
        outputs = model(image)
        predicted = torch.argmax(outputs, dim=1).item()
        print(f"Predicted Class: {idx_to_label[predicted]}")
        return idx_to_label[predicted]

test_images = [
    "fathomnet_data/Lampocteis_cruentiventer/images/Lampocteis_cruentiventer_131.jpg",
    "fathomnet_data/Acanthascinae/images/Acanthascinae_145.jpg",
    "fathomnet_data/Bathochordaeus/images/Bathochordaeus_31.jpg"
]

for img in test_images:
    if os.path.exists(img):
        predict(img)
    else:
        print(f"Test image not found: {img}")
