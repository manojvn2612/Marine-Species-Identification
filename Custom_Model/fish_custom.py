import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms  
from torch.utils.data import DataLoader, random_split
import torch_directml as dml
import pandas as pd
import PIL.Image as Image
from torch.utils.tensorboard import SummaryWriter

device = dml.device()
summary_writer = SummaryWriter()

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, 1, 0)
        self.pool = nn.MaxPool2d(2, 2, 0)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 0)
        self.conv3 = nn.Conv2d(32, 64, 5, 1, 0)
        self.conv4 = nn.Conv2d(64, 128, 5, 1, 0)
        self.fc1 = None
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
        x = x.view(x.size(0), -1)
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

class FishImage():
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]  
        label = self.data.iloc[idx, -1]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == '__main__':
    epochs = 20
    model = CNNModel().to(device)
    data = pd.read_csv("fathomnet_images.csv")
    unique_labels = data["label"].unique()
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    data["label"] = data["label"].map(label_to_idx)
    transform = transforms.Compose([transforms.Resize((400, 400)), transforms.ToTensor()])
    dataset = FishImage(data, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train_samples = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train_samples += labels.size(0)
        train_accuracy = correct_train / total_train_samples * 100
        
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val_samples = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val_samples += labels.size(0)
        val_accuracy = correct_val / total_val_samples * 100

        summary_writer.add_scalar('Loss/Train', total_train_loss, epoch)
        summary_writer.add_scalar('Loss/Validation', total_val_loss, epoch)
        summary_writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
        summary_writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        
        print(f"Epoch [{epoch+1}/{epochs}] -> Train Loss: {total_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%  Val Loss: {total_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    summary_writer.flush()
    summary_writer.close()
    torch.save(model.state_dict(), "fish_species_model.pth")
    model.load_state_dict(torch.load("fish_species_model.pth", map_location=device))
    model.eval()
    
    def preprocess(image_path):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
        image = transform(image)
        return image.unsqueeze(0)

    def predict(image_path):
        image = preprocess(image_path).to(device)
        with torch.no_grad():
            outputs = model(image)
            predicted = torch.argmax(outputs, dim=1)
            idx_to_label = {idx: label for label, idx in label_to_idx.items()}
            print(f"Predicted Class: {idx_to_label[predicted.item()]}")
            return idx_to_label[predicted.item()]

    test_images = [
        "fathomnet_data/Lampocteis_cruentiventer/images/Lampocteis_cruentiventer_0.jpg",
        "fathomnet_data/Nanomia/images/Nanomia_0.jpg",
        "fathomnet_data/Bathochordaeus/images/Bathochordaeus_0.jpg"
    ]
    predict(test_images[0])
    predict(test_images[1])
    predict(test_images[2])