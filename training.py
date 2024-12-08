import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Dataset Path
dataset_folder = 'C://Users//hp//Downloads//fingers//test'  # Update this to your dataset path

# 2. Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128
    transforms.ToTensor(),         # Convert images to PyTorch tensors
    transforms.Normalize([0.5], [0.5])  # Normalize pixel values to range [-1, 1]
])

# 3. Load Dataset
finger_dataset = datasets.ImageFolder(root=dataset_folder, transform=transform)

# Number of classes in the dataset (e.g., 0L, 1L, ..., 5R)
num_classes = len(finger_dataset.classes)
print("Classes found:", finger_dataset.classes)
print(f"Total images in dataset: {len(finger_dataset)}")

# 4. Create DataLoader
train_loader = DataLoader(finger_dataset, batch_size=32, shuffle=True)

# 5. Define Model
class FingerNet(nn.Module):
    def __init__(self, num_classes):
        super(FingerNet, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),  # Flattened feature map size
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

model = FingerNet(num_classes=num_classes).to(device)

# 6. Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Training Loop
num_epochs = 10  # Adjust the number of epochs as needed

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 8. Save Model
torch.save(model.state_dict(), 'finger_recognition_model.pth')
print("Model saved as 'finger_recognition_model.pth'")
