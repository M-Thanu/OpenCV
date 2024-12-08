import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Define the model class
class FingerRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FingerRecognitionModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),  # Adjust input size if image size differs
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)

# Load the trained model
model_path = "finger_recognition_model.pth"
model = FingerRecognitionModel(num_classes=12)  # Adjust based on your number of classes
model.load_state_dict(torch.load(model_path))
model.eval()

# Define transformation for input images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Path to a test image
test_image_path = "C://Users//hp//Downloads//2.png"

# Load and preprocess the test image
image = Image.open(test_image_path).convert("RGB")  # Ensure 3-channel image
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Perform prediction
with torch.no_grad():
    output = model(input_tensor)
    predicted_class = torch.argmax(output, dim=1).item()

# Map class index to label
class_labels = ['0L', '0R', '1L', '1R', '2L', '2R', '3L', '3R', '4L', '4R', '5L', '5R']
print(f"Predicted Class: {class_labels[predicted_class]}")


