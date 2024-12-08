import torch
import torch.nn as nn

# Define the model class (same as used during training)
class FingerRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FingerRecognitionModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.conv_layers(x)

model_path = "finger_recognition_model.pth"
model = FingerRecognitionModel(num_classes=12)  # Adjust based on your number of classes

# Load the saved model weights
state_dict = torch.load(model_path)

# Create a new dictionary to store only the matched keys
new_state_dict = {}

# Iterate over the saved state_dict and check if the key exists in the current model's state_dict
for k, v in state_dict.items():
    if k in model.state_dict():
        new_state_dict[k] = v

# Load the new state dictionary into the model
model.load_state_dict(new_state_dict)
model.eval()



