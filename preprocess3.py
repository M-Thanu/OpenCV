import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# Custom Dataset Class
class FingerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the dataset folder with subfolders for each class.
            transform (callable, optional): Transform to be applied to each image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))  # List of class folders
        self.image_paths = []
        self.labels = []

        # Populate image paths and labels
        for label, class_folder in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_folder)
            if os.path.isdir(class_path):  # Ensure it's a directory
                for img_file in os.listdir(class_path):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):  # Only image files
                        self.image_paths.append(os.path.join(class_path, img_file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Define Dataset Path
dataset_path = 'C://Users//hp//Downloads//fingers//test'

# Define Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Initialize Dataset and DataLoader
finger_dataset = FingerDataset(dataset_path, transform=transform)
data_loader = DataLoader(finger_dataset, batch_size=32, shuffle=True)

# Print Dataset Info
print(f"Number of samples in dataset: {len(finger_dataset)}")
print(f"Classes: {finger_dataset.classes}")

# Display a Few Samples
def show_samples(dataset, num_samples=5):
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        image, label = dataset[i]
        image = image.permute(1, 2, 0)  # Convert CHW to HWC for display
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image)
        plt.title(f"Label: {finger_dataset.classes[label]}")
        plt.axis('off')
    plt.show()

# Call show_samples to visualize the data
show_samples(finger_dataset, num_samples=5)



