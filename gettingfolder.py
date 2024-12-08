import os
import shutil

# Path to your dataset folder
dataset_folder = 'C://Users//hp//Downloads//fingers//test'

# Loop through all files in the dataset folder
for file_name in os.listdir(dataset_folder):
    file_path = os.path.join(dataset_folder, file_name)

    # Ensure we're only processing files
    if os.path.isfile(file_path):
        # Extract the label (e.g., '3R' from '000e7aa6-100b-4c6b-9ff0-e7a8e53e4465_3R.png')
        label = file_name.split('_')[-1].split('.')[0]

        # Create a folder for the label if it doesn't exist
        label_folder = os.path.join(dataset_folder, label)
        os.makedirs(label_folder, exist_ok=True)

        # Move the file into the corresponding folder
        shutil.move(file_path, os.path.join(label_folder, file_name))

print("Dataset organization complete!")

