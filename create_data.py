import os
from datasets import load_dataset

# Create data folder if it doesn't exist
folder_name = 'data'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"Folder '{folder_name}' has been created.")
else:
    print(f"Folder '{folder_name}' already exists.")

# Download the complete dataset with a simple call
print("Downloading nEMO dataset...")
dataset = load_dataset("amu-cai/nEMO")

# Save the dataset to disk
dataset_path = os.path.join(folder_name, 'nemo_dataset')
dataset.save_to_disk(dataset_path)

