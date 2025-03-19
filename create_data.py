import os
from datasets import load_dataset

# Function to create a folder
def create_data_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' has been created.")
    else:
        print(f"Folder '{folder_name}' already exists.")

# Function to download data using the datasets library
def download_nemo_dataset(folder_name):
    print("Downloading nEMO dataset...")
    
    # Load the dataset
    dataset = load_dataset("amu-cai/nEMO", split="train")
    
    # Save the dataset to disk in a format that can be loaded later
    dataset_path = os.path.join(folder_name, 'nemo_dataset')
    dataset.save_to_disk(dataset_path)
    
    print(f"Dataset has been downloaded and saved to '{dataset_path}'")
    print(f"Number of audio samples: {len(dataset)}")
    print(f"Available emotions: {dataset.unique('emotion')}")
    
    # Display a sample
    print("\nSample record:")
    print(f"Audio ID: {dataset[0]['file_id']}")
    print(f"Emotion: {dataset[0]['emotion']}")
    print(f"Text: {dataset[0]['raw_text']}")

# Main script part
if __name__ == "__main__":
    folder_name = 'data'
    create_data_folder(folder_name)
    
    # Use datasets library to download and process the dataset
    download_nemo_dataset(folder_name)
