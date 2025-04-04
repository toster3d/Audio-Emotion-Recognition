import os
from datasets import load_dataset

def download_and_save_dataset():
    # Create data folder if it doesn't exist
    folder_name = 'data'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' has been created.")

    # Download the nEMO dataset from Hugging Face
    print("Downloading nEMO dataset from Hugging Face...")
    dataset = load_dataset("amu-cai/nEMO")
    
    # Save dataset
    save_path = os.path.join(folder_name, 'nemo_dataset')
    dataset.save_to_disk(save_path)
    print(f"Dataset saved to {save_path}")
    
    return dataset

if __name__ == "__main__":
    download_and_save_dataset()
