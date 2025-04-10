import os
from datasets import load_dataset
from config import DATASET_PATH  # Importowanie ścieżki do zbioru danych z konfiguracji

def download_and_save_dataset():
    # Utwórz folder danych, jeśli nie istnieje
    folder_name = os.path.dirname(DATASET_PATH)  # Użyj ścieżki z konfiguracji
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' został utworzony.")

    # Pobierz zbiór danych nEMO z Hugging Face
    print("Pobieranie zbioru danych nEMO z Hugging Face...")
    dataset = load_dataset("amu-cai/nEMO")
    
    # Zapisz zbiór danych
    dataset.save_to_disk(DATASET_PATH)  # Zapisz w ścieżce z konfiguracji
    print(f"Zbiór danych zapisany w {DATASET_PATH}")
    
    return dataset

if __name__ == "__main__":
    download_and_save_dataset()