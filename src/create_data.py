import os

from datasets import load_dataset

from src.config import DATASET_PATH  # Ścieżka do zbioru danych z konfiguracji


def download_and_save_dataset():
    # Tworzenie folderu danych, jeśli nie istnieje
    folder_name = os.path.dirname(DATASET_PATH)  # Ścieżka z konfiguracji
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' został utworzony.")

    # Proces pobierania zbioru danych nEMO z Hugging Face
    print("Pobieranie zbioru danych nEMO z Hugging Face...")
    dataset = load_dataset("amu-cai/nEMO")

    # Zapis zbioru danych w określonej ścieżce
    dataset.save_to_disk(DATASET_PATH)  # type: ignore
    print(f"Zbiór danych zapisany w {DATASET_PATH}")

    return dataset
