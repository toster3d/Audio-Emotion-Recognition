import torch


# Ustawienia
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
MAX_LENGTH = 48000 * 3  # 3 sekundy przy 16kHz
TARGET_SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "model_outputs"  # Katalog do zapisywania wyników
DATASET_PATH = 'data/nemo_dataset'
