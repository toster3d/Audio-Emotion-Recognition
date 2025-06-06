import os
from datetime import datetime

import torch

# Ścieżki projektowe
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ustawienia urządzenia
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nazwy klas emocji
CLASS_NAMES = ["anger", "fear", "happiness", "neutral", "sadness", "surprised"]
NUM_CLASSES = len(CLASS_NAMES)

# Ustawienia walidacji krzyżowej
CV_FOLDS = 5
SEED = 42
TEST_SPLIT = 0.2

# Konfiguracja batcha
BATCH_SIZE = 32
NUM_WORKERS = 4

# Ścieżki katalogów
MODEL_DIR = os.path.join(BASE_DIR, "ResNet_mel", "model_outputs")
SIMPLE_CNN_MODEL_DIR = os.path.join(BASE_DIR, "Simple_CNN", "model_outputs")
VGG16_MODEL_DIR = os.path.join(BASE_DIR, "VGG16", "model_outputs")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_PATH = os.path.join(MODEL_DIR, f"best_model_{TIMESTAMP}.pt")
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "nemo_dataset")
ENSEMBLE_OUTPUT_DIR = os.path.join(BASE_DIR, "Ensemble", "ensemble_outputs")
FEATURE_RESULTS_DIR = os.path.join(
    BASE_DIR, "ResNet_for_all_repr", "feature_comparison_results"
)
PROCESSED_FEATURES_DIR = os.path.join(
    BASE_DIR, "ResNet_for_all_repr", "processed_features"
)

# Ustawienia optymalizacji
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.5
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7

# Ustawienia dla Simple_CNN
SIMPLE_CNN_NUM_EPOCHS = 30
TARGET_SAMPLE_RATE = 16000

# Ustawienia dla VGG16
VGG16_EPOCHS = 50
VGG16_LEARNING_RATE = 0.0001
VGG16_SAMPLE_RATE = 16000  # Specjalna częstotliwość dla VGG16
VGG16_N_MELS = 128
VGG16_DURATION = 3
VGG16_REDUCE_LR_PATIENCE = 5
VGG16_REDUCE_LR_FACTOR = 0.5
VGG16_MIN_LR = 1e-6

# Mapowanie emocji dla VGG16 (polskie nazwy dla wyświetlania)
VGG16_EMOTION_NAMES = [
    "Złość",
    "Strach",
    "Szczęście",
    "Smutek",
    "Zaskoczenie",
    "Neutralny",
]
VGG16_EMOTION_MAPPING = {
    "anger": 0,  # złość
    "fear": 1,  # strach
    "happiness": 2,  # szczęście
    "sadness": 3,  # smutek
    "surprised": 4,  # zaskoczenie
    "neutral": 5,  # neutralny
}

# Optuna
OPTUNA_TRIALS = 50
OPTUNA_TIMEOUT = 3600  # 1 godzina

# Ogólne ustawienia
MAX_LENGTH = 3.0
SAMPLE_RATE = 22050  # Dodana domyślna częstotliwość próbkowania

# Ustawienia ekstrakcji cech
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128
N_MFCC = 40
N_CHROMA = 12

# Utwórz katalogi, jeśli nie istnieją
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SIMPLE_CNN_MODEL_DIR, exist_ok=True)
os.makedirs(VGG16_MODEL_DIR, exist_ok=True)
os.makedirs(ENSEMBLE_OUTPUT_DIR, exist_ok=True)
