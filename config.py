import os
import torch
from datetime import datetime

# Ścieżki projektowe
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ustawienia urządzenia
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Nazwy klas emocji
CLASS_NAMES = ['anger', 'fear', 'happiness', 'neutral', 'sadness', 'surprised']

# Ustawienia walidacji krzyżowej
CV_FOLDS = 5
SEED = 42
TEST_SPLIT = 0.2

# Konfiguracja batcha
BATCH_SIZE = 32
NUM_WORKERS = 4

# Ścieżki katalogów
MODEL_DIR = os.path.join(BASE_DIR, 'model_outputs')
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
MODEL_PATH = os.path.join(MODEL_DIR, f'best_model_{TIMESTAMP}.pt')
DATASET_PATH = 'data/nemo_dataset'
ENSEMBLE_OUTPUT_DIR = os.path.join(BASE_DIR, 'ensemble_outputs')
FEATURE_RESULTS_DIR = os.path.join(BASE_DIR, 'feature_comparison_results')
PROCESSED_FEATURES_DIR = os.path.join(BASE_DIR, 'processed_features')

# Ustawienia optymalizacji
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.5
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7

# Optuna
OPTUNA_TRIALS = 50
OPTUNA_TIMEOUT = 3600  # 1 godzina

# Ogólne ustawienia
MAX_LENGTH = 3.0

# Ustawienia ekstrakcji cech
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128
N_MFCC = 40
N_CHROMA = 12

# Utwórz katalogi, jeśli nie istnieją
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ENSEMBLE_OUTPUT_DIR, exist_ok=True)