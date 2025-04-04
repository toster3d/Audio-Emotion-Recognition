import os
from datetime import datetime

# Konfiguracja treningu
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.5
EARLY_STOPPING_PATIENCE = 7

# Ścieżki
MODEL_DIR = 'model_outputs'
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
MODEL_PATH = os.path.join(MODEL_DIR, f'best_model_{TIMESTAMP}.pt')

# Upewnij się, że katalog modeli istnieje
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
MAX_LENGTH = 3.0

