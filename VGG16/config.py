"""
Plik konfiguracyjny zawierający wszystkie parametry projektu
"""
import os

# Parametry dla spektrogramów Mela
SAMPLE_RATE = 16000  # Częstotliwość próbkowania
N_MELS = 128         # Liczba filtrów Mela
N_FFT = 2048         # Długość okna FFT
HOP_LENGTH = 512     # Długość przeskoku między ramkami
DURATION = 3         # Maksymalna długość nagrania w sekundach

# Parametry dla modelu
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
NUM_CLASSES = 6      # 6 emocji: złość, strach, szczęście, smutek, zaskoczenie, neutralny

# Nazwy emocji
EMOTION_NAMES = ['Złość', 'Strach', 'Szczęście', 'Smutek', 'Zaskoczenie', 'Neutralny']

# Mapowanie etykiet na indeksy
EMOTION_MAPPING = {
    'anger': 0,      # złość
    'fear': 1,       # strach 
    'happiness': 2,  # szczęście
    'sadness': 3,    # smutek
    'surprised': 4,  # zaskoczenie
    'neutral': 5     # neutralny
}

# Parametry dla augmentacji danych
DATA_AUGMENTATION = {
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'zoom_range': 0.1,
    'horizontal_flip': True
}

# Parametry dla callbacks
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6
