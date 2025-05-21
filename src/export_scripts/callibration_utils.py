import os
import numpy as np
import glob
import librosa
from onnxruntime.quantization import CalibrationDataReader
# Zaimportuj nową funkcję ekstrakcji cech
from helpers.feature_extractor import extract_features_for_model 
from typing import Dict, List, Optional

class EnsembleCalibrationDataReader(CalibrationDataReader):
    """
    Implementacja CalibrationDataReader dla modelu Ensemble.
    Dostarcza dane kalibracyjne do kwantyzacji statycznej.
    
    Args:
        audio_samples: Lista próbek audio jako tablice numpy
        feature_types: Lista typów cech używanych przez model (np. ['melspectrogram', 'mfcc', 'chroma'])
        sample_rate: Częstotliwość próbkowania audio
    """
    def __init__(self, audio_samples: list[np.ndarray], feature_types: list[str], sample_rate: int = 22050):
        self.audio_samples = audio_samples
        self.feature_types = feature_types # To jest lista np. ['melspectrogram', 'mfcc', 'chroma']
        self.sample_rate = sample_rate
        
        # Inicjalizacja indeksu dla iteracji przez dane
        self.datasize = len(audio_samples)
        self.current_index = 0
        
        # Cache dla przetworzonych cech (optymalizacja)
        self.feature_cache: dict[int, dict[str, np.ndarray]] = {}
        
        # Mapowanie nazw wejściowych ONNX na typy cech - już niepotrzebne tutaj, 
        # bo extract_features_for_model zwraca właściwe klucze
        # self.input_mapping = {
        #     'mel_input': 'melspectrogram',
        #     'mfcc_input': 'mfcc',
        #     'chroma_input': 'chroma'
        # }
    
    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Pobiera następną partię danych kalibracyjnych.
        
        Returns:
            Dict[str, ndarray] lub None, jeśli nie ma więcej danych
        """
        if self.current_index >= self.datasize:
            return None
        
        # Pobierz próbkę audio
        audio = self.audio_samples[self.current_index]
        
        inputs: dict[str, np.ndarray]
        # Sprawdź, czy cechy są w cache
        if self.current_index in self.feature_cache:
            inputs = self.feature_cache[self.current_index]
        else:
            # Ekstrakcja cech z próbki audio przy użyciu nowej funkcji
            # Przekazujemy listę typów cech, których oczekuje model ensemble
            inputs = extract_features_for_model(
                audio_array=audio,
                sr=self.sample_rate,
                feature_types=self.feature_types # np. ['melspectrogram', 'mfcc', 'chroma']
                # Pozostałe parametry (n_mels, n_fft, etc.) będą brane z feature_extractor (config lub default)
            )
            # Zapisz do cache dla potencjalnego ponownego użycia
            self.feature_cache[self.current_index] = inputs
        
        # Inkrementuj indeks
        self.current_index += 1
        return inputs
    
    # Usunięto metodę extract_feature, ponieważ używamy scentralizowanej funkcji
    # def extract_feature(self, audio, feature_type):
    #     ...
    
    def rewind(self):
        """Przewija czytnik na początek danych."""
        self.current_index = 0