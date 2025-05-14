#!/usr/bin/env python
"""
Skrypt do testowania modelu ONNX bez pliku audio.
Używa losowych danych jako wejścia.
"""

import os
import sys
import json
from datetime import datetime

# Importy zewnętrzne
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from datetime import datetime
from src.config import N_MELS, N_MFCC, N_CHROMA


def generate_random_features(batch_size=1):
    """Generuje losowe cechy do testowania modelu."""
    print("Generowanie losowych danych wejściowych...")
    # Symulujemy dane o wymiarach [batch, channel, feature_dim, time]
    features = {}
    
    # Melspectrogram: [batch, channel, n_mels, time]
    features['mel_input'] = np.random.randn(batch_size, 1, N_MELS, 126).astype(np.float32)
    
    # MFCC: [batch, channel, n_mfcc, time]
    features['mfcc_input'] = np.random.randn(batch_size, 1, N_MFCC, 126).astype(np.float32)
    
    # Chroma: [batch, channel, n_chroma, time]
    features['chroma_input'] = np.random.randn(batch_size, 1, N_CHROMA, 126).astype(np.float32)
    
    return features

def create_onnx_session(model_path):
    """Tworzy sesję ONNX Runtime."""
    print(f"Inicjalizacja sesji ONNX Runtime z modelem: {model_path}")
    
    # Sprawdź, czy model istnieje
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model ONNX nie istnieje: {model_path}")
    
    # Tworzenie sesji ONNX Runtime
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        model_path, 
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )
    
    print("Sesja ONNX Runtime zainicjalizowana.")
    return session

def get_class_names(model_path):
    """Pobiera nazwy klas emocji z metadanych lub używa domyślnych."""
    # Sprawdź metadane
    metadata_path = os.path.splitext(model_path)[0] + "_metadata.json"
    
    if os.path.exists(metadata_path):
        print(f"Wczytywanie metadanych z: {metadata_path}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata.get("class_names", ["anger", "fear", "happiness", "neutral", "sadness", "surprised"])
    
    # Domyślne nazwy klas
    print("Nie znaleziono metadanych, używam domyślnych nazw klas.")
    return ["anger", "fear", "happiness", "neutral", "sadness", "surprised"]

def find_latest_model():
    """Znajduje najnowszy model ONNX."""
    # Sprawdź różne lokalizacje
    locations = [
        "exported_models/*/ensemble_model_optimized.onnx",
        "ensemble_outputs/*/onnx/ensemble_model_optimized.onnx",
        "src/exported_models/*/ensemble_model_optimized.onnx"
    ]
    
    for pattern in locations:
        import glob
        models = glob.glob(pattern)
        if models:
            return sorted(models, key=os.path.getmtime)[-1]  # Bierzemy najnowszy
    
    return None

def main(model_path=None):
    """Główna funkcja testowania modelu ONNX."""
    # Znajdź model, jeśli nie podano
    if model_path is None:
        model_path = find_latest_model()
        if model_path:
            print(f"Znaleziono najnowszy model: {model_path}")
        else:
            print("Nie znaleziono modelu ONNX.")
            print("Użycie: python test_without_audio.py [ścieżka_do_modelu_onnx]")
            return False
    
    try:
        # Utwórz katalog na wyniki
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"test_results/test_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generowanie losowych cech
        features = generate_random_features()
        
        # Inicjalizacja sesji ONNX
        session = create_onnx_session(model_path)
        
        # Pobierz nazwy klas
        class_names = get_class_names(model_path)
        
        # Uruchomienie inferencji
        print("\nUruchamianie inferencji ONNX...")
        outputs = session.run(None, features)
        
        # Interpretacja wyników
        probabilities = outputs[0][0]
        emotion_index = np.argmax(probabilities)
        emotion = class_names[emotion_index]
        confidence = probabilities[emotion_index]
        
        print("\n===== WYNIKI ROZPOZNAWANIA EMOCJI =====")
        print(f"Rozpoznana emocja: {emotion}")
        print(f"Pewność: {confidence:.2%}")
        print("\nPrawdopodobieństwa dla wszystkich emocji:")
        
        for i, name in enumerate(class_names):
            print(f"  {name}: {probabilities[i]:.2%}")
        
        # Wizualizacja wyników
        plt.figure(figsize=(10, 6))
        plt.bar(class_names, probabilities)
        plt.title(f'Rozpoznawanie emocji: {emotion} ({confidence:.2%})')
        plt.ylabel('Prawdopodobieństwo')
        plt.ylim(0, 1)
        
        for i, prob in enumerate(probabilities):
            plt.text(i, prob + 0.01, f'{prob:.2%}', ha='center')
        
        results_path = os.path.join(output_dir, "emotion_results.png")
        plt.tight_layout()
        plt.savefig(results_path)
        
        # Zapisz wyniki do JSON
        results = {
            "model_file": model_path,
            "detected_emotion": emotion,
            "confidence": float(confidence),
            "probabilities": {class_names[i]: float(p) for i, p in enumerate(probabilities)},
            "timestamp": timestamp
        }
        
        results_json_path = os.path.join(output_dir, "results.json")
        with open(results_json_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nWyniki zapisane w katalogu: {output_dir}")
        return True
    
    except Exception as e:
        import traceback
        print(f"Błąd podczas testowania modelu: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Obsługa argumentów linii poleceń
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main()
