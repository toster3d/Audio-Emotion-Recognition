#!/usr/bin/env python
"""
Skrypt do testowania modelu ONNX rozpoznającego emocje w mowie.

Użycie:
python test_onnx_model.py [ścieżka_do_modelu_onnx] [ścieżka_do_pliku_audio]

Jeśli nie podano argumentów, skrypt poszuka domyślnych lokalizacji.
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, List

# Importy zewnętrzne
import numpy as np
import onnxruntime as ort
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Dodaj katalog główny projektu do ścieżki
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importy lokalne
try:
    from src.config import (
        MAX_LENGTH, N_FFT, HOP_LENGTH, N_MELS, N_MFCC, N_CHROMA, 
        CLASS_NAMES, SAMPLE_RATE as CONFIG_SAMPLE_RATE
    )
except ImportError:
    print("Ostrzeżenie: Nie można zaimportować pełnej konfiguracji z config.py. Używam wartości domyślnych.")
    CONFIG_SAMPLE_RATE = 22050 # Domyślna wartość, jeśli import zawiedzie
    MAX_LENGTH = 3.0
    # N_FFT, HOP_LENGTH, etc. również mogą potrzebować domyślnych wartości, jeśli nie są używane przez feature_extractor
    CLASS_NAMES = ["anger", "fear", "happiness", "neutral", "sadness", "surprised"] # Domyślne klasy
    # Pozostałe (N_MELS, N_MFCC, N_CHROMA, N_FFT, HOP_LENGTH) są używane przez feature_extractor, który ma swoje defaulty

from helpers.feature_extractor import extract_features_for_model


def extract_and_load_audio(audio_file_path: str, sr: int = CONFIG_SAMPLE_RATE, max_length_sec: float = MAX_LENGTH) -> Optional[Tuple[Dict[str, np.ndarray], np.ndarray]]:
    """Wczytuje plik audio i ekstrahuje z niego cechy za pomocą ujednoliconej funkcji."""
    try:
        print(f"Wczytywanie pliku audio: {audio_file_path}")
        audio, loaded_sr = librosa.load(audio_file_path, sr=None) # Wczytaj oryginalną SR
        if loaded_sr != sr:
            print(f"Resampling audio z {loaded_sr} Hz do {sr} Hz")
            audio = librosa.resample(audio, orig_sr=loaded_sr, target_sr=sr)
        
        target_length_samples = int(sr * max_length_sec)
        waveform_for_viz = audio[:target_length_samples] if len(audio) > target_length_samples else np.pad(audio, (0, max(0, target_length_samples - len(audio))), mode='constant')

        print("Ekstrahowanie cech...")
        feature_types_to_extract = ['melspectrogram', 'mfcc', 'chroma'] 
        
        extracted_features: Dict[str, np.ndarray] = extract_features_for_model(
            audio_array=audio, 
            sr=sr,
            feature_types=feature_types_to_extract,
            max_length=max_length_sec
        )
        if not extracted_features: # Jeśli słownik jest pusty
             print(f"Nie udało się wyekstrahować żadnych cech dla {audio_file_path}")
             return None
        return extracted_features, waveform_for_viz
    except Exception as e:
        print(f"Błąd podczas wczytywania lub przetwarzania pliku {audio_file_path}: {e}")
        return None


def visualize_features(features: Dict[str, np.ndarray], audio_waveform: np.ndarray, sr: int, output_path: str):
    """Wizualizuje cechy audio."""
    num_features_to_plot = 1 
    plot_map = {}
    if features.get('mel_input') is not None: plot_map['mel_input'] = ('Mel Spectrogram', 'mel', '%+2.0f dB')
    if features.get('mfcc_input') is not None: plot_map['mfcc_input'] = ('MFCC', None, None)
    if features.get('chroma_input') is not None: plot_map['chroma_input'] = ('Chroma', 'chroma', None)
    
    num_features_to_plot += len(plot_map)
    if num_features_to_plot == 1 and not plot_map: 
        print("Brak cech do zwizualizowania poza waveform.")
        return
        
    fig, axes = plt.subplots(num_features_to_plot, 1, figsize=(12, 3 * num_features_to_plot), squeeze=False)
    axes = axes.flatten() 
    
    current_axis = 0
    librosa.display.waveshow(audio_waveform, sr=sr, ax=axes[current_axis])
    axes[current_axis].set_title('Waveform')
    current_axis += 1
    
    for feature_key, (title, y_axis_type, colorbar_format) in plot_map.items():
        feature_data = features[feature_key][0, 0] 
        img = librosa.display.specshow(feature_data, x_axis='time', y_axis=y_axis_type, 
                                       sr=sr, ax=axes[current_axis], fmax=8000 if y_axis_type == 'mel' else None)
        axes[current_axis].set_title(title)
        if colorbar_format:
            fig.colorbar(img, ax=axes[current_axis], format=colorbar_format)
        else:
            fig.colorbar(img, ax=axes[current_axis])
        current_axis += 1
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Wizualizacja zapisana do: {output_path}")
    plt.close(fig) # Jawne zamknięcie figury


def create_onnx_session(model_path: str) -> Optional[ort.InferenceSession]:
    """Tworzy sesję ONNX Runtime."""
    try:
        print(f"Inicjalizacja sesji ONNX Runtime z modelem: {model_path}")
        if not os.path.exists(model_path):
            print(f"BŁĄD: Model ONNX nie istnieje: {model_path}")
            return None
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options,
            providers=['CPUExecutionProvider'] # Można dodać 'CUDAExecutionProvider' jeśli dostępne
        )
        print("Sesja ONNX Runtime zainicjalizowana.")
        return session
    except Exception as e:
        print(f"Błąd podczas tworzenia sesji ONNX dla {model_path}: {e}")
        return None

def get_class_names_from_metadata(model_path: str, default_class_names: List[str]) -> List[str]:
    metadata_path = os.path.splitext(model_path)[0] + "_metadata.json"
    if os.path.exists(metadata_path):
        try:
            print(f"Wczytywanie metadanych z: {metadata_path}")
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata.get("class_names", default_class_names)
        except Exception as e:
            print(f"Błąd podczas wczytywania metadanych z {metadata_path}: {e}. Używam domyślnych klas.")
            return default_class_names
    print(f"Nie znaleziono pliku metadanych ({metadata_path}), używam domyślnych nazw klas.")
    return default_class_names

def find_latest_model(locations: Optional[List[str]] = None) -> Optional[str]:
    if locations is None:
        locations = [
            "exported_models/onnx_*/ensemble_model_optimized.onnx",
            "exported_models/onnx_*/ensemble_model.onnx", # Dodaj też nie-optymalizowane, jeśli optymalizacja pominięta
            "ensemble_outputs/*/onnx/ensemble_model_optimized.onnx",
            "ensemble_outputs/*/onnx/ensemble_model.onnx",
            "src/exported_models/onnx_*/ensemble_model_optimized.onnx",
            "src/exported_models/onnx_*/ensemble_model.onnx"
        ]
    all_models = []
    for pattern in locations:
        import glob
        all_models.extend(glob.glob(pattern))
    if not all_models:
        return None
    return sorted(all_models, key=os.path.getmtime)[-1]


def main(model_path: Optional[str] = None, audio_path: Optional[str] = None) -> bool:
    if model_path is None:
        model_path = find_latest_model()
        if model_path:
            print(f"Znaleziono najnowszy model: {model_path}")
        else:
            print("Nie znaleziono modelu ONNX w domyślnych lokalizacjach.")
            print("Użycie: python test_onnx_model.py [ścieżka_do_modelu_onnx] [ścieżka_do_pliku_audio]")
            return False
    elif not os.path.exists(model_path):
        print(f"Podana ścieżka modelu ONNX nie istnieje: {model_path}")
        return False
    
    if audio_path is None:
        # ... (logika szukania audio bez zmian)
        test_dirs = ["data/test/", "test_audio/", "test_data/", "./"] # Dodano ./ dla testów lokalnych
        found_audio_path: Optional[str] = None
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                for ext in [".wav", ".mp3", ".flac", ".ogg"]:
                    import glob
                    # Wyszukaj pliki bezpośrednio w test_dir oraz w podkatalogach (rekurencyjnie do 1 poziomu dla prostoty)
                    audio_files = glob.glob(os.path.join(test_dir, f"*{ext}"))
                    audio_files.extend(glob.glob(os.path.join(test_dir, f"*/*{ext}")))
                    if audio_files:
                        found_audio_path = sorted(audio_files, key=os.path.getmtime, reverse=True)[0] # Najnowszy, jeśli jest wiele
                        print(f"Znaleziono przykładowy plik audio: {found_audio_path}")
                        break
            if found_audio_path: 
                audio_path = found_audio_path
                break 
        
        if audio_path is None: 
            print("Nie znaleziono pliku audio do testów w domyślnych lokalizacjach.")
            print("Użycie: python test_onnx_model.py [ścieżka_do_modelu_onnx] [ścieżka_do_pliku_audio]")
            return False
    elif not os.path.exists(audio_path):
        print(f"Podana ścieżka pliku audio nie istnieje: {audio_path}")
        return False

    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Użyj katalogu skryptu jako bazy dla wyników, jeśli nie jest uruchamiany z głównego katalogu projektu
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir_base = os.path.join(script_dir, "test_results_onnx") 
        output_dir = os.path.join(output_dir_base, f"test_{os.path.splitext(os.path.basename(model_path))[0]}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        extraction_result = extract_and_load_audio(audio_path, sr=CONFIG_SAMPLE_RATE, max_length_sec=MAX_LENGTH)
        if extraction_result is None:
            print("Nie udało się załadować i przetworzyć audio. Przerywam.")
            return False
        features, waveform_for_viz = extraction_result

        visualization_path = os.path.join(output_dir, "features_visualization.png")
        visualize_features(features, waveform_for_viz, sr=CONFIG_SAMPLE_RATE, output_path=visualization_path)
        
        session = create_onnx_session(model_path)
        if session is None:
            print("Nie udało się utworzyć sesji ONNX. Przerywam.")
            return False
            
        class_names_list = get_class_names_from_metadata(model_path, CLASS_NAMES)
        
        print("\nUruchamianie inferencji ONNX...")
        onnx_inputs = features
        try:
            outputs = session.run(None, onnx_inputs)
        except Exception as e_infer:
            print(f"Błąd podczas inferencji ONNX: {e_infer}")
            # Dodatkowe informacje o wejściach mogą pomóc w debugowaniu
            print("Typy danych wejściowych przekazanych do ONNX:")
            for name, arr in onnx_inputs.items():
                print(f"  - {name}: {arr.dtype}, shape: {arr.shape}")
            return False
        
        probabilities = outputs[0][0]
        emotion_index = np.argmax(probabilities)
        emotion = class_names_list[emotion_index]
        confidence = probabilities[emotion_index]
        
        print("\n===== WYNIKI ROZPOZNAWANIA EMOCJI =====")
        print(f"Plik audio: {os.path.basename(audio_path)}")
        print(f"Model: {os.path.basename(model_path)}")
        print(f"Rozpoznana emocja: {emotion}")
        print(f"Pewność: {confidence:.2%}")
        print("\nPrawdopodobieństwa dla wszystkich emocji:")
        
        for i, name in enumerate(class_names_list):
            print(f"  {name}: {probabilities[i]:.2%}")
        
        # Wizualizacja prawdopodobieństw
        fig_probs, ax_probs = plt.subplots(figsize=(10, 6))
        ax_probs.bar(class_names_list, probabilities, color='skyblue')
        ax_probs.set_title(f'Rozpoznawanie emocji: {emotion} ({confidence:.2%})\nModel: {os.path.basename(model_path)}\nAudio: {os.path.basename(audio_path)}', fontsize=10)
        ax_probs.set_ylabel('Prawdopodobieństwo')
        ax_probs.set_ylim(0, 1)
        for i, prob in enumerate(probabilities):
            ax_probs.text(i, prob + 0.01, f'{prob:.2%}', ha='center')
        
        results_plot_path = os.path.join(output_dir, "emotion_probabilities.png")
        plt.tight_layout()
        plt.savefig(results_plot_path)
        plt.close(fig_probs) 
        
        results_data = {
            "audio_file": os.path.abspath(audio_path),
            "model_file": os.path.abspath(model_path),
            "detected_emotion": emotion,
            "confidence": float(confidence),
            "probabilities": {class_names_list[i]: float(p) for i, p in enumerate(probabilities)},
            "timestamp": timestamp,
            "output_directory": os.path.abspath(output_dir)
        }
        
        results_json_path = os.path.join(output_dir, "results_summary.json")
        with open(results_json_path, "w") as f:
            json.dump(results_data, f, indent=4)
        
        print(f"\nWizualizacje i wyniki zapisane w katalogu: {os.path.abspath(output_dir)}")
        return True
    
    except Exception as e:
        import traceback
        print(f"Krytyczny błąd podczas testowania modelu: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    model_arg: Optional[str] = sys.argv[1] if len(sys.argv) > 1 else None
    audio_arg: Optional[str] = sys.argv[2] if len(sys.argv) > 2 else None
    success = main(model_path=model_arg, audio_path=audio_arg)
    sys.exit(0 if success else 1)
