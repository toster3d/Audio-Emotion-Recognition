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

# Importy zewnętrzne
import numpy as np
import onnxruntime as ort
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Dodaj katalog główny projektu do ścieżki
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importy lokalne
from src.config import MAX_LENGTH, N_FFT, HOP_LENGTH, N_MELS, N_MFCC, N_CHROMA, CLASS_NAMES



def extract_features(audio_file, sr=22050, max_length=MAX_LENGTH):
    """Ekstrahuje cechy audio dla modelu ONNX."""
    print(f"Wczytywanie pliku audio: {audio_file}")
    audio, _ = librosa.load(audio_file, sr=sr)
    
    # Przytnij lub uzupełnij do max_length
    target_length = int(sr * max_length)
    if len(audio) > target_length:
        print(f"Audio dłuższe niż {max_length}s - przycinanie...")
        audio = audio[:target_length]
    else:
        print(f"Audio krótsze niż {max_length}s - uzupełnianie zerami...")
        audio = np.pad(audio, (0, max(0, target_length - len(audio))), mode='constant')
    
    # Parametry ekstrakcji
    n_fft = N_FFT
    hop_length = HOP_LENGTH
    
    # Ekstrakcja cech
    features = {}
    
    # Melspectrogram
    print("Ekstrahowanie melspectrogramu...")
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS,
        n_fft=n_fft, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features['mel_input'] = mel_spec_db.reshape(1, 1, N_MELS, -1).astype(np.float32)
    
    # MFCC
    print("Ekstrahowanie MFCC...")
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=N_MFCC,
        n_fft=n_fft, hop_length=hop_length
    )
    features['mfcc_input'] = mfcc.reshape(1, 1, N_MFCC, -1).astype(np.float32)
    
    # Chroma
    print("Ekstrahowanie chroma...")
    chroma = librosa.feature.chroma_stft(
        y=audio, sr=sr, n_chroma=N_CHROMA,
        n_fft=n_fft, hop_length=hop_length
    )
    features['chroma_input'] = chroma.reshape(1, 1, N_CHROMA, -1).astype(np.float32)
    
    return features, audio

def visualize_features(features, audio, sr, output_path):
    """Wizualizuje cechy audio."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    
    # Plot waveform
    librosa.display.waveshow(audio, sr=sr, ax=axes[0])
    axes[0].set_title('Waveform')
    
    # Plot mel spectrogram
    mel_spec = features['mel_input'][0, 0]
    img = librosa.display.specshow(mel_spec, x_axis='time', y_axis='mel', 
                                   sr=sr, fmax=8000, ax=axes[1])
    axes[1].set_title('Mel Spectrogram')
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    # Plot MFCC
    mfcc = features['mfcc_input'][0, 0]
    img = librosa.display.specshow(mfcc, x_axis='time', ax=axes[2])
    axes[2].set_title('MFCC')
    fig.colorbar(img, ax=axes[2])
    
    # Plot chroma
    chroma = features['chroma_input'][0, 0]
    img = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', 
                                  ax=axes[3])
    axes[3].set_title('Chroma')
    fig.colorbar(img, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Wizualizacja zapisana do: {output_path}")

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
        return metadata.get("class_names", CLASS_NAMES)
    
    # Domyślne nazwy klas
    print("Nie znaleziono metadanych, używam domyślnych nazw klas.")
    return CLASS_NAMES

def find_latest_model():
    """Znajduje najnowszy model ONNX."""
    # Sprawdź różne lokalizacje
    locations = [
        "exported_models/onnx_*/ensemble_model_optimized.onnx",
        "ensemble_outputs/*/onnx/ensemble_model_optimized.onnx",
        "src/exported_models/onnx_*/ensemble_model_optimized.onnx"
    ]
    
    for pattern in locations:
        import glob
        models = glob.glob(pattern)
        if models:
            return sorted(models)[-1]  # Bierzemy najnowszy
    
    return None

def main(model_path=None, audio_path=None):
    """Główna funkcja testowania modelu ONNX."""
    # Znajdź model, jeśli nie podano
    if model_path is None:
        model_path = find_latest_model()
        if model_path:
            print(f"Znaleziono najnowszy model: {model_path}")
        else:
            print("Nie znaleziono modelu ONNX.")
            print("Użycie: python test_onnx_model.py [ścieżka_do_modelu_onnx] [ścieżka_do_pliku_audio]")
            return False
    
    # Sprawdź, czy podano plik audio
    if audio_path is None:
        # Sprawdź, czy istnieją jakieś pliki audio w folderze testowym
        test_dirs = ["data/test/", "test_audio/", "test_data/"]
        for test_dir in test_dirs:
            if os.path.exists(test_dir):
                for ext in [".wav", ".mp3", ".flac", ".ogg"]:
                    import glob
                    audio_files = glob.glob(f"{test_dir}/*{ext}")
                    if audio_files:
                        audio_path = audio_files[0]
                        print(f"Znaleziono przykładowy plik audio: {audio_path}")
                        break
        
        if audio_path is None:
            print("Nie znaleziono pliku audio do testów.")
            print("Użycie: python test_onnx_model.py [ścieżka_do_modelu_onnx] [ścieżka_do_pliku_audio]")
            return False
    
    try:
        # Utwórz katalog na wyniki
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"test_results/test_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Ekstrakcja cech audio
        features, audio = extract_features(audio_path)
        
        # Wizualizacja cech
        visualization_path = os.path.join(output_dir, "features_visualization.png")
        visualize_features(features, audio, sr=22050, output_path=visualization_path)
        
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
        print(f"Plik audio: {os.path.basename(audio_path)}")
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
            "audio_file": audio_path,
            "model_file": model_path,
            "detected_emotion": emotion,
            "confidence": float(confidence),
            "probabilities": {class_names[i]: float(p) for i, p in enumerate(probabilities)},
            "timestamp": timestamp
        }
        
        results_json_path = os.path.join(output_dir, "results.json")
        with open(results_json_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nWizualizacje i wyniki zapisane w katalogu: {output_dir}")
        return True
    
    except Exception as e:
        import traceback
        print(f"Błąd podczas testowania modelu: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Obsługa argumentów linii poleceń
    if len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main()
