"""
Moduł zawierający funkcje przetwarzania danych dla modelu VGG16
"""

import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import time
from tqdm import tqdm
import os

# Importowanie konfiguracji
import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import *


def extract_melspectrogram(audio, sr=SAMPLE_RATE):
    """
    Ekstrakcja spektrogramu Mela z sygnału audio
    
    Args:
        audio: Sygnał audio
        sr: Częstotliwość próbkowania (domyślnie: SAMPLE_RATE)
        
    Returns:
        mel_spectrogram: Znormalizowany spektrogram Mela
    """
    # Przycinanie lub paddowanie do stałej długości
    if len(audio) > sr * DURATION:
        audio = audio[:sr * DURATION]
    else:
        audio = np.pad(audio, (0, max(0, sr * DURATION - len(audio))), 'constant')

    # Generowanie spektrogramu Mela
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # Konwersja do decybeli i normalizacja 
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram = (mel_spectrogram - np.min(mel_spectrogram)) / (np.max(mel_spectrogram) - np.min(mel_spectrogram))
    return mel_spectrogram


def load_nemo_dataset():
    """
    Ładowanie datasetu nEMO z Hugging Face z paskiem postępu
    
    Returns:
        dataset: Załadowany dataset nEMO
    """
    print("🔄 Rozpoczynam ładowanie datasetu nEMO...")
    try:
        start_time = time.time()
        dataset = load_dataset("amu-cai/nEMO")
        elapsed_time = time.time() - start_time
        print(f"✅ Dataset nEMO został pomyślnie załadowany ({elapsed_time:.2f}s)")
        print(f"Przykłady treningowe: {len(dataset['train'])}")
        print(f"Przykłady testowe: {len(dataset.get('test', dataset.get('validation', [])))} (test/validation)")
        
        # Sprawdzenie dostępnych emocji
        emotions = {}
        for sample in dataset['train']:
            emotion = sample['emotion']
            emotions[emotion] = emotions.get(emotion, 0) + 1
        
        print("Rozkład emocji w zbiorze treningowym:")
        for emotion, count in emotions.items():
            print(f"   - {emotion}: {count} próbek ({count/len(dataset['train'])*100:.1f}%)")
            
        return dataset
    except Exception as e:
        print(f"❌ Wystąpił błąd podczas ładowania datasetu: {str(e)}")
        raise e


def prepare_data(dataset, for_torch=True):
    """
    Przygotowanie danych: ekstrakcja spektrogramów Mela i etykiet z paskiem postępu
    
    Args:
        dataset: Dataset do przetworzenia
        for_torch: Czy dane mają być przygotowane dla PyTorch (domyślnie: True)
        
    Returns:
        X_train, X_val, X_test: Zbiory danych (treningowe, walidacyjne, testowe)
        y_train, y_val, y_test: Etykiety dla zbiorów
    """
    print("🔄 Rozpoczynam przygotowanie danych...")

    # Listy na dane
    mel_spectrograms = []
    labels = []
    count_by_emotion = {emotion: 0 for emotion in EMOTION_MAPPING.keys()}
    skipped_samples = 0
    
    # Przetwarzanie zestawu treningowego z paskiem postępu
    print("Ekstrahowanie spektrogramów Mela...")
    for sample in tqdm(dataset['train'], desc="Przetwarzanie próbek"):
        try:
            audio = np.array(sample['audio']['array'])
            emotion = sample['emotion']

            if emotion in EMOTION_MAPPING:
                mel_spec = extract_melspectrogram(audio)
                mel_spectrograms.append(mel_spec)
                labels.append(EMOTION_MAPPING[emotion])
                count_by_emotion[emotion] += 1
            else:
                skipped_samples += 1
        except Exception as e:
            skipped_samples += 1
            print(f"❌ Pominięto próbkę z powodu błędu: {str(e)}")

    # Konwersja do numpy arrays
    X = np.array(mel_spectrograms)
    y = np.array(labels)

    print(f"Przygotowano {len(X)} próbek z {len(dataset['train'])} dostępnych")
    print(f"Pominięto {skipped_samples} próbek")
    
    print("Liczba próbek dla każdej emocji:")
    for emotion, count in count_by_emotion.items():
        if count > 0:
            print(f"   - {emotion}: {count} próbek")

    # Dostosowanie wymiarów dla VGG16 (wymaga 3 kanałów kolorów)
    print("Dostosowywanie danych do formatu VGG16...")
    X = np.repeat(X[:, :, :, np.newaxis], 3, axis=3)
    
    if for_torch:
        print("Dane będą używane w formacie PyTorch")
    else:
        print("Dane przygotowane w formacie TensorFlow")

    # Podział na zbiory treningowe i testowe
    print("Dzielenie danych na zbiory treningowy, walidacyjny i testowy...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    # Wyświetlenie informacji o kształcie danych
    print(f"✅ Dane zostały pomyślnie przygotowane:")
    print(f"Zbiór treningowy: {X_train.shape[0]} próbek, kształt: {X_train.shape}")
    print(f"Zbiór walidacyjny: {X_val.shape[0]} próbek, kształt: {X_val.shape}")
    print(f"Zbiór testowy: {X_test.shape[0]} próbek, kształt: {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def predict_emotion(model, audio_path, device=None):
    """
    Predykcja emocji dla nowego pliku audio
    
    Args:
        model: Wytrenowany model
        audio_path: Ścieżka do pliku audio
        device: Urządzenie (cuda/cpu) - dla PyTorch
        
    Returns:
        predicted_class: Indeks przewidzianej klasy
        prediction: Wektor prawdopodobieństw dla wszystkich klas
    """
    print(f"🎵 Analizowanie pliku audio: {os.path.basename(audio_path)}")
    try:
        # Wczytanie audio
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        print(f"Plik wczytany: długość {len(audio)/sr:.2f}s, częstotliwość {sr}Hz")

        # Ekstrakcja spektrogramu Mela
        mel_spec = extract_melspectrogram(audio)
        
        # Przygotowanie do predykcji - replikacja na 3 kanały
        mel_spec = np.repeat(mel_spec[:, :, np.newaxis], 3, axis=2)
        
        import torch
        if isinstance(model, torch.nn.Module):
            # Obsługa modeli PyTorch
            print(" Wykryto model PyTorch")
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
            # Przygotowanie tensora wejściowego
            mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0).permute(0, 3, 1, 2)
            mel_spec = mel_spec.to(device)
            
            # Predykcja z modelem PyTorch
            model.eval()
            with torch.no_grad():
                outputs = model(mel_spec)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()
                probabilities = probabilities.cpu().numpy()
        else:
            # Obsługa modeli TensorFlow/Keras
            print("   Wykryto model TensorFlow/Keras")
            mel_spec = mel_spec.reshape(1, mel_spec.shape[0], mel_spec.shape[1], 3)
            
            # Predykcja
            probabilities = model.predict(mel_spec)[0]
            predicted_class = np.argmax(probabilities)
            
        # Wyniki
        print(f"✅ Wyniki analizy emocji:")
        print(f"   Przewidziana emocja: {EMOTION_NAMES[predicted_class]} ({probabilities[predicted_class]:.2%})")
        print("   Wszystkie prawdopodobieństwa:")
        
        # Sortowanie emocji według prawdopodobieństwa
        emotions_probs = [(EMOTION_NAMES[i], probabilities[i]) for i in range(len(EMOTION_NAMES))]
        emotions_probs.sort(key=lambda x: x[1], reverse=True)
        
        for emotion, prob in emotions_probs:
            print(f"   - {emotion}: {prob:.2%}")
            
        return predicted_class, probabilities
        
    except Exception as e:
        print(f"❌ Wystąpił błąd podczas analizy pliku audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None



