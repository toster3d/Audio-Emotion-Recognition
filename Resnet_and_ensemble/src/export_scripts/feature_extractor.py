import librosa
import numpy as np
from typing import Dict, Optional

# Importuj konfiguracje z głównego pliku config, jeśli tam są zdefiniowane
# Jeśli nie, można je przekazać jako argumenty lub zdefiniować tutaj.
# Dla przykładu załóżmy, że są dostępne globalnie lub przez import.
# Należy dostosować importy do rzeczywistej struktury projektu.
try:
    from config import MAX_LENGTH as CFG_MAX_LENGTH
    from config import N_MELS as CFG_N_MELS
    from config import N_MFCC as CFG_N_MFCC
    from config import N_CHROMA as CFG_N_CHROMA
    from config import N_FFT as CFG_N_FFT
    from config import HOP_LENGTH as CFG_HOP_LENGTH
    from config import SAMPLE_RATE as CFG_SAMPLE_RATE
except ImportError:
    # Domyślne wartości, jeśli konfiguracja nie jest dostępna
    print("Ostrzeżenie: Nie można zaimportować konfiguracji. Używam domyślnych wartości dla ekstrakcji cech.")
    CFG_MAX_LENGTH: float = 3.0
    CFG_N_MELS: int = 128
    CFG_N_MFCC: int = 40
    CFG_N_CHROMA: int = 12
    CFG_N_FFT: int = 2048
    CFG_HOP_LENGTH: int = 512
    CFG_SAMPLE_RATE: int = 22050


def extract_feature_single(
    audio_array: np.ndarray,
    sr: int,
    feature_type: str,
    max_length: float = CFG_MAX_LENGTH,
    n_mels: int = CFG_N_MELS,
    n_mfcc: int = CFG_N_MFCC,
    n_chroma: int = CFG_N_CHROMA,
    n_fft: int = CFG_N_FFT,
    hop_length: int = CFG_HOP_LENGTH,
    normalize: bool = True
) -> Optional[np.ndarray]:
    """
    Ekstrakcja pojedynczej cechy z sygnału audio.
    Bazująca na funkcji z ResNet_porownanie.ipynb.

    Args:
        audio_array: Sygnał audio w formie tablicy numpy.
        sr: Częstotliwość próbkowania.
        feature_type: Typ cechy do ekstrakcji ('melspectrogram', 'mfcc', 'chroma', itp.).
        max_length: Maksymalna długość sygnału w sekundach.
        n_mels: Liczba pasm melowych dla melspektrogramu.
        n_mfcc: Liczba współczynników MFCC.
        n_chroma: Liczba pasm chromatycznych.
        n_fft: Długość okna dla krótkoterminowej transformaty Fouriera.
        hop_length: Przesunięcie okna między kolejnymi ramkami.
        normalize: Czy normalizować wynikowe cechy.

    Returns:
        Wyekstrahowane cechy w formie tablicy numpy lub None, jeśli typ cechy jest nieznany.
    """
    target_length_samples: int = int(max_length * sr)
    if len(audio_array) > target_length_samples:
        audio_array = audio_array[:target_length_samples]
    else:
        padding: np.ndarray = np.zeros(target_length_samples - len(audio_array))
        audio_array = np.concatenate([audio_array, padding])

    feature: Optional[np.ndarray] = None

    if feature_type == "melspectrogram":
        S: np.ndarray = librosa.feature.melspectrogram(
            y=audio_array, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        feature = librosa.power_to_db(S, ref=np.max)
    elif feature_type == "mfcc":
        feature = librosa.feature.mfcc(
            y=audio_array, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
        )
    elif feature_type == "chroma":
        feature = librosa.feature.chroma_stft(
            y=audio_array, sr=sr, n_chroma=n_chroma, n_fft=n_fft, hop_length=hop_length
        )
    # Można dodać inne typy cech z ResNet_porownanie.ipynb, jeśli są potrzebne w procesie ONNX
    # np. 'spectrogram', 'spectral_contrast', 'zcr', 'rms', 'tempogram', 'tonnetz', 'delta_mfcc', 'delta_tempogram'
    else:
        print(f"Ostrzeżenie: Nieznany lub nieobsługiwany typ cechy w tym kontekście: {feature_type}")
        return None

    if normalize and feature is not None:
        if feature_type in ["mfcc", "delta_mfcc"]: # delta_mfcc do rozważenia
            # CMVN normalizacja - spójna z ResNet_porownanie.py
            mean = np.mean(feature, axis=1, keepdims=True)
            std = np.std(feature, axis=1, keepdims=True)
            feature = (feature - mean) / (std + 1e-8)
        elif feature_type in ["melspectrogram", "spectrogram"]:
            pass  # Już w dB
        else: # Dla chroma i innych
            feature_min: float = np.min(feature)
            feature_max: float = np.max(feature)
            if feature_max > feature_min:
                feature = (feature - feature_min) / (feature_max - feature_min)
    return feature


def extract_features_for_model(
    audio_array: np.ndarray,
    sr: int,
    feature_types: list[str],
    max_length: float = CFG_MAX_LENGTH,
    n_mels: int = CFG_N_MELS,
    n_mfcc: int = CFG_N_MFCC,
    n_chroma: int = CFG_N_CHROMA,
    n_fft: int = CFG_N_FFT,
    hop_length: int = CFG_HOP_LENGTH,
    normalize: bool = True
) -> Dict[str, np.ndarray]:
    """
    Ekstrahuje zestaw zdefiniowanych cech z sygnału audio i przygotowuje je
    w formacie oczekiwanym przez model ONNX (słownik z kluczami 'mel_input', itp.).

    Args:
        audio_array: Sygnał audio w formie tablicy numpy.
        sr: Częstotliwość próbkowania.
        feature_types: Lista typów cech do ekstrakcji (np. ['melspectrogram', 'mfcc', 'chroma']).
        max_length, n_mels, ...: Parametry ekstrakcji cech.
        normalize: Czy normalizować cechy.

    Returns:
        Słownik, gdzie klucze to nazwy wejść modelu ONNX (np. 'mel_input'),
        a wartości to wyekstrahowane cechy jako tablice numpy w kształcie [1, 1, N_FEATURES, TIME_FRAMES].
    """
    output_features: Dict[str, np.ndarray] = {}

    feature_params_map = {
        "melspectrogram": {"n_features": n_mels, "onnx_input_name": "mel_input"},
        "mfcc": {"n_features": n_mfcc, "onnx_input_name": "mfcc_input"},
        "chroma": {"n_features": n_chroma, "onnx_input_name": "chroma_input"},
    }

    for ft_type in feature_types:
        if ft_type not in feature_params_map:
            print(f"Ostrzeżenie: Typ cechy '{ft_type}' nie jest mapowany na znaną nazwę wejścia ONNX. Pomijam.")
            continue

        current_feature: Optional[np.ndarray] = extract_feature_single(
            audio_array, sr, ft_type, max_length, n_mels, n_mfcc, n_chroma, n_fft, hop_length, normalize
        )

        if current_feature is not None:
            # Reshape do [batch_size=1, channels=1, num_features, time_frames]
            # Upewniamy się, że liczba cech jest poprawna
            num_expected_features = feature_params_map[ft_type]["n_features"]
            if current_feature.shape[0] != num_expected_features:
                 print(f"Ostrzeżenie: Wyekstrahowana cecha {ft_type} ma {current_feature.shape[0]} wymiarów cech, oczekiwano {num_expected_features}. Sprawdź parametry.")
                 # Można tu dodać padding/cropping lub rzucić błąd, jeśli to krytyczne
                 # Na razie kontynuujemy, ale model może się wywalić
            
            reshaped_feature: np.ndarray = current_feature.reshape(1, 1, current_feature.shape[0], -1).astype(np.float32)
            output_features[feature_params_map[ft_type]["onnx_input_name"]] = reshaped_feature
        else:
            print(f"Nie udało się wyekstrahować cechy: {ft_type}")
            
    return output_features 