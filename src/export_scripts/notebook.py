# Komórka 1: Importy i konfiguracja początkowa
import glob
import os
import torch
from datetime import datetime
import numpy as np
import librosa

# Importy z Twojego projektu
from export_scripts.export_onnx_notebook_helper import export_model_to_onnx
from export_scripts.callibration_utils import EnsembleCalibrationDataReader
from config import (
    BATCH_SIZE, DEVICE, TEST_SPLIT, CLASS_NAMES, FEATURE_RESULTS_DIR, 
    PROCESSED_FEATURES_DIR, ENSEMBLE_OUTPUT_DIR, OPTUNA_TRIALS,
    MAX_LENGTH, N_FFT, HOP_LENGTH, N_MELS, N_MFCC, N_CHROMA
)

# Funkcja do automatycznego konfigurowania ensemble modelu
def auto_configure_ensemble(models_base_dir=None, ensemble_save_dir=None, best_models_only=True):
    """
    Automatycznie konfiguruje model ensemble na podstawie dostępnych modeli.
    
    Args:
        models_base_dir: Ścieżka bazowa do katalogu z modelami
        ensemble_save_dir: Ścieżka do zapisu modelu ensemble
        best_models_only: Czy używać tylko najlepszych modeli
        
    Returns:
        tuple: (model_ensemble, feature_types)
    """
    from helpers.ensemble_model import WeightedEnsembleModel
    from helpers.utils import load_pretrained_model
    from helpers.resnet_model_definition import AudioResNet
    
    # Jeśli nie podano ścieżek, użyj domyślnych
    if models_base_dir is None:
        models_base_dir = FEATURE_RESULTS_DIR
    
    if ensemble_save_dir is None:
        ensemble_save_dir = ENSEMBLE_OUTPUT_DIR
    
    # Szukaj modeli dla różnych typów cech
    feature_types = ['melspectrogram', 'mfcc', 'chroma']
    model_paths = {}
    models_dict = {}
    
    for feature_type in feature_types:
        # Wzorzec dla najlepszych modeli
        if best_models_only:
            pattern = f"{models_base_dir}/{feature_type}/best_model_{feature_type}_*.pt"
        else:
            pattern = f"{models_base_dir}/{feature_type}/model_{feature_type}_*.pt"
        
        # Szukaj modeli pasujących do wzorca
        model_files = glob.glob(pattern)
        
        if model_files:
            # Wybierz najnowszy model (zakładając, że nazwy plików zawierają timestamp)
            newest_model = sorted(model_files)[-1]
            model_paths[feature_type] = newest_model
            
            # Załaduj model
            print(f"Ładowanie modelu dla {feature_type}: {os.path.basename(newest_model)}")
            model = load_pretrained_model(
                model_path=newest_model,
                model_class=AudioResNet,
                num_classes=len(CLASS_NAMES),
                device='cpu'  # Używanie CPU do eksportu
            )
            
            if model is not None:
                models_dict[feature_type] = model
            else:
                print(f"Ostrzeżenie: Nie udało się załadować modelu dla {feature_type}")
    
    # Sprawdź, czy znaleziono jakiekolwiek modele
    if not models_dict:
        raise ValueError("Nie znaleziono żadnych modeli! Sprawdź ścieżki do katalogów z modelami.")
    
    # Utwórz model ensemble
    ensemble_model = WeightedEnsembleModel(models_dict, weights=None)
    
    return ensemble_model, list(models_dict.keys())

# Komórka 2: Załadowanie lub utworzenie modelu ensemble
# Sprawdź, czy model ensemble jest już dostępny w sesji
ensemble_model_loaded = None
# Zmienna ensemble_model może nie być zdefiniowana przy pierwszym uruchomieniu,
# więc sprawdzamy jej istnienie przed próbą odwołania się do niej.
if ('ensemble_model' in locals() and isinstance(locals()['ensemble_model'], torch.nn.Module)) or \
   ('ensemble_model' in globals() and isinstance(globals()['ensemble_model'], torch.nn.Module)):
    print("Model ensemble już załadowany w sesji. Próbuję go użyć.")
    # Jeśli ensemble_model istnieje i jest modelem, przypisz go
    # Ta linia była problematyczna, jeśli ensemble_model nie było zdefiniowane:
    # ensemble_model_loaded = ensemble_model
    # Bezpieczniej jest odwołać się przez locals() lub globals() po sprawdzeniu
    if 'ensemble_model' in locals():
        ensemble_model_loaded = locals()['ensemble_model']
    elif 'ensemble_model' in globals():
        ensemble_model_loaded = globals()['ensemble_model']
else:
    print("Model ensemble nie znaleziony w bieżącej sesji. Próba załadowania/utworzenia...")
    # Spróbuj znaleźć zapisany model ensemble
    ensemble_model_path = None
    ensemble_patterns = [
        "ensemble_outputs/*/models/best_ensemble_model.pt",  # Najlepsze modele z Optuna
        "ensemble_outputs/*/models/ensemble_model.pt",       # Modele bez optymalizacji
        "exported_models/*/best_ensemble_model.pt",          # Eksportowane najlepsze modele
        "exported_models/*/ensemble_model.pt"                # Eksportowane modele
    ]
    
    for pattern in ensemble_patterns:
        model_files = glob.glob(pattern)
        if model_files:
            # Sortuj według daty modyfikacji - najnowszy na końcu
            ensemble_model_path = sorted(model_files, key=os.path.getmtime)[-1]
            print(f"Znaleziono model ensemble: {ensemble_model_path}")
            break
    
    # Jeśli znaleziono model, spróbuj go załadować
    if ensemble_model_path:
        try:
            from helpers.ensemble_model import WeightedEnsembleModel
            from helpers.resnet_model_definition import AudioResNet
            
            # Załaduj modele bazowe (potrzebne do rekonstrukcji)
            _, feature_types = auto_configure_ensemble()
            
            # Utwórz słownik modeli bazowych
            from helpers.utils import load_pretrained_model
            models_dict = {}
            for ft in feature_types:
                model = AudioResNet(num_classes=len(CLASS_NAMES))
                models_dict[ft] = model
            
            # Załaduj model ensemble
            ensemble_model_loaded, _ = WeightedEnsembleModel.load(ensemble_model_path, models_dict)
            print(f"Model ensemble załadowany z {ensemble_model_path}")
            print(f"Typy cech: {ensemble_model_loaded.feature_types}")
            print(f"Wagi: {ensemble_model_loaded.get_weights()}")
        except Exception as e:
            print(f"Błąd podczas ładowania modelu ensemble: {e}")
            ensemble_model_loaded = None
    
    # Jeśli nie udało się załadować modelu, utwórz nowy
    if ensemble_model_loaded is None:
        try:
            print("Tworzenie nowego modelu ensemble...")
            ensemble_model_loaded, feature_types = auto_configure_ensemble()
            print(f"Utworzono model ensemble z typami cech: {feature_types}")
        except Exception as e:
            print(f"Błąd podczas tworzenia modelu ensemble: {e}")
            raise

# Przypisz załadowany model do zmiennej ensemble_model dla dalszego użycia
ensemble_model = ensemble_model_loaded
print(f"Model ensemble gotowy do eksportu. Typy cech: {ensemble_model.feature_types}")

# Komórka 3: Eksport modelu do ONNX
# Utwórz katalog wyjściowy z timestampem
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f"exported_models/onnx_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
print(f"Katalog wyjściowy: {output_dir}")

# Konfiguracja eksportu i kwantyzacji
quantize_model = True         # Czy kwantyzować model po eksporcie
quantization_type = "static"  # Typ kwantyzacji (static/dynamic) - dla CNN zalecana static
activation_type = "int8"      # Typ danych aktywacji (int8 dla modeli CNN)
weight_type = "int8"          # Typ danych wag (int8 dla modeli CNN)
quant_format = "qdq"          # Format kwantyzacji (qdq jest zalecane dla modeli CNN)
per_channel = True            # Kwantyzacja per-channel dla lepszej dokładności

# Przygotowanie danych kalibracyjnych dla kwantyzacji statycznej
print("\nPrzygotowywanie danych kalibracyjnych dla kwantyzacji statycznej...")

# Załadowanie próbek audio do kalibracji
audio_samples = []
num_samples = 10  # Liczba próbek audio do kalibracji

# Szukaj plików audio w różnych katalogach
audio_paths = []
data_dirs = ['data/test', 'data/valid', 'data_processed/test', 'data_processed/valid']

for data_dir in data_dirs:
    if os.path.exists(data_dir):
        for ext in ['.wav', '.mp3', '.ogg']:
            audio_files = glob.glob(f"{data_dir}/**/*{ext}", recursive=True)
            if audio_files:
                audio_paths.extend(audio_files[:num_samples])
                if len(audio_paths) >= num_samples:
                    break
        if len(audio_paths) >= num_samples:
            break

# Wczytaj pliki audio lub wygeneruj sztuczne dane, jeśli nie znaleziono
if not audio_paths:
    print("Nie znaleziono plików audio. Generowanie sztucznych danych...")
    
    # Generowanie sztucznych danych
    for i in range(num_samples):
        t = np.linspace(0, MAX_LENGTH, int(22050 * MAX_LENGTH), endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t) * 0.3  # Ton A4 (440 Hz)
        audio += np.sin(2 * np.pi * 880 * t) * 0.2  # Harmoniczna
        audio += np.random.normal(0, 0.1, size=len(t))  # Szum
        audio = audio / np.max(np.abs(audio))  # Normalizacja
        audio_samples.append(audio)
        print(f"Wygenerowano sztuczną próbkę audio {i+1}/{num_samples}")
else:
    # Ładowanie rzeczywistych plików audio
    for i, path in enumerate(audio_paths[:num_samples]):
        try:
            audio, sr = librosa.load(path, sr=22050)
            target_length = int(MAX_LENGTH * sr)
            if len(audio) > target_length:
                audio = audio[:target_length]
            else:
                audio = np.pad(audio, (0, max(0, target_length - len(audio))), mode='constant')
            audio_samples.append(audio)
            print(f"Załadowano próbkę {i+1}/{len(audio_paths[:num_samples])}: {os.path.basename(path)}")
        except Exception as e:
            print(f"Błąd ładowania {path}: {e}")

# Utwórz czytnik danych kalibracyjnych
feature_types = ensemble_model.feature_types  # Pobierz typy cech z modelu
print(f"Typy cech modelu: {feature_types}")

calibration_reader = EnsembleCalibrationDataReader(
    audio_samples=audio_samples,
    feature_types=feature_types,
    sample_rate=22050
)

print(f"Utworzono czytnik danych kalibracyjnych z {len(audio_samples)} próbkami audio")

print("\nRozpoczynam eksport modelu...")

# Eksport modelu do formatu ONNX
try:
    export_result = export_model_to_onnx(
        ensemble_model=ensemble_model,
        output_dir=output_dir,
        class_names=CLASS_NAMES,
        export_params={
            "quantize_model": quantize_model,
            "quantization_type": quantization_type,
            "activation_type": activation_type,
            "weight_type": weight_type,
            "quant_format": quant_format,
            "per_channel": per_channel,
            "reduce_range": False,  # True tylko dla starszych CPU bez wspierania instrukcji VNNI
            "use_cuda": torch.cuda.is_available(),
            "opset_version": 17  # Używaj najnowszej wersji opset
        },
        calibration_data_reader=calibration_reader  # Przekazanie czytnika danych kalibracyjnych
    )
except Exception as e:
    print(f"Błąd podczas eksportu modelu: {e}")
    import traceback
    traceback.print_exc()
    raise

if export_result.get("success", False):
    print("\nEksport modelu ensemble do ONNX zakończony sukcesem!")
    print(f"Model ONNX zapisany w: {export_result['onnx_path']}")
    print(f"Zoptymalizowany model ONNX zapisany w: {export_result['optimized_onnx_path']}")
    
    # Rozmiar modelu
    original_size_mb = export_result['sizes_mb'].get('original', 0)
    optimized_size_mb = export_result['sizes_mb'].get('optimized', 0)
    print(f"Rozmiar oryginalnego modelu: {original_size_mb:.2f} MB")
    print(f"Rozmiar zoptymalizowanego modelu: {optimized_size_mb:.2f} MB")
    print(f"Redukcja rozmiaru po optymalizacji: {export_result.get('optimization_reduction', 0):.2f}%")
    
    # Sprawdź czy kwantyzacja została wykonana
    if "quantized_onnx_path" in export_result and export_result['quantized_onnx_path'] and os.path.exists(export_result['quantized_onnx_path']):
        quantized_size_mb = export_result['sizes_mb'].get('quantized', 0)
        print(f"Skwantyzowany model ONNX zapisany w: {export_result['quantized_onnx_path']}")
        print(f"Rozmiar skwantyzowanego modelu: {quantized_size_mb:.2f} MB")
        print(f"Redukcja rozmiaru po kwantyzacji: {export_result.get('quantization_reduction_vs_source', 0):.2f}%")
        
        # Jeśli użyto kwantyzacji statycznej, pokaż szczegóły
        if export_result.get("quantization_type_used") == "static":
            print(f"\nSzczegóły kwantyzacji statycznej:")
            if "quantization_params_used" in export_result:
                params = export_result["quantization_params_used"] 
                print(f"- Używana parametryzacja: {params}")
                if "op_types_to_quantize" in params and params["op_types_to_quantize"]:
                    print(f"- Kwantyzowane operacje: {params['op_types_to_quantize']}")
    else:
        print("Kwantyzacja nie została przeprowadzona lub nie powiodła się.")
        if "quantization_error" in export_result:
            print(f"Błąd kwantyzacji: {export_result['quantization_error']}")
    
    # Informacje o metadanych i weryfikacji
    print(f"Metadane modelu zapisane w: {export_result.get('metadata_path', 'N/A')}")
    
    # Wyświetl wyniki weryfikacji
    if "verification_result" in export_result:
        result = export_result["verification_result"]
        if result.get("success", False):
            print("\nWeryfikacja modelu zakończona powodzeniem")
            if "max_diff" in result:
                print(f"Maksymalna różnica między PyTorch i ONNX: {result['max_diff']:.6f}")
            if "has_warning" in result and result["has_warning"]:
                print(f"Ostrzeżenie: {result.get('warning', 'Wykryto różnice, ale mieszczą się w tolerancji')}")
        else:
            print(f"\nWeryfikacja modelu nie powiodła się: {result.get('error', 'Nieznany błąd')}")
else:
    print("\nEksport modelu się nie powiódł")
    print(f"Błąd: {export_result.get('error', 'Nieznany błąd')}") 