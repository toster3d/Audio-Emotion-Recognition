import os
import sys
import json
import torch
import argparse
from datetime import datetime
import numpy as np
import librosa

# Importowanie niezbędnych modułów
from export_scripts.ensemble_onnx_wrapper import (
    export_onnx_model, 
    generate_dummy_input, 
    optimize_onnx_model, 
    quantize_dynamic_onnx_model,
    quantize_static_onnx_model,
    verify_onnx_model,
    create_onnx_inference_session,
    get_model_size
)
from helpers.ensemble_model import WeightedEnsembleModel
from helpers.utils import load_pretrained_model
from helpers.resnet_model_definition import AudioResNet
from config import CLASS_NAMES, MAX_LENGTH, N_FFT, HOP_LENGTH, N_MELS, N_MFCC, N_CHROMA

def load_ensemble_model(model_path=None):
    """Ładowanie modelu ensemble z pliku lub tworzenie domyślnego."""
    if model_path and os.path.exists(model_path):
        print(f"Ładowanie modelu z: {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Wczytywanie modeli składowych
        models_dict = {}
        feature_types = state_dict.get('feature_types', ['melspectrogram', 'mfcc', 'chroma'])
        
        # Sprawdzanie, czy model zawiera stan modelu ensemble czy tylko wagi
        if 'model_state_dict' in state_dict:
            # Tworzenie modeli składowych
            for name in feature_types:
                model = AudioResNet(num_classes=len(CLASS_NAMES))
                models_dict[name] = model
            
            # Tworzenie modelu ensemble
            ensemble_model = WeightedEnsembleModel(models_dict)
            ensemble_model.load_state_dict(state_dict['model_state_dict'])
            
            # Wczytywanie wag, jeśli są dostępne
            if 'normalized_weights' in state_dict:
                ensemble_model.weights = state_dict['normalized_weights']
                print(f"Wczytano wagi: {state_dict['normalized_weights']}")
            
            return ensemble_model, feature_types
        else:
            print("Model nie zawiera state_dict, próba wczytania zaawansowanego modelu ensemble...")
            # Zakładanie, że mamy do czynienia z zapisanym stanem z modułem ensemble_trainer
            raise NotImplementedError("Funkcjonalność jeszcze nie zaimplementowana")
    else:
        # Tworzenie domyślnego modelu ensemble
        print("Tworzenie domyślnego modelu ensemble")
        models_dict = {}
        feature_types = ['melspectrogram', 'mfcc', 'chroma']
        
        # Wykorzystanie glob do znalezienia plików modeli
        import glob
        for name, pattern, shape in [
            ('melspectrogram', 'feature_comparison_results/melspectrogram/best_model_melspectrogram_*.pt', 128),
            ('mfcc', 'feature_comparison_results/mfcc/best_model_mfcc_*.pt', 40),
            ('chroma', 'feature_comparison_results/chroma/best_model_chroma_*.pt', 12)
        ]:
            model_files = glob.glob(pattern)
            if model_files:
                model_path = sorted(model_files)[-1]  # Wybieranie najnowszego
                model = load_pretrained_model(
                    model_path=model_path,
                    model_class=AudioResNet,
                    num_classes=len(CLASS_NAMES),
                    device='cpu'  # Używanie CPU do eksportu
                )
                
                if model is None:
                    raise ValueError(f"Nie udało się załadować modelu {name} z {model_path}")
                
                models_dict[name] = model
                print(f"Załadowano model {name} z {model_path}")
            else:
                raise ValueError(f"Nie znaleziono plików modelu dla wzorca: {pattern}")
        
        # Inicjalizacja modelu ensemble z równymi wagami
        ensemble_model = WeightedEnsembleModel(models_dict, weights=None)
        
        return ensemble_model, feature_types


def main():
    parser = argparse.ArgumentParser(description='Eksport modelu Ensemble do ONNX')
    parser.add_argument('--model_path', type=str, default=None, help='Ścieżka do pliku modelu')
    parser.add_argument('--output_dir', type=str, default=None, help='Katalog wyjściowy')
    parser.add_argument('--sample_rate', type=int, default=22050, help='Próbkowanie audio')
    parser.add_argument('--max_length', type=float, default=MAX_LENGTH, help='Maksymalna długość audio w sekundach')
    parser.add_argument('--n_fft', type=int, default=N_FFT, help='Rozmiar FFT')
    parser.add_argument('--hop_length', type=int, default=HOP_LENGTH, help='Długość kroku')
    parser.add_argument('--quantize', action='store_true', help='Czy wykonać kwantyzację modelu')
    parser.add_argument('--quant_type', type=str, choices=['static', 'dynamic'], default='static', 
                        help='Typ kwantyzacji (static/dynamic), dla modeli CNN zalecana jest static')
    parser.add_argument('--calib_samples', type=int, default=10, 
                        help='Liczba próbek do kalibracji dla kwantyzacji statycznej')
    args = parser.parse_args()
    
    # Domyślne ścieżki, jeśli nie podano argumentów
    if args.model_path is None:
        # Szukanie najnowszego pliku modelu w katalogach ensemble_outputs
        import glob
        model_files = glob.glob("ensemble_outputs/ensemble_run_*/models/ensemble_model.pt")
        if model_files:
            args.model_path = sorted(model_files)[-1]  # Wybieranie najnowszego
            print(f"Używanie najnowszego modelu: {args.model_path}")
        else:
            print("Nie znaleziono modeli ensemble.")
            return False
    
    if args.output_dir is None:
        # Utworzenie katalogu na modele ONNX z timestampem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f"exported_models/onnx_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Katalog wyjściowy: {args.output_dir}")
    
    # Ścieżki do modeli ONNX
    onnx_path = os.path.join(args.output_dir, "ensemble_model.onnx")
    onnx_optimized_path = os.path.join(args.output_dir, "ensemble_model_optimized.onnx")
    
    # Załadowanie modelu ensemble
    try:
        ensemble_model, feature_types = load_ensemble_model(args.model_path)
        ensemble_model.eval()  # Przełączanie modelu w tryb ewaluacji
    except Exception as e:
        print(f"Błąd podczas ładowania modelu: {e}")
        return False
    
    # Generowanie danych wejściowych
    print("\nGenerowanie przykładowych danych wejściowych...")
    dummy_input = generate_dummy_input(
        sample_rate=args.sample_rate,
        max_length=args.max_length,
        n_fft=args.n_fft,
        hop_length=args.hop_length
    )
    
    # Eksport modelu do formatu ONNX
    print("\n--- Eksport modelu do formatu ONNX ---")
    export_result = export_onnx_model(
        ensemble_model,
        onnx_path,
        dummy_input,
        opset_version=17
    )
    
    if export_result.get("success", False):
        # Optymalizacja modelu ONNX
        print("\n--- Optymalizacja modelu ONNX ---")
        optimize_result = optimize_onnx_model(onnx_path, onnx_optimized_path)
        current_model_path = onnx_optimized_path
        
        # Weryfikacja modelu ONNX
        print("\n--- Weryfikacja modelu ONNX ---")
        verify_result = verify_onnx_model(
            current_model_path,
            dummy_input,
            original_model=ensemble_model
        )
        
        if verify_result.get("success", False):
            print("\n✓ Podstawowy proces eksportu zakończony sukcesem!")
            print(f"Model ONNX zapisany w: {onnx_path}")
            print(f"Zoptymalizowany model ONNX zapisany w: {onnx_optimized_path}")
            
            # Kwantyzacja modelu ONNX jeśli wybrano
            if args.quantize:
                print(f"\n--- Kwantyzacja modelu ONNX (typ: {args.quant_type}) ---")
                
                # Ścieżka do skwantyzowanego modelu
                onnx_quantized_path = os.path.join(args.output_dir, f"ensemble_model_quantized_{args.quant_type}.onnx")
                
                if args.quant_type == "static":
                    # Przygotowanie danych kalibracyjnych dla kwantyzacji statycznej
                    print("Przygotowanie danych kalibracyjnych dla kwantyzacji statycznej...")
                    
                    try:
                        # Funkcja pomocnicza do ładowania próbek audio
                        def load_audio_samples():
                            audio_samples = []
                            
                            # Szukaj w różnych katalogach
                            audio_paths = []
                            data_dirs = ['data/test', 'data/valid', 'data_processed/test', 'data_processed/valid']
                            
                            for data_dir in data_dirs:
                                if os.path.exists(data_dir):
                                    for ext in ['.wav', '.mp3', '.ogg']:
                                        audio_files = glob.glob(f"{data_dir}/**/*{ext}", recursive=True)
                                        if audio_files:
                                            audio_paths.extend(audio_files[:args.calib_samples])
                                            if len(audio_paths) >= args.calib_samples:
                                                break
                                    if len(audio_paths) >= args.calib_samples:
                                        break
                            
                            # Jeśli nie znaleziono plików, generuj sztuczne dane
                            if not audio_paths:
                                print("Nie znaleziono plików audio. Generowanie sztucznych danych...")
                                
                                # Generowanie sztucznych danych
                                for i in range(args.calib_samples):
                                    t = np.linspace(0, args.max_length, int(args.sample_rate * args.max_length), endpoint=False)
                                    audio = np.sin(2 * np.pi * 440 * t) * 0.3  # Ton A4 (440 Hz)
                                    audio += np.sin(2 * np.pi * 880 * t) * 0.2  # Harmoniczna
                                    audio += np.random.normal(0, 0.1, size=len(t))  # Szum
                                    audio = audio / np.max(np.abs(audio))  # Normalizacja
                                    audio_samples.append(audio)
                                    print(f"Wygenerowano sztuczną próbkę audio {i+1}/{args.calib_samples}")
                                return audio_samples
                            
                            # Przetwarzanie rzeczywistych plików
                            for i, path in enumerate(audio_paths[:args.calib_samples]):
                                try:
                                    audio, sr = librosa.load(path, sr=args.sample_rate)
                                    target_length = int(args.max_length * sr)
                                    if len(audio) > target_length:
                                        audio = audio[:target_length]
                                    else:
                                        audio = np.pad(audio, (0, max(0, target_length - len(audio))), mode='constant')
                                    audio_samples.append(audio)
                                    print(f"Załadowano próbkę {i+1}/{len(audio_paths[:args.calib_samples])}: {os.path.basename(path)}")
                                except Exception as e:
                                    print(f"Błąd ładowania {path}: {e}")
                            
                            return audio_samples
                        
                        # Ładowanie próbek audio
                        audio_samples = load_audio_samples()
                        
                        if audio_samples:
                            # Utworzenie czytnika danych kalibracyjnych
                            from export_scripts.callibration_utils import EnsembleCalibrationDataReader
                            calibration_reader = EnsembleCalibrationDataReader(
                                audio_samples=audio_samples,
                                feature_types=feature_types,
                                sample_rate=args.sample_rate,
                                # n_fft=args.n_fft,
                                # hop_length=args.hop_length,
                                # n_mels=N_MELS,
                                # n_mfcc=N_MFCC,
                                # n_chroma=N_CHROMA
                            )
                            
                            # Kwantyzacja statyczna z danymi kalibracyjnymi
                            quantize_result = quantize_static_onnx_model(
                                current_model_path,
                                onnx_quantized_path,
                                calibration_reader,
                                activation_type_str="int8",  # int8 jest zalecane dla modeli CNN
                                weight_type_str="int8",      # int8 jest zalecane dla modeli CNN
                                quant_format_str="qdq",      # QDQ format jest zalecany jako domyślny
                                per_channel=False,           # Per-channel może poprawić dokładność
                                reduce_range=False           # Redukcja zakresu dla starszych CPU
                            )
                            
                            if quantize_result.get("success", False):
                                print(f"\n✓ Kwantyzacja statyczna zakończona sukcesem!")
                                print(f"Skwantyzowany model zapisany w: {onnx_quantized_path}")
                                print(f"Rozmiar skwantyzowanego modelu: {get_model_size(onnx_quantized_path):.2f} MB")
                                print(f"Redukcja rozmiaru: {quantize_result.get('quantization_reduction', 0):.2f}%")
                                
                                # Weryfikacja skwantyzowanego modelu
                                verify_quant_result = verify_onnx_model(
                                    onnx_quantized_path,
                                    dummy_input,
                                    original_model=ensemble_model
                                )
                                
                                if verify_quant_result.get("success", False):
                                    print(f"✓ Weryfikacja skwantyzowanego modelu zakończona sukcesem!")
                                    print(f"Maksymalna różnica vs PyTorch: {verify_quant_result.get('max_diff', 0):.6f}")
                                else:
                                    print(f"⚠ Weryfikacja skwantyzowanego modelu nie powiodła się: {verify_quant_result.get('error', 'Nieznany błąd')}")
                            else:
                                print(f"\n⚠ Kwantyzacja statyczna nie powiodła się: {quantize_result.get('error', 'Nieznany błąd')}")
                                print("Próba wykonania kwantyzacji dynamicznej jako fallback...")
                                
                                # Jeśli kwantyzacja statyczna nie powiodła się, próbuj dynamicznej
                                onnx_quantized_fallback_path = os.path.join(args.output_dir, "ensemble_model_quantized_dynamic_fallback.onnx")
                                quantize_dynamic_result = quantize_dynamic_onnx_model(
                                    current_model_path,
                                    onnx_quantized_fallback_path,
                                    activation_type_str="int8",
                                    weight_type_str="int8"
                                )
                                
                                if quantize_dynamic_result.get("success", False):
                                    print(f"\n✓ Kwantyzacja dynamiczna (fallback) zakończona sukcesem!")
                                    print(f"Skwantyzowany model zapisany w: {onnx_quantized_fallback_path}")
                                    print(f"Rozmiar skwantyzowanego modelu: {get_model_size(onnx_quantized_fallback_path):.2f} MB")
                                    print(f"Redukcja rozmiaru: {quantize_dynamic_result.get('quantization_reduction', 0):.2f}%")
                                else:
                                    print(f"\n⚠ Kwantyzacja dynamiczna (fallback) nie powiodła się: {quantize_dynamic_result.get('error', 'Nieznany błąd')}")
                        else:
                            print("Nie udało się przygotować danych kalibracyjnych. Próba kwantyzacji dynamicznej...")
                            # Jeśli nie ma próbek audio, próbuj kwantyzacji dynamicznej
                            quantize_result = quantize_dynamic_onnx_model(
                                current_model_path,
                                onnx_quantized_path,
                                activation_type_str="int8",
                                weight_type_str="int8"
                            )
                            
                            if quantize_result.get("success", False):
                                print(f"\n✓ Kwantyzacja dynamiczna zakończona sukcesem!")
                                print(f"Skwantyzowany model zapisany w: {onnx_quantized_path}")
                                print(f"Rozmiar skwantyzowanego modelu: {get_model_size(onnx_quantized_path):.2f} MB")
                                print(f"Redukcja rozmiaru: {quantize_result.get('quantization_reduction', 0):.2f}%")
                            else:
                                print(f"\n⚠ Kwantyzacja dynamiczna nie powiodła się: {quantize_result.get('error', 'Nieznany błąd')}")
                    
                    except Exception as e:
                        print(f"Błąd podczas kwantyzacji statycznej: {e}")
                        import traceback
                        traceback.print_exc()
                
                elif args.quant_type == "dynamic":
                    # Kwantyzacja dynamiczna
                    print("Wykonywanie kwantyzacji dynamicznej...")
                    quantize_result = quantize_dynamic_onnx_model(
                        current_model_path,
                        onnx_quantized_path,
                        activation_type_str="int8",  # int8 jest zalecane dla modeli CNN
                        weight_type_str="int8"       # int8 jest zalecane dla modeli CNN
                    )
                    
                    if quantize_result.get("success", False):
                        print(f"\n✓ Kwantyzacja dynamiczna zakończona sukcesem!")
                        print(f"Skwantyzowany model zapisany w: {onnx_quantized_path}")
                        print(f"Rozmiar skwantyzowanego modelu: {get_model_size(onnx_quantized_path):.2f} MB")
                        print(f"Redukcja rozmiaru: {quantize_result.get('quantization_reduction', 0):.2f}%")
                        
                        # Weryfikacja skwantyzowanego modelu
                        verify_quant_result = verify_onnx_model(
                            onnx_quantized_path,
                            dummy_input,
                            original_model=ensemble_model
                        )
                        
                        if verify_quant_result.get("success", False):
                            print(f"✓ Weryfikacja skwantyzowanego modelu zakończona sukcesem!")
                            print(f"Maksymalna różnica vs PyTorch: {verify_quant_result.get('max_diff', 0):.6f}")
                        else:
                            print(f"⚠ Weryfikacja skwantyzowanego modelu nie powiodła się: {verify_quant_result.get('error', 'Nieznany błąd')}")
                    else:
                        print(f"\n⚠ Kwantyzacja dynamiczna nie powiodła się: {quantize_result.get('error', 'Nieznany błąd')}")
            
            # Zapisanie metadanych modelu
            metadata = {
                "model_name": "ensemble_emotion_model",
                "feature_types": feature_types,
                "class_names": CLASS_NAMES,
                "input_shapes": {
                    "mel_input": [int(x) for x in dummy_input[0].shape],
                    "mfcc_input": [int(x) for x in dummy_input[1].shape],
                    "chroma_input": [int(x) for x in dummy_input[2].shape]
                },
                "export_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "pytorch_version": torch.__version__
            }
            
            metadata_path = os.path.join(args.output_dir, "ensemble_model_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\nMetadane modelu zapisane w: {metadata_path}")
            
            # Przykład uruchomienia inferencji na modelu ONNX
            print("\n--- Przykład inferencji na modelu ONNX ---")
            try:
                # Użyj zoptymalizowanego modelu
                final_model_path = onnx_optimized_path
                
                # Jeśli kwantyzacja się powiodła, użyj skwantyzowanego modelu
                if args.quantize and os.path.exists(onnx_quantized_path):
                    final_model_path = onnx_quantized_path
                
                session = create_onnx_inference_session(final_model_path)
                
                # Przygotowanie wejść
                test_inputs = {
                    'mel_input': dummy_input[0].numpy(),
                    'mfcc_input': dummy_input[1].numpy(),
                    'chroma_input': dummy_input[2].numpy()
                }
                
                # Uruchomienie inferencji
                outputs = session.run(None, test_inputs)
                print(f"Kształt wyjścia modelu ONNX: {outputs[0].shape}")
                print(f"Przewidywania dla pierwszej próbki: {outputs[0][0]}")
                
                return True
            except Exception as e:
                print(f"Błąd podczas inferencji ONNX: {e}")
                import traceback
                print(traceback.format_exc())
                return False
        else:
            print(f"\nWeryfikacja modelu ONNX nie powiodła się: {verify_result.get('error', 'Nieznany błąd')}")
            return False
    else:
        print(f"\nEksport modelu ONNX nie powiódł się: {export_result.get('error', 'Nieznany błąd')}")
        return False

if __name__ == "__main__":
    main()