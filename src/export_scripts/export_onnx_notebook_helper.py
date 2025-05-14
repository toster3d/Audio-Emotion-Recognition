# Skrypt pomocniczy do eksportu modelu do ONNX z poziomu notebooka

import os
import sys
import shutil
import json
import traceback
from datetime import datetime
import torch
import numpy as np
# Upewnij się, że wszystkie importy są na poziomie modułu, a nie w blokach try/except
import onnx
from onnx import shape_inference
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

# Importy lokalne (z naszego projektu)
from export_scripts.ensemble_onnx_wrapper import (
    export_onnx_model, 
    generate_dummy_input, 
    optimize_onnx_model, 
    quantize_onnx_model,
    verify_onnx_model,
    create_onnx_inference_session
)

def export_model_to_onnx(ensemble_model, output_dir=None, class_names=None, export_params=None):
    """
    Funkcja do eksportu modelu ensemble do formatu ONNX, wywoływana z notebooka.
    
    Args:
        ensemble_model: Model ensemble do wyeksportowania
        output_dir: Katalog wyjściowy (opcjonalnie)
        class_names: Lista nazw klas (opcjonalnie)
        export_params: Słownik z dodatkowymi parametrami eksportu (opcjonalnie)
            - quantize_model: Czy kwantyzować model (bool, domyślnie False)
            - quantization_type: Typ kwantyzacji ('dynamic' lub 'static', domyślnie 'dynamic')
            - quantization_dtype: Typ danych kwantyzacji ('uint8' lub 'int8', domyślnie 'uint8')
            - use_cuda: Czy używać CUDA do inferencji (None = auto-detect, domyślnie None)
        
    Returns:
        dict: Słownik z wynikami eksportu
    """
    # Ustawienie domyślnych parametrów eksportu
    if export_params is None:
        export_params = {}
    
    # Parametry eksportu z wartościami domyślnymi
    quantize_model = export_params.get("quantize_model", False)
    quantization_type = export_params.get("quantization_type", "dynamic")
    quantization_dtype = export_params.get("quantization_dtype", "uint8")
    use_cuda = export_params.get("use_cuda", None)
    
    if output_dir is None:
        # Utworzenie katalogu na modele ONNX z timestampem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"exported_models/onnx_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Katalog wyjściowy: {output_dir}")
    
    # Ścieżki do modeli ONNX
    onnx_path = os.path.join(output_dir, "ensemble_model.onnx")
    onnx_optimized_path = os.path.join(output_dir, "ensemble_model_optimized.onnx")
    onnx_quantized_path = os.path.join(output_dir, "ensemble_model_quantized.onnx")
    
    # Przełączenie modelu w tryb ewaluacji
    ensemble_model.eval()
    
    # Pobranie typów cech z modelu
    feature_types = list(ensemble_model.models.keys())
    
    # Generowanie danych wejściowych
    print("\nGenerowanie przykładowych danych wejściowych...")
    dummy_input = generate_dummy_input(
        sample_rate=22050,
        max_length=3.0,
        n_fft=2048,
        hop_length=512
    )
    
    results = {
        "success": False,
        "output_dir": output_dir,
        "onnx_path": onnx_path,
        "optimized_onnx_path": onnx_optimized_path,
        "feature_types": feature_types
    }
    
    # Eksport modelu do formatu ONNX
    print("\n--- Eksport modelu do formatu ONNX ---")
    export_result = export_onnx_model(
        ensemble_model,
        onnx_path,
        dummy_input,
        opset_version=17
    )
    
    if not export_result.get("success", False):
        results["error"] = f"Eksport modelu ONNX nie powiódł się: {export_result.get('error', 'Nieznany błąd')}"
        return results
    
    results["onnx_export_success"] = True
    
    # Optymalizacja modelu ONNX
    print("\n--- Optymalizacja modelu ONNX ---")
    optimize_result = optimize_onnx_model(onnx_path, onnx_optimized_path)
    
    if not optimize_result.get("success", False):
        results["error"] = f"Optymalizacja modelu ONNX nie powiodła się: {optimize_result.get('error', 'Nieznany błąd')}"
        # Mimo błędu optymalizacji, kontynuujemy
        results["optimization_warning"] = results["error"]
        # Używamy oryginalnego modelu
        shutil.copy(onnx_path, onnx_optimized_path)
        results["optimization_success"] = False
    else:
        results["optimization_success"] = True
        results["optimization_method"] = optimize_result.get("method", "unknown")
        results["optimization_reduction"] = optimize_result.get("reduction_percent", 0)
    
    # Sprawdź czy optymalizacja faktycznie zmniejszyła rozmiar modelu
    original_size = os.path.getsize(onnx_path)
    optimized_size = os.path.getsize(onnx_optimized_path)
    
    # Wybierz odpowiedni model do kwantyzacji - jeśli optymalizacja drastycznie zwiększyła rozmiar,
    # użyj oryginalnego modelu zamiast zoptymalizowanego
    if optimized_size > 1.5 * original_size:  # Więcej niż 50% zwiększenie rozmiaru
        model_to_quantize = onnx_path
        print(f"\nUwaga: Zoptymalizowany model jest większy ({optimized_size / (1024*1024):.2f} MB) niż oryginalny ({original_size / (1024*1024):.2f} MB).")
        print("Użycie oryginalnego modelu do kwantyzacji zamiast zoptymalizowanego.")
        results["quantization_source"] = "original"
    else:
        model_to_quantize = onnx_optimized_path
        results["quantization_source"] = "optimized"
    
    # Kwantyzacja modelu (opcjonalna)
    if quantize_model:
        print("\n--- Kwantyzacja modelu ONNX ---")
        try:
            # Użyj ulepszonej funkcji kwantyzacji:
            quantize_result = quantize_onnx_model(
                model_to_quantize, 
                onnx_quantized_path,
                quantization_type=quantization_type,
                dtype=quantization_dtype
            )
            
            if quantize_result.get("success", False):
                results["quantization_success"] = True
                results["quantized_onnx_path"] = onnx_quantized_path
                results["quantization_reduction"] = quantize_result.get("reduction_percent", 0)
                results["quantization_method"] = quantize_result.get("method", "unknown")
                print(f"Kwantyzacja zakończona sukcesem. Redukcja rozmiaru: {results['quantization_reduction']:.2f}%")
            else:
                error_msg = f"Kwantyzacja modelu nie powiodła się: {quantize_result.get('error', 'Nieznany błąd')}"
                print(f"Ostrzeżenie: {error_msg}")
                print("Kontynuowanie bez kwantyzacji.")
                results["quantization_warning"] = error_msg
                
        except Exception as e:
            # Zapisz szczegółowe informacje o błędzie
            error_msg = f"Nieoczekiwany błąd podczas kwantyzacji: {e}"
            tb = traceback.format_exc()
            print(f"Ostrzeżenie: {error_msg}")
            print(f"Szczegóły błędu: {tb}")
            results["quantization_warning"] = error_msg
            print("Kontynuowanie bez kwantyzacji.")
    
    # Wybierz model do weryfikacji - użyj skwantyzowanego jeśli dostępny,
    # w przeciwnym razie zoptymalizowanego, lub oryginalnego jeśli optymalizacja zawiodła
    verify_model_path = None
    
    if quantize_model and results.get("quantization_success", False):
        verify_model_path = onnx_quantized_path
    elif results.get("optimization_success", False) and optimized_size <= 1.5 * original_size:
        verify_model_path = onnx_optimized_path
    else:
        verify_model_path = onnx_path
    
    # Weryfikacja modelu ONNX
    print("\n--- Weryfikacja modelu ONNX ---")
    try:
        # Utwórz sesję ONNX Runtime i wykonaj przykładową inferencję
        session = create_onnx_inference_session(verify_model_path, use_cuda=use_cuda)
        
        if session is None:
            results["warning"] = "Nie udało się utworzyć sesji ONNX Runtime dla modelu."
            print(f"Ostrzeżenie: {results['warning']}")
        else:
            # Weryfikacja pełna z porównaniem z modelem PyTorch
            verify_result = verify_onnx_model(
                verify_model_path,
                dummy_input,
                original_model=ensemble_model
            )
            
            results["verification_success"] = verify_result.get("success", False)
            results["max_diff"] = verify_result.get("max_diff", float('inf'))
            
            if not results["verification_success"]:
                results["warning"] = f"Weryfikacja modelu nie powiodła się: {verify_result.get('error', 'Nieznany błąd')}"
                print(f"Ostrzeżenie: {results['warning']}")
            else:
                print(f"Weryfikacja modelu zakończona powodzeniem")
                print(f"Maksymalna różnica między PyTorch i ONNX: {results['max_diff']:.6f}")
    except Exception as e:
        results["warning"] = f"Błąd podczas weryfikacji modelu: {e}"
        print(f"Ostrzeżenie: {results['warning']}")
    
    # Zapisz metadane modelu
    metadata = {
        "feature_types": feature_types,
        "class_names": class_names,
        "input_shapes": {
            "mel_input": [1, 1, 128, None],  # None oznacza wymiar dynamiczny
            "mfcc_input": [1, 1, 40, None],
            "chroma_input": [1, 1, 12, None]
        },
        "export_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "pytorch_version": torch.__version__,
        "export_params": export_params,
        "optimization_reduction": results.get("optimization_reduction", 0),
        "quantization_reduction": results.get("quantization_reduction", 0)
    }
    
    metadata_path = os.path.join(output_dir, "ensemble_model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\nMetadane modelu zapisane w: {metadata_path}")
    results["metadata_path"] = metadata_path
    
    # Podsumowanie eksportu
    print("\nEksport modelu ensemble do ONNX zakończony sukcesem!")
    print(f"Model ONNX zapisany w: {onnx_path}")
    print(f"Zoptymalizowany model ONNX zapisany w: {onnx_optimized_path}")
    
    # Raportuj rozmiary plików
    original_size_mb = os.path.getsize(onnx_path) / (1024*1024)
    optimized_size_mb = os.path.getsize(onnx_optimized_path) / (1024*1024)
    print(f"Rozmiar oryginalnego modelu: {original_size_mb:.2f} MB")
    print(f"Rozmiar zoptymalizowanego modelu: {optimized_size_mb:.2f} MB")
    
    size_reduction = ((original_size_mb - optimized_size_mb) / original_size_mb) * 100 if original_size_mb > 0 else 0
    print(f"Redukcja rozmiaru po optymalizacji: {size_reduction:.2f}%")
    
    if quantize_model:
        if results.get("quantization_success", False):
            quantized_size_mb = os.path.getsize(onnx_quantized_path) / (1024*1024)
            print(f"Rozmiar skwantyzowanego modelu: {quantized_size_mb:.2f} MB")
            quant_reduction = ((original_size_mb - quantized_size_mb) / original_size_mb) * 100 if original_size_mb > 0 else 0
            print(f"Redukcja rozmiaru po kwantyzacji: {quant_reduction:.2f}%")
        else:
            print("Kwantyzacja nie została przeprowadzona w ramach procesu eksportu.")
    
    print(f"Metadane modelu zapisane w: {metadata_path}")
    
    if results.get("verification_success", False):
        print("\nWeryfikacja modelu zakończona powodzeniem")
        print(f"Maksymalna różnica między PyTorch i ONNX: {results.get('max_diff', 0):.6f}")
    
    # Dodaj instrukcje dla użytkownika
    print("\nAby użyć wyeksportowanego modelu ONNX:")
    print("1. Zainstaluj onnxruntime: pip install onnxruntime")
    print("2. Do inferencji na GPU zainstaluj: pip install onnxruntime-gpu")
    print("3. Skorzystaj z przykładowego kodu w pliku example_usage.py")
    
    results["success"] = True
    return results 