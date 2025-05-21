# Skrypt pomocniczy do eksportu modelu do ONNX z poziomu notebooka

import os
import shutil
import json
from datetime import datetime
import torch
import onnxruntime as ort

# Importy lokalne (z naszego projektu)
from export_scripts.ensemble_onnx_wrapper import (
    export_onnx_model as export_onnx_model_core,
    generate_dummy_input, 
    optimize_onnx_model, 
    quantize_static_onnx_model,
    quantize_dynamic_onnx_model,
    verify_onnx_model,
    create_onnx_inference_session,
    get_model_size
)
from config import N_MELS, N_MFCC, N_CHROMA

def export_model_to_onnx(ensemble_model, output_dir=None, class_names=None, export_params=None, calibration_data_reader=None):
    """
    Funkcja do eksportu modelu ensemble do formatu ONNX, wywoływana z notebooka.
    
    Args:
        ensemble_model: Model PyTorch do eksportu
        output_dir: Katalog wyjściowy dla modelu ONNX
        class_names: Lista nazw klas
        export_params: Parametry eksportu i kwantyzacji
        calibration_data_reader: Obiekt CalibrationDataReader dla kwantyzacji statycznej
    
    Returns:
        dict: Słownik z wynikami eksportu
    """
    # Sprawdź parametry eksportu
    if export_params is None:
        export_params = {}
    
    # Parametry kwantyzacji
    quantize_model = export_params.get("quantize_model", True)
    quantization_type = export_params.get("quantization_type", "static")  # Domyślnie static dla modeli CNN
    activation_type = export_params.get("activation_type", "int8")        # Domyślnie int8 dla lepszej dokładności
    weight_type = export_params.get("weight_type", "int8")                # Domyślnie int8
    quant_format = export_params.get("quant_format", "qdq")               # QDQ format jest zalecany jako domyślny
    per_channel = export_params.get("per_channel", False)                 # Per-channel może poprawić dokładność
    reduce_range = export_params.get("reduce_range", False)               # Redukcja zakresu do 7-bitów dla starszych CPU
    use_cuda = export_params.get("use_cuda", None)                        # Użycie CUDA, jeśli jest dostępne
    opset_version = export_params.get("opset_version", 19)                # Wersja opset dla ONNX
    run_onnx_optimizer_flag = export_params.get("run_onnx_optimizer", False)  # Czy uruchomić onnxoptimizer
    
    # Sprawdź, czy mamy oczekiwane dane kalibracyjne dla kwantyzacji statycznej
    if quantize_model and quantization_type == "static" and calibration_data_reader is None:
        print(f"OSTRZEŻENIE: Brak danych kalibracyjnych dla kwantyzacji statycznej. Kwantyzacja może być niedokładna.")
    
    # Utwórz katalog wyjściowy
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"exported_models/onnx_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Katalog wyjściowy dla eksportu: {output_dir}")
    
    # Ścieżki do plików modeli
    onnx_path = os.path.join(output_dir, "ensemble_model.onnx")
    onnx_optimized_path = os.path.join(output_dir, "ensemble_model_optimized.onnx")
    onnx_quantized_path = os.path.join(output_dir, f"ensemble_model_quantized_{quantization_type}.onnx")
    
    # Przełącz model w tryb ewaluacji
    ensemble_model.eval()
    feature_types = list(ensemble_model.models.keys()) if hasattr(ensemble_model, 'models') else []
    
    # Generowanie danych testowych
    print("\nGenerowanie przykładowych danych wejściowych (dummy input)...")
    dummy_input = generate_dummy_input(sample_rate=22050, max_length=3.0, n_fft=2048, hop_length=512)
    
    # Przygotowanie słownika wyników
    results = {
        "success": False,
        "output_dir": output_dir,
        "onnx_path": onnx_path,
        "optimized_onnx_path": onnx_path,
        "feature_types": feature_types,
        "quantized_onnx_path": None,
        "sizes_mb": {}
    }
    
    # 1. Eksport modelu do formatu ONNX
    print("\n--- Krok 1: Eksport modelu do formatu ONNX ---")
    export_result = export_onnx_model_core(
        ensemble_model, 
        onnx_path, 
        dummy_input, 
        opset_version=opset_version
    )
    
    if not export_result.get("success", False):
        results["error"] = f"Eksport modelu ONNX nie powiódł się: {export_result.get('error', 'Nieznany błąd')}"
        print(results["error"])
        return results
    
    results["onnx_export_success"] = True
    results["sizes_mb"]["original"] = get_model_size(onnx_path)
    print(f"Model ONNX wyeksportowany pomyślnie do: {onnx_path}")
    print(f"Rozmiar oryginalnego modelu: {results['sizes_mb']['original']:.2f} MB")

    # 2. Optymalizacja modelu ONNX (opcjonalna)
    current_model_for_next_step = onnx_path # Domyślnie używamy oryginalnego wyeksportowanego modelu
    results["optimized_onnx_path"] = onnx_path # Jeśli optymalizacja nie jest uruchamiana, "zoptymalizowany" to oryginał

    if run_onnx_optimizer_flag:
        print("\n--- Krok 2: Optymalizacja modelu ONNX (onnxoptimizer) ---")
        optimize_result = optimize_onnx_model(onnx_path, onnx_optimized_path)
        
        if not optimize_result.get("success", False):
            results["error_optimization"] = f"Optymalizacja modelu ONNX (onnxoptimizer) nie powiodła się: {optimize_result.get('error', 'Nieznany błąd')}"
            print(f"Ostrzeżenie: {results['error_optimization']}")
            # W przypadku niepowodzenia optymalizacji, używamy oryginalnego modelu ONNX
            # Kopiujemy go do ścieżki zoptymalizowanej dla spójności, jeśli ta ścieżka jest inna
            if onnx_path != onnx_optimized_path:
                 try:
                     shutil.copy(onnx_path, onnx_optimized_path)
                     print(f"Skopiowano oryginalny model {onnx_path} do {onnx_optimized_path} z powodu błędu optymalizacji.")
                 except Exception as e_copy:
                     print(f"Błąd podczas kopiowania {onnx_path} do {onnx_optimized_path}: {e_copy}")
                     # Jeśli kopiowanie się nie powiedzie, a ścieżki są różne, zoptymalizowana ścieżka może nie istnieć.
                     # Ustawiamy ją na None, aby uniknąć problemów w dalszych krokach.
                     results["optimized_onnx_path"] = None 

            results["optimization_success"] = False
        else:
            results["optimization_success"] = True
            results["optimization_method"] = optimize_result.get("method", "onnxoptimizer")
            results["optimization_reduction"] = optimize_result.get("reduction_percent", 0)
            current_model_for_next_step = onnx_optimized_path
            results["optimized_onnx_path"] = onnx_optimized_path # Aktualizujemy, bo optymalizacja się powiodła
            print(f"Model ONNX zoptymalizowany (onnxoptimizer) pomyślnie. Wynik w: {onnx_optimized_path}")
    else:
        print("\n--- Krok 2: Optymalizacja modelu ONNX (onnxoptimizer) POMINIĘTA ---")
        results["optimization_success"] = None # Oznaczamy, że nie było próby
        results["optimization_method"] = "skipped"
        # current_model_for_next_step pozostaje onnx_path
        # results["optimized_onnx_path"] już ustawione na onnx_path

    # Aktualizacja informacji o rozmiarze "zoptymalizowanego" modelu
    # (który może być oryginalnym, jeśli optymalizacja była pominięta lub nieudana)
    if results["optimized_onnx_path"] and os.path.exists(results["optimized_onnx_path"]):
        results["sizes_mb"]["optimized"] = get_model_size(results["optimized_onnx_path"])
        print(f"Rozmiar modelu używanego do dalszych kroków ('zoptymalizowany'): {results['sizes_mb']['optimized']:.2f} MB ({os.path.basename(results['optimized_onnx_path'])})")
    elif results["optimized_onnx_path"] is None and run_onnx_optimizer_flag : # Optymalizacja miała być, ale ścieżka jest None
        print(f"Ostrzeżenie: Zoptymalizowany model ({onnx_optimized_path}) nie istnieje po próbie optymalizacji i błędzie kopiowania. Używam oryginalnego {onnx_path}")
        current_model_for_next_step = onnx_path
        results["optimized_onnx_path"] = onnx_path # Fallback na oryginalny
        results["sizes_mb"]["optimized"] = results["sizes_mb"]["original"]
    elif not run_onnx_optimizer_flag:
        results["sizes_mb"]["optimized"] = results["sizes_mb"]["original"]
        print(f"Rozmiar modelu używanego do dalszych kroków ('zoptymalizowany' - oryginał): {results['sizes_mb']['optimized']:.2f} MB ({os.path.basename(results['optimized_onnx_path'])})")
    else: # run_onnx_optimizer_flag is True, ale optimized_onnx_path jest None (błąd kopiowania i ścieżka inna)
        print(f"Ostrzeżenie: Brak pliku dla zoptymalizowanego modelu. Rozmiar 'optimized' nie został ustawiony.")
        # W tym przypadku current_model_for_next_step powinien być nadal onnx_path, jeśli on istnieje
        if not os.path.exists(current_model_for_next_step):
            print(f"BŁĄD KRYTYCZNY: Model {current_model_for_next_step} nie istnieje dla dalszych kroków!")
            results["error"] = f"Model {current_model_for_next_step} nie istnieje po nieudanej optymalizacji."
            return results


    # Sprawdzenie czy optymalizacja (jeśli była uruchomiona) nie zwiększyła znacznie rozmiaru modelu
    if run_onnx_optimizer_flag and results.get("optimization_success") and results["sizes_mb"]["optimized"] > 1.5 * results["sizes_mb"]["original"]:
        print(f"\nUwaga: Zoptymalizowany model (onnxoptimizer) jest znacznie większy ({results['sizes_mb']['optimized']:.2f} MB) niż oryginalny ({results['sizes_mb']['original']:.2f} MB).")

    # 3. Kwantyzacja modelu ONNX
    if quantize_model:
        print(f"\n--- Krok 3: Kwantyzacja modelu ONNX ({quantization_type}) ---")
        
        if quantization_type == "static":
            # Sprawdź czy mamy dane kalibracyjne
            if calibration_data_reader is None:
                print("Ostrzeżenie: Brak danych kalibracyjnych dla kwantyzacji statycznej.")
                print("Dla najlepszych wyników, dostarcz CalibrationDataReader z próbkami audio.")
                print("Próbuję wykonać kwantyzację dynamiczną jako fallback...")
                
                # Jako fallback, spróbuj kwantyzacji dynamicznej
                quantize_result = quantize_dynamic_onnx_model(
                    current_model_for_next_step,
                    onnx_quantized_path,
                    activation_type_str=activation_type,
                    weight_type_str=weight_type
                )
                results["quantization_type_used"] = "dynamic (fallback)"
            else:
                print(f"Używam kwantyzacji statycznej. Model bazowy: {current_model_for_next_step}")
                print(f"Format kwantyzacji: {quant_format}, Per-channel: {per_channel}, Reduce range: {reduce_range}")
                
                # Użyj dostarczonego czytnika danych kalibracyjnych
                quantize_result = quantize_static_onnx_model(
                    current_model_for_next_step, 
                    onnx_quantized_path,
                    calibration_data_reader,
                    activation_type_str=activation_type,
                    weight_type_str=weight_type,
                    quant_format_str=quant_format,
                    per_channel=per_channel,
                    reduce_range=reduce_range
                )
                results["quantization_type_used"] = "static"
                
            # Przetwarzanie wyników kwantyzacji
            if quantize_result.get("success", False):
                results["quantization_success"] = True
                results["quantized_onnx_path"] = onnx_quantized_path
                results["quantization_reduction_vs_source"] = quantize_result.get("quantization_reduction", 0)
                results["quantization_params_used"] = quantize_result.get("params_used")
                results["sizes_mb"]["quantized"] = get_model_size(onnx_quantized_path)
                
                print(f"Kwantyzacja {results['quantization_type_used']} zakończona sukcesem.")
                print(f"Model w: {results['quantized_onnx_path']}")
                print(f"Rozmiar modelu: {results['sizes_mb']['quantized']:.2f} MB")
                print(f"Redukcja rozmiaru względem modelu wejściowego: {results['quantization_reduction_vs_source']:.2f}%")
            else:
                error_msg = f"Kwantyzacja {results['quantization_type_used']} nie powiodła się: {quantize_result.get('error', 'Nieznany błąd')}"
                print(f"Ostrzeżenie: {error_msg}")
                results["quantization_error"] = error_msg
                results["quantization_success"] = False
        else:
            # Kwantyzacja dynamiczna
            print(f"Używam kwantyzacji dynamicznej. Model bazowy: {current_model_for_next_step}")
            quantize_result = quantize_dynamic_onnx_model(
                current_model_for_next_step,
                onnx_quantized_path,
                activation_type_str=activation_type,
                weight_type_str=weight_type
            )
            results["quantization_type_used"] = "dynamic"
            
            if quantize_result.get("success", False):
                results["quantization_success"] = True
                results["quantized_onnx_path"] = onnx_quantized_path
                results["quantization_reduction_vs_source"] = quantize_result.get("quantization_reduction", 0)
                results["sizes_mb"]["quantized"] = get_model_size(onnx_quantized_path)
                
                print(f"Kwantyzacja dynamiczna zakończona sukcesem. Model w: {results['quantized_onnx_path']}")
                print(f"Redukcja rozmiaru: {results['quantization_reduction_vs_source']:.2f}%")
            else:
                error_msg = f"Kwantyzacja dynamiczna nie powiodła się: {quantize_result.get('error', 'Nieznany błąd')}"
                print(f"Ostrzeżenie: {error_msg}")
                results["quantization_error"] = error_msg
                results["quantization_success"] = False
    else:
        print("\nKwantyzacja pominięta zgodnie z konfiguracją.")
    
    # 4. Weryfikacja modelu ONNX
    # Wybierz, który model weryfikować (skwantyzowany, jeśli dostępny, w przeciwnym razie zoptymalizowany)
    verify_model_path = current_model_for_next_step
    if quantize_model and results.get("quantization_success", False) and results.get("quantized_onnx_path"):
        verify_model_path = results["quantized_onnx_path"]
        print(f"\nModel wybrany do weryfikacji: SKWANTYZOWANY ({verify_model_path})")
    else:
        print(f"\nModel wybrany do weryfikacji: NIESKWANTYZOWANY ({verify_model_path})")
    
    print("\n--- Krok 4: Weryfikacja finalnego modelu ONNX ---")
    try:
        # Utwórz sesję inferencyjną
        session = create_onnx_inference_session(verify_model_path, use_cuda=use_cuda)
        if session is None:
            results["verification_error"] = "Nie udało się utworzyć sesji ONNX Runtime dla weryfikacji."
            print(f"Ostrzeżenie: {results['verification_error']}")
            results["verification_success"] = False
        else:
            # Weryfikacja modelu
            verify_result = verify_onnx_model(verify_model_path, dummy_input, original_model=ensemble_model)
            results["verification_success"] = verify_result.get("success", False)
            results["max_diff"] = verify_result.get("max_diff", float('inf'))
            results["verification_result"] = verify_result
            
            if not results["verification_success"]:
                results["verification_error"] = f"Weryfikacja modelu nie powiodła się: {verify_result.get('error', 'Nieznany błąd')}"
                if verify_result.get("error_comparison"):
                    results["verification_error"] += f" (Błąd porównania: {verify_result.get('error_comparison')})"
                print(f"Ostrzeżenie: {results['verification_error']}")
            else:
                print(f"Weryfikacja modelu zakończona. Max różnica vs PyTorch: {results['max_diff']:.6f}")
                if verify_result.get("has_warning"):
                    results["verification_warning"] = verify_result.get("warning", "Wystąpiło ostrzeżenie podczas weryfikacji.")
                    print(f"Ostrzeżenie weryfikacji: {results['verification_warning']}")
    
    except Exception as e_verify:
        results["verification_error"] = f"Błąd podczas weryfikacji modelu: {str(e_verify)}"
        print(f"Ostrzeżenie: {results['verification_error']}")
        results["verification_success"] = False
    
    # 5. Tworzenie metadanych
    # Wybierz ścieżkę do modelu dla metadanych
    final_model_path = results.get("quantized_onnx_path", results["optimized_onnx_path"])
    
    # Przygotuj metadane modelu
    metadata = {
        "model_name": "audio_emotion_ensemble",
        "original_pytorch_model_class": ensemble_model.__class__.__name__,
        "feature_types": feature_types,
        "class_names": class_names if class_names else "Nie podano",
        "input_shapes_onnx": {
            "mel_input": [1, 1, N_MELS, -1], 
            "mfcc_input": [1, 1, N_MFCC, -1],
            "chroma_input": [1, 1, N_CHROMA, -1]
        },
        "pytorch_version": torch.__version__,
        "onnx_opset_version": opset_version,
        "onnxruntime_version": ort.__version__,
        "export_parameters_used": export_params,
        "paths": {
            "original_onnx": os.path.basename(onnx_path) if onnx_path else None,
            "optimized_onnx": os.path.basename(results["optimized_onnx_path"]) if results.get("optimized_onnx_path") else None,
            "quantized_onnx": os.path.basename(results["quantized_onnx_path"]) if results.get("quantized_onnx_path") else None,
            "final_verified_model": os.path.basename(verify_model_path) if verify_model_path else None,
        },
        "sizes_mb": results["sizes_mb"],
        "optimization_details": {
            "success": results.get("optimization_success"),
            "method": results.get("optimization_method"),
            "reduction_percent": results.get("optimization_reduction"),
            "warning": results.get("optimization_warning"),
            "error": results.get("error_optimization")
        },
        "quantization_details": {
            "performed": quantize_model,
            "type": results.get("quantization_type_used", quantization_type) if quantize_model else None,
            "success": results.get("quantization_success"),
            "source_model_for_quantization": results.get("quantization_source_model"),
            "reduction_vs_source_percent": results.get("quantization_reduction_vs_source"),
            "params_used": {
                "activation_type": activation_type,
                "weight_type": weight_type,
                "quant_format": quant_format if quantization_type == "static" else "dynamic",
                "per_channel": per_channel if quantization_type == "static" else False,
                "reduce_range": reduce_range if quantization_type == "static" else False
            },
            "error": results.get("quantization_error")
        },
        "verification_details": {
            "model_verified": os.path.basename(verify_model_path) if verify_model_path else None,
            "success": results.get("verification_success"),
            "max_difference_vs_pytorch": results.get("max_diff"),
            "warning": results.get("verification_warning"),
            "error": results.get("verification_error")
        }
    }
    
    # Zapisz metadane do pliku JSON
    metadata_filename = f"metadata_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    metadata_path = os.path.join(output_dir, metadata_filename)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4, default=lambda o: str(o))
    
    results["metadata_path"] = metadata_path
    
    # Podsumowanie procesu eksportu
    if results.get("onnx_export_success", False) and results.get("optimization_success", False) and (not quantize_model or results.get("quantization_success", False)):
        results["success"] = True
    else:
        results["success"] = False
    
    return results

# Pomocnicza funkcja do pobierania rozmiaru modelu
def get_model_size(model_path):
    """Zwraca rozmiar modelu w MB."""
    if model_path and os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)  # Konwersja bajtów na MB
    return 0.0 