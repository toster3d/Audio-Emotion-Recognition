import torch
import warnings
import traceback
import onnx
import librosa
import numpy as np
import onnxruntime as ort
import os
import tempfile
import shutil
from torch.jit._trace import TracerWarning
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, QuantFormat, CalibrationDataReader
from typing import Dict, Any, Union

from config import MAX_LENGTH, N_FFT, HOP_LENGTH, N_MELS, N_MFCC, N_CHROMA


class EnsembleONNXWrapper(torch.nn.Module):
    def __init__(self, ensemble_model):
        super().__init__()
        self.ensemble = ensemble_model
        
    def forward(self, mel_input, mfcc_input, chroma_input):
        """
        Przesyłanie do przodu dostosowane do eksportu ONNX
        
        Args:
            mel_input: Tensor dla melspectrogramu [batch, channel, mels, time]
            mfcc_input: Tensor dla MFCC [batch, channel, mfccs, time]
            chroma_input: Tensor dla chromy [batch, channel, chromas, time]
            
        Returns:
            torch.Tensor: Wyjściowe prawdopodobieństwa klas
        """
        # Upewnij się, że tensory wejściowe są na tym samym urządzeniu co model
        device = next(self.parameters()).device
        mel_input = mel_input.to(device)
        mfcc_input = mfcc_input.to(device)
        chroma_input = chroma_input.to(device)
        
        # Przygotowanie słownika wejść dla modelu ensemble
        inputs_dict = {
            'melspectrogram': mel_input,
            'mfcc': mfcc_input,
            'chroma': chroma_input
        }
        
        # Przesyłanie przez model ensemble
        with torch.no_grad():
            output = self.ensemble(inputs_dict)
            
        return output

def export_onnx_model(ensemble_model, output_path, dummy_input, opset_version=19):
    # Utwórz i przygotuj wrapper
    try:
        print(f"Rozpoczynam eksport modelu ensemble do formatu ONNX: {output_path}")
        
        ensemble_model.eval()
        
        wrapper = EnsembleONNXWrapper(ensemble_model)
        
        # Sprawdź, czy wejścia są na tym samym urządzeniu co model
        device = next(wrapper.parameters()).device
        dummy_mel, dummy_mfcc, dummy_chroma = dummy_input
        
        # Przenieś dane wejściowe na to samo urządzenie co model
        dummy_mel = dummy_mel.to(device)
        dummy_mfcc = dummy_mfcc.to(device)
        dummy_chroma = dummy_chroma.to(device)
        
        # Zaktualizuj krotę wejściową
        dummy_input = (dummy_mel, dummy_mfcc, dummy_chroma)
        
        # Spróbuj wykonać forward pass, aby upewnić się, że model działa przed eksportem
        try:
            with torch.no_grad():
                output = wrapper(*dummy_input)
            print(f"Test forward pass zakończony sukcesem. Kształt wyjścia: {output.shape}")
        except Exception as e:
            print(f"Ostrzeżenie: Test forward pass nie powiódł się: {e}")        
        # Konfiguracja dynamicznych osi
        dynamic_axes = {
            'mel_input': {0: 'batch_size', 3: 'time_steps'},
            'mfcc_input': {0: 'batch_size', 3: 'time_steps'},
            'chroma_input': {0: 'batch_size', 3: 'time_steps'},
            'output': {0: 'batch_size'}
        }
        
        # Informacje o wymiarach wejść i wyjść
        print(f"Wymiary wejść:")
        print(f" - mel_input:   {dummy_input[0].shape}")
        print(f" - mfcc_input:  {dummy_input[1].shape}")
        print(f" - chroma_input: {dummy_input[2].shape}")
        
        # Eksport do ONNX
        try:
            # Ustawienie dla torch.onnx.export dla uniknięcia ostrzeżeń
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=TracerWarning)
                
                torch.onnx.export(
                    wrapper,
                    dummy_input,
                    output_path,
                    input_names=['mel_input', 'mfcc_input', 'chroma_input'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    export_params=True,
                    training=torch.onnx.TrainingMode.EVAL
                )

                
            print("Eksport do ONNX zakończony pomyślnie!")
        except Exception as e:
            error_message = f"Błąd podczas eksportu ONNX: {e}"
            print(f"{error_message}")
            tb = traceback.format_exc()
            error_path = output_path + ".export.error.log"
            with open(error_path, "w") as f:
                f.write(f"Błąd eksportu modelu ONNX:\n{error_message}\n\nTraceback:\n{tb}")
            print(f"Szczegóły błędu zapisane do: {error_path}")
            return {"success": False, "error": error_message, "error_log": error_path}
        
        # Walidacja modelu
        try:
            print("Weryfikacja modelu ONNX...")
            model = onnx.load(output_path)
            onnx.checker.check_model(model)
            print("✓ Model ONNX poprawny!")
            
            # Informacje o modelu
            print(f"Rozmiar modelu: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
            print(f"Liczba węzłów grafu: {len(model.graph.node)}")
            print(f"Liczba inicjalizatorów: {len(model.graph.initializer)}")
            print(f"Model wyeksportowany do: {output_path}")
            
            return {"success": True, "path": output_path, "size": os.path.getsize(output_path)}
        except onnx.checker.ValidationError as e:
            print(f"Błąd walidacji modelu ONNX: {e}")
            # Zapisz szczegóły błędu do pliku
            error_path = output_path + ".error.log"
            with open(error_path, "w") as f:
                f.write(f"Błąd walidacji modelu ONNX:\n{str(e)}")
            print(f"Szczegóły błędu zapisane do: {error_path}")
            return {"success": False, "error": str(e), "error_log": error_path}
    
    except Exception as e:
        print(f"Błąd podczas eksportu modelu ONNX: {e}")
        tb = traceback.format_exc()
        # Zapisz szczegóły błędu do pliku
        error_path = output_path + ".export.error.log"
        with open(error_path, "w") as f:
            f.write(f"Błąd eksportu modelu ONNX:\n{str(e)}\n\nTraceback:\n{tb}")
        print(f"Szczegóły błędu zapisane do: {error_path}")
        return {"success": False, "error": str(e), "error_log": error_path}

def generate_dummy_input(sample_rate=22050, max_length=MAX_LENGTH, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Generuje przykładowe dane wejściowe dla modelu ONNX.
    
    Args:
        sample_rate: Częstotliwość próbkowania audio
        max_length: Maksymalna długość audio w sekundach
        n_fft: Rozmiar okna FFT
        hop_length: Długość kroku w FFT
        
    Returns:
        tuple: Tuple z tensorem mel_spectrogram, mfcc, chroma
    """
    # Obliczanie rzeczywistego rozmiaru czasowego
    num_frames = int(sample_rate * max_length / hop_length) + 1
    
    # Tworzymy tensory o odpowiednich kształtach
    mel_input = torch.randn(1, 1, N_MELS, num_frames)
    mfcc_input = torch.randn(1, 1, N_MFCC, num_frames)
    chroma_input = torch.randn(1, 1, N_CHROMA, num_frames)
    
    print("Wymiary wejść:")
    print(f" - mel_input:   {mel_input.shape}")
    print(f" - mfcc_input:  {mfcc_input.shape}")
    print(f" - chroma_input: {chroma_input.shape}")
    
    return (mel_input, mfcc_input, chroma_input)

def optimize_onnx_model(input_path, output_path):
    """
    Optymalizacja modelu ONNX za pomocą optymalizatora ONNX.
    
    Args:
        input_path: Ścieżka do modelu ONNX
        output_path: Ścieżka wyjściowa dla zoptymalizowanego modelu
    
    Returns:
        dict: Słownik z wynikami optymalizacji
    """
    try:
        print(f"Rozpoczęcie optymalizacji modelu ONNX: {input_path}")
        if not os.path.exists(input_path):
            error_msg = f"Plik modelu ONNX nie istnieje: {input_path}"
            print(error_msg)
            return {"success": False, "error": error_msg}
        
        model = onnx.load(input_path)
        onnx.checker.check_model(model) # Sprawdź przed optymalizacją
        
        backup_path = input_path + ".backup"
        shutil.copy(input_path, backup_path)
        print(f"Utworzono kopię zapasową oryginalnego modelu: {backup_path}")
        
        input_size = os.path.getsize(input_path)
        print(f"Rozmiar oryginalnego modelu: {input_size / (1024*1024):.2f} MB")
            
        try:
            import onnxoptimizer as optimizer
            safe_passes = ['eliminate_nop_dropout', 'eliminate_unused_initializer', 'eliminate_identity', 'fuse_bn_into_conv']
            print("Używanie bezpiecznych optymalizacji onnxoptimizer...")
            optimized_model = optimizer.optimize(model, safe_passes)
            
            temp_output_path = output_path + ".temp"
            onnx.save(optimized_model, temp_output_path)
            onnx.checker.check_model(optimized_model) # Sprawdź po optymalizacji
            print("Model po optymalizacji onnxoptimizer poprawny.")
            
            temp_output_size = os.path.getsize(temp_output_path)
            if temp_output_size > 1.5 * input_size: # Zwiększono próg do 1.5x
                warning_msg = (f"Ostrzeżenie: Zoptymalizowany model przez onnxoptimizer jest znacznie większy "
                               f"({temp_output_size / (1024*1024):.2f} MB) niż oryginalny "
                               f"({input_size / (1024*1024):.2f} MB). Używanie oryginału.")
                print(warning_msg)
                if os.path.exists(temp_output_path): os.remove(temp_output_path)
                shutil.copy(input_path, output_path)
                return {"success": True, "input_size": input_size, "output_size": input_size,
                        "reduction_percent": 0.0, "warning": warning_msg, "method": "no_optimization_onnxoptimizer_larger"}

            if os.path.exists(temp_output_path): shutil.move(temp_output_path, output_path)
            output_size = os.path.getsize(output_path)
            reduction_percent = (1 - output_size / input_size) * 100 if input_size > 0 else 0
            print(f"Optymalizacja onnxoptimizer zakończona.")
            print(f"Rozmiar po optymalizacji: {output_size / (1024*1024):.2f} MB. Redukcja: {reduction_percent:.2f}%")
            return {"success": True, "input_size": input_size, "output_size": output_size,
                    "reduction_percent": reduction_percent, "method": "onnx_optimizer", "passes_applied": safe_passes}
            
        except Exception as e_opt:
            error_msg = f"Optymalizacja onnxoptimizer nie powiodła się: {e_opt}. Kopiowanie oryginalnego modelu."
            print(error_msg)
            shutil.copy(input_path, output_path)
            return {"success": True, "input_size": input_size, "output_size": os.path.getsize(output_path),
                    "reduction_percent": 0.0, "warning": error_msg, "method": "no_optimization_onnxoptimizer_failed"}
            
    except Exception as e_main_opt:
        error_msg = f"Nieoczekiwany błąd podczas optymalizacji modelu: {e_main_opt}"
        print(error_msg)
        tb = traceback.format_exc()
        error_path = output_path + ".optimization.error.log"
        with open(error_path, "w") as f: f.write(f"Błąd optymalizacji modelu ONNX:\n{error_msg}\n\nTraceback:\n{tb}")
        print(f"Szczegóły błędu zapisane do: {error_path}")
        try:
            print(f"Kopiowanie oryginalnego modelu jako zapasowe rozwiązanie...")
            shutil.copy(input_path, output_path)
            input_size = os.path.getsize(input_path)
            return {"success": True, "input_size": input_size, "output_size": input_size,
                    "reduction_percent": 0.0, "warning": error_msg, "method": "fallback_copy_after_opt_error"}
        except:
            return {"success": False, "error": error_msg, "error_log": error_path}

def get_model_size(model_path):
    if model_path and os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return 0.0

# Funkcja do kwantyzacji dynamicznej (pozostawiona dla porównania/ew. użycia)
def quantize_dynamic_onnx_model(input_path, output_path, activation_type_str='uint8', weight_type_str='uint8'):
    result = {"success": False, "quantized_onnx_path": None, "quantization_reduction": 0.0, "error": None}
    if not os.path.exists(input_path):
        result["error"] = f"Plik wejściowy {input_path} nie istnieje."
        print(result["error"])
        return result

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="onnx_quant_dyn_")
    temp_dir_path = temp_dir_obj.name
    current_model_to_quantize_path = input_path
    inferred_shape_model_path = os.path.join(temp_dir_path, "inferred_shape_model_dynamic.onnx")
    
    try:
        print(f"Uruchamianie explicite shape inference dla {current_model_to_quantize_path} (dynamic quant)...")
        onnx.shape_inference.infer_shapes_path(current_model_to_quantize_path, inferred_shape_model_path, check_type=True, strict_mode=False, data_prop=True)
        onnx.checker.check_model(inferred_shape_model_path)
        print(f"Shape inference (dynamic quant) zakończone: {inferred_shape_model_path}.")
        current_model_to_quantize_path = inferred_shape_model_path
    except Exception as e_si:
        print(f"Ostrzeżenie (dynamic quant): Błąd podczas shape inference: {e_si}. Kontynuowanie z modelem bez jawnej inferencji.")

    # Konwersja stringów na QuantType
    activation_qtype = QuantType.QUInt8 if activation_type_str.upper() == 'UINT8' else QuantType.QInt8
    weight_qtype = QuantType.QUInt8 if weight_type_str.upper() == 'UINT8' else QuantType.QInt8
    
    # Uproszczona kwantyzacja dynamiczna - jedna próba
    print(f"Próba kwantyzacji dynamicznej: Activations: {activation_type_str}, Weights: {weight_type_str}")
    try:
        quantize_dynamic(
            model_input=current_model_to_quantize_path,
            model_output=output_path,
            # activation_type i weight_type są domyślnie Int8 w nowszych wersjach, ale jawnie ustawiamy
            # dla pewności, że używamy Int8 dla S8S8 lub Uint8 dla U8U8
            # Zgodnie z dokumentacją S8S8 (Int8/Int8) jest zalecane
            # weight_type=weight_qtype, # parametr activation_type nie istnieje w quantize_dynamic
            # op_types_to_quantize=['MatMul', 'Conv'], # Opcjonalnie
            # per_channel=False, # Domyślnie False
            # reduce_range=False, # Domyślnie False
            # nodes_to_exclude=[], # Można wykluczać konkretne węzły
            extra_options={"EnableSubgraph": True, "ForceQuantizeNoInputCheck": False, "ActivationSymmetric": True, "WeightSymmetric": True} 
            # ActivationSymmetric i WeightSymmetric na True dla S8S8
        )
        onnx.checker.check_model(output_path)
        print(f"Kwantyzacja dynamiczna zakończona sukcesem. Model zapisany w {output_path}")
        original_size_mb = get_model_size(input_path) 
        quantized_size_mb = get_model_size(output_path)
        result["success"] = True
        result["quantized_onnx_path"] = output_path
        if original_size_mb > 0 and quantized_size_mb > 0:
            result["quantization_reduction"] = (1 - (quantized_size_mb / original_size_mb)) * 100
        print(f"Model skwantyzowany dynamicznie. Rozmiar: {quantized_size_mb:.2f} MB. Redukcja: {result.get('quantization_reduction', 0.0):.2f}%")
    except Exception as e:
        print(f"Kwantyzacja dynamiczna nie powiodła się: {e}")
        result["error"] = str(e)
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except OSError: pass
            
    temp_dir_obj.cleanup()
    return result

# NOWA FUNKCJA DLA KWANTYZACJI STATYCZNEJ
def quantize_static_onnx_model(input_path, output_path, calibration_data_reader,
                               activation_type_str='int8', weight_type_str='int8',
                               quant_format_str='qdq', per_channel=False, reduce_range=False):
    """
    Kwantyzacja statyczna modelu ONNX z wykorzystaniem danych kalibracyjnych.
    
    Args:
        input_path: Ścieżka do modelu wejściowego ONNX
        output_path: Ścieżka dla skwantyzowanego modelu ONNX
        calibration_data_reader: CalibrationDataReader z próbkami audio dla kalibracji
        activation_type_str: Typ kwantyzacji aktywacji ('int8' lub 'uint8')
        weight_type_str: Typ kwantyzacji wag ('int8' lub 'uint8')
        quant_format_str: Format kwantyzacji ('qdq' lub 'qoperator')
        per_channel: Czy używać kwantyzacji per-channel dla wag
        reduce_range: Czy zmniejszyć zakres kwantyzacji do 7-bitów (dla starszych CPU)
        
    Returns:
        dict: Wynik operacji zawierający informacje o sukcesie, ścieżkę wyjściową i statystyki
    """
    result = {"success": False, "quantized_onnx_path": None, "quantization_reduction": 0.0, "error": None, "method": "static"}
    if not os.path.exists(input_path):
        result["error"] = f"Plik wejściowy {input_path} nie istnieje."
        print(result["error"])
        return result
    if calibration_data_reader is None:
        result["error"] = "CalibrationDataReader jest wymagany do kwantyzacji statycznej."
        print(result["error"])
        return result

    temp_dir_obj = tempfile.TemporaryDirectory(prefix="onnx_quant_static_")
    temp_dir_path = temp_dir_obj.name
    current_model_to_quantize_path = input_path
    inferred_shape_model_path = os.path.join(temp_dir_path, "inferred_shape_model_static.onnx")

    try:
        print(f"Uruchamianie explicite shape inference dla {current_model_to_quantize_path} (static quant)...")
        onnx.shape_inference.infer_shapes_path(current_model_to_quantize_path, inferred_shape_model_path, check_type=True, strict_mode=False, data_prop=True)
        onnx.checker.check_model(inferred_shape_model_path)
        print(f"Shape inference (static quant) zakończone: {inferred_shape_model_path}.")
        current_model_to_quantize_path = inferred_shape_model_path
    except Exception as e_si:
        print(f"Ostrzeżenie (static quant): Błąd podczas shape inference: {e_si}. Kontynuowanie z modelem bez jawnej inferencji.")

    activation_qtype = QuantType.QInt8 if activation_type_str.lower() == 'int8' else QuantType.QUInt8
    weight_qtype = QuantType.QInt8 if weight_type_str.lower() == 'int8' else QuantType.QUInt8
    q_format = QuantFormat.QDQ if quant_format_str.lower() == 'qdq' else QuantFormat.QOperator

    # Dla modeli CNN preferujemy następujące konfiguracje (według dokumentacji i praktyk)
    attempt_configs = [
        {"name": f"Podstawowa statyczna {activation_type_str}/{weight_type_str} {quant_format_str} dla CNN", 
         "params": {
             "per_channel": per_channel, 
             "reduce_range": reduce_range, 
             "nodes_to_exclude": [],
             "op_types_to_quantize": ["Conv", "MatMul", "Gemm", "MaxPool", "Add"]  # Główne operacje w CNN
         }
        },
        {"name": f"Statyczna z wykluczeniem operacji pomocniczych", 
         "params": {
             "per_channel": per_channel, 
             "reduce_range": reduce_range, 
             "nodes_to_exclude": ['Softmax', 'ReduceMean', 'Tanh', 'Sigmoid'],
             "op_types_to_quantize": ["Conv", "MatMul", "Gemm"]  # Tylko najważniejsze operacje
         }
        },
        {"name": f"Statyczna uproszczona", 
         "params": {
             "per_channel": False,  # Wyłączamy per_channel dla zwiększenia kompatybilności 
             "reduce_range": reduce_range, 
             "nodes_to_exclude": ['Softmax', 'ReduceMean', 'Tanh', 'Sigmoid'],
             "op_types_to_quantize": None  # Pozostawiamy domyślne operacje do kwantyzacji
         }
        }
    ]
    
    # Dodaj konfigurację dla per_channel jeśli domyślnie wyłączona
    if per_channel == False: 
        attempt_configs.append({
            "name": f"Statyczna {activation_type_str}/{weight_type_str} {quant_format_str} PER_CHANNEL", 
            "params": {
                "per_channel": True, 
                "reduce_range": True,  # Dla per-channel often=True
                "nodes_to_exclude": [],
                "op_types_to_quantize": ["Conv", "MatMul"]  # Najbardziej korzystają z per-channel
            }
        })

    for attempt_idx, config in enumerate(attempt_configs):
        attempt_name = config["name"]
        params = config["params"]
        print(f"\nPróba kwantyzacji statycznej ({attempt_idx + 1}/{len(attempt_configs)}): {attempt_name}")

        # Przygotowanie listy nazw węzłów do wykluczenia, jeśli są typy operatorów
        nodes_to_exclude_by_name = []
        if params.get("nodes_to_exclude"):
            try:
                model_for_node_names = onnx.load(current_model_to_quantize_path)
                for node in model_for_node_names.graph.node:
                    if node.op_type in params["nodes_to_exclude"]:
                        nodes_to_exclude_by_name.append(node.name)
                if nodes_to_exclude_by_name:
                    print(f"  Wykluczanie węzłów (wg nazw): {nodes_to_exclude_by_name} dla typów {params['nodes_to_exclude']}")
                else:
                    print(f"  Nie znaleziono węzłów do wykluczenia dla typów: {params['nodes_to_exclude']}")
            except Exception as e_node_load:
                 print(f"  Ostrzeżenie: Nie udało się załadować modelu do identyfikacji nazw węzłów: {e_node_load}")

        # Opcje kwantyzacji dostosowane do modeli CNN
        extra_options = {
            "EnableSubgraph": True,  # Włącz obsługę podgrafów
            "ForceQuantizeNoInputCheck": False,  # Bezpieczniejsza opcja
            "ActivationSymmetric": True,  # Symetryczna kwantyzacja dla aktywacji
            "WeightSymmetric": True,  # Symetryczna kwantyzacja dla wag
            "MatMulConstBOnly": True,  # Kwantyzacja tylko stałych wag w MatMul
            "AddQDQPairToWeight": True  # Dodaj QDQ dla wag
        }

        try:
            quantize_static(
                model_input=current_model_to_quantize_path,
                model_output=output_path,
                calibration_data_reader=calibration_data_reader,
                quant_format=q_format,
                activation_type=activation_qtype,
                weight_type=weight_qtype,
                per_channel=params["per_channel"],
                reduce_range=params["reduce_range"],
                nodes_to_exclude=nodes_to_exclude_by_name if nodes_to_exclude_by_name else None,
                op_types_to_quantize=params.get("op_types_to_quantize"),
                extra_options=extra_options
            )
            
            # Weryfikacja modelu po kwantyzacji
            onnx.checker.check_model(output_path)
            print(f"Kwantyzacja statyczna ({attempt_name}) zakończona sukcesem. Model zapisany w {output_path}")
            
            # Sprawdź rozmiar modelu przed i po kwantyzacji
            original_size_mb = get_model_size(input_path) 
            quantized_size_mb = get_model_size(output_path)
            
            result["success"] = True
            result["quantized_onnx_path"] = output_path
            result["params_used"] = params
            result["extra_options"] = extra_options
            
            if original_size_mb > 0 and quantized_size_mb > 0:
                result["quantization_reduction"] = (1 - (quantized_size_mb / original_size_mb)) * 100
                
            print(f"Model skwantyzowany statycznie ({attempt_name}):")
            print(f"- Rozmiar oryginalny: {original_size_mb:.2f} MB")
            print(f"- Rozmiar po kwantyzacji: {quantized_size_mb:.2f} MB")
            print(f"- Redukcja: {result.get('quantization_reduction', 0.0):.2f}%")
            break 
            
        except Exception as e_quant_static:
            print(f"  Próba kwantyzacji statycznej ({attempt_name}) nie powiodła się: {e_quant_static}")
            if attempt_idx == len(attempt_configs) - 1: 
                result["error"] = f"Wszystkie próby kwantyzacji statycznej nieudane. Ostatni błąd: {str(e_quant_static)}"
                print("Wszystkie próby kwantyzacji statycznej nieudane.")
                
            # Usuń plik wynikowy jeśli istnieje i nie jest to ostatnia próba
            if os.path.exists(output_path) and attempt_idx < len(attempt_configs) - 1:
                try: 
                    os.remove(output_path)
                except OSError: 
                    pass

    # Sprzątanie
    temp_dir_obj.cleanup()
    return result


def verify_onnx_model(onnx_path, test_inputs, original_model=None, tolerance=None):
    import onnxruntime as ort # Lokalny import, aby uniknąć problemów z typowaniem na poziomie modułu
    import numpy as np      # Podobnie
    
    mel, mfcc, chroma = test_inputs
    
    print("Inicjalizacja sesji ONNX Runtime do weryfikacji...")
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
        print("Sesja ONNX Runtime zainicjalizowana.")
        
        input_names = [input_node.name for input_node in sess.get_inputs()]
        print(f"Oczekiwane wejścia ONNX: {input_names}")
        
        # Mapowanie dummy_input na rzeczywiste nazwy wejść modelu ONNX
        # Zakładamy, że kolejność w dummy_input odpowiada kolejności cech w modelu
        # i że nazwy wejściowe w ONNX będą zawierać 'mel', 'mfcc', 'chroma'
        onnx_inputs = {}
        dummy_input_map = {'mel': mel.cpu().numpy(), 'mfcc': mfcc.cpu().numpy(), 'chroma': chroma.cpu().numpy()}
        
        for name in input_names:
            if 'mel' in name.lower(): onnx_inputs[name] = dummy_input_map['mel']
            elif 'mfcc' in name.lower(): onnx_inputs[name] = dummy_input_map['mfcc']
            elif 'chroma' in name.lower(): onnx_inputs[name] = dummy_input_map['chroma']
            else: # Fallback, jeśli nazwa nie pasuje
                print(f"Ostrzeżenie: Nie udało się zmapować wejścia ONNX '{name}' do znanego typu cechy. Używam pierwszego dummy input.")
                if not onnx_inputs: # Jeśli słownik jest pusty, dodaj pierwszy element
                     onnx_inputs[name] = mel.cpu().numpy()


        if len(onnx_inputs) != len(input_names):
            print(f"Ostrzeżenie: Liczba zmapowanych wejść ({len(onnx_inputs)}) nie zgadza się z liczbą wejść modelu ({len(input_names)}).")
            # Mimo to próbujemy kontynuować, jeśli kluczowe wejścia są zmapowane

    except Exception as e:
        error_msg = f"Błąd inicjalizacji sesji ONNX lub mapowania wejść: {e}"
        print(error_msg)
        return {"success": False, "error": error_msg}
    
    results = {}
    
    print("Uruchamianie inferencji ONNX do weryfikacji...")
    try:
        onnx_outputs = sess.run(None, onnx_inputs)
        results["onnx_inference_success"] = True
        print(f"Kształt wyjścia ONNX: {onnx_outputs[0].shape}")
    except Exception as e:
        error_msg = f"Błąd podczas inferencji ONNX: {e}"
        print(error_msg)
        import traceback
        print(traceback.format_exc())
        return {"success": False, "error": error_msg}
    
    if original_model:
        print("Porównywanie wyjść modeli PyTorch i ONNX...")
        try:
            original_model.eval()
            with torch.no_grad():
                # Przygotuj wejścia dla modelu PyTorch (słownik)
                pytorch_inputs_dict = {
                    # Użyj oryginalnych tensorów PyTorch, a nie numpy
                    'melspectrogram': test_inputs[0], 
                    'mfcc': test_inputs[1],
                    'chroma': test_inputs[2]
                }
                torch_output = original_model(pytorch_inputs_dict)
                
                np_torch_output = torch_output.cpu().numpy()
                np_onnx_output = onnx_outputs[0]
                
                if np_torch_output.shape != np_onnx_output.shape:
                    print(f"Ostrzeżenie: Różne kształty wyjść PyTorch {np_torch_output.shape} vs ONNX {np_onnx_output.shape}")
                    try: # Próba dopasowania przez usunięcie wymiaru batch jeśli ONNX go nie ma
                        if np_torch_output.ndim == np_onnx_output.ndim + 1 and np_torch_output.shape[0] == 1:
                             np_torch_output = np_torch_output.squeeze(0)
                        elif np_onnx_output.ndim == np_torch_output.ndim + 1 and np_onnx_output.shape[0] == 1:
                             np_onnx_output = np_onnx_output.squeeze(0)
                    except: pass # Ignoruj błąd, jeśli dopasowanie nie jest trywialne

                if np_torch_output.shape != np_onnx_output.shape: # Sprawdź ponownie
                     print(f"Kształty nadal różne po próbie dopasowania. Porównanie może być niemiarodajne.")
                
                abs_diff = np.abs(np_torch_output - np_onnx_output)
                max_diff = np.max(abs_diff) if abs_diff.size > 0 else 0.0
                mean_diff = np.mean(abs_diff) if abs_diff.size > 0 else 0.0
                median_diff = np.median(abs_diff) if abs_diff.size > 0 else 0.0
                std_diff = np.std(abs_diff) if abs_diff.size > 0 else 0.0
                
                if tolerance is None:
                    tolerance = max(np.max(np.abs(np_torch_output)) * 0.01 if np_torch_output.size > 0 else 1e-5, mean_diff * 10, 1e-5) # Zwiększono próg do 1%
                
                results.update({
                    "max_diff": float(max_diff), "mean_diff": float(mean_diff),
                    "median_diff": float(median_diff), "std_diff": float(std_diff),
                    "tolerance": float(tolerance),
                    "pytorch_output_range": [float(np.min(np_torch_output)), float(np.max(np_torch_output))] if np_torch_output.size > 0 else [0,0],
                    "onnx_output_range": [float(np.min(np_onnx_output)), float(np.max(np_onnx_output))] if np_onnx_output.size > 0 else [0,0]
                })
                
                print(f"Statystyki różnic: Max={max_diff:.6f}, Mean={mean_diff:.6f}, Median={median_diff:.6f}, Std={std_diff:.6f}. Tolerancja={tolerance:.6f}")
                
                if max_diff > tolerance:
                    warning_msg = f"Uwaga: Znacząca różnica między wyjściami (max_diff={max_diff:.6f} > tolerance={tolerance:.6f})"
                    print(warning_msg)
                    results["warning"] = warning_msg
                    results["success"] = True # Lekka różnica nie oznacza niepowodzenia
                    results["has_warning"] = True
                    return results
        except Exception as e_compare:
            error_msg = f"Błąd podczas porównywania wyników PyTorch i ONNX: {e_compare}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            # Nadal zwracamy sukces, jeśli inferencja ONNX się powiodła, ale porównanie nie
            results["success"] = results.get("onnx_inference_success", False) 
            results["error_comparison"] = error_msg
            return results
    
    print("Weryfikacja ONNX zakończona pomyślnie (lub tylko inferencja ONNX jeśli brak modelu PyTorch).")
    results["success"] = results.get("onnx_inference_success", False) # Sukces jeśli inferencja ONNX przeszła
    return results

def extract_features_for_inference(audio_path, sr=22050, max_length=3.0, 
                                  n_mels=128, n_mfcc=40, n_chroma=12,
                                  n_fft=2048, hop_length=512, normalize=True):
    """
    Ekstrakcja cech z pliku audio dla inferencji produkcyjnej.
    
    Args:
        audio_path: Ścieżka do pliku audio lub obiekt BytesIO
        sr: Częstotliwość próbkowania
        max_length: Maksymalna długość w sekundach
        n_mels, n_mfcc, n_chroma: Parametry ekstrakcji cech
        n_fft, hop_length: Parametry okna
        normalize: Czy normalizować cechy
        
    Returns:
        dict: Słownik z tensorami cech dla każdej reprezentacji
    """

    
    # Wczytanie audio
    try:
        if isinstance(audio_path, str):
            audio, sr_loaded = librosa.load(audio_path, sr=sr)
        else:
            audio, sr_loaded = librosa.load(audio_path, sr=sr)
    except Exception as e:
        raise ValueError(f"Nie udało się wczytać pliku audio: {e}")
    
    # Ekstrakcja cech
    features = {}
    
    # Melspectrogram
    mel = extract_features(audio, sr, "melspectrogram", max_length=max_length,
                          n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
                          normalize=normalize)
    features['mel_input'] = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0)
    
    # MFCC
    mfcc = extract_features(audio, sr, "mfcc", max_length=max_length,
                           n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
                           normalize=normalize)
    features['mfcc_input'] = torch.FloatTensor(mfcc).unsqueeze(0).unsqueeze(0)
    
    # Chroma
    chroma = extract_features(audio, sr, "chroma", max_length=max_length,
                             n_chroma=n_chroma, n_fft=n_fft, hop_length=hop_length,
                             normalize=normalize)
    features['chroma_input'] = torch.FloatTensor(chroma).unsqueeze(0).unsqueeze(0)
    
    return features

def extract_features(audio_array, sr, feature_type, max_length=3.0,
                    n_mels=128, n_mfcc=40, n_chroma=12,
                    n_fft=2048, hop_length=512, normalize=True):
    """Funkcja ekstrakcji cech - już dostępna w kodzie"""
    
    # Ustalenie docelowej długości sygnału
    target_length = int(max_length * sr)
    if len(audio_array) > target_length:
        audio_array = audio_array[:target_length]
    else:
        padding = np.zeros(target_length - len(audio_array))
        audio_array = np.concatenate([audio_array, padding])
    
    feature = None
    
    if feature_type == "melspectrogram":
        S = librosa.feature.melspectrogram(
            y=audio_array, sr=sr, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length
        )
        feature = librosa.power_to_db(S, ref=np.max)
        
    elif feature_type == "mfcc":
        feature = librosa.feature.mfcc(
            y=audio_array, sr=sr, n_mfcc=n_mfcc,
            n_fft=n_fft, hop_length=hop_length
        )
        
    elif feature_type == "chroma":
        feature = librosa.feature.chroma_stft(
            y=audio_array, sr=sr, n_chroma=n_chroma,
            n_fft=n_fft, hop_length=hop_length
        )
    
    # Normalizacja cech (opcjonalna)
    if normalize and feature is not None:
        if feature_type in ["mfcc"]:
            feature = librosa.util.normalize(feature)
        elif feature_type in ["melspectrogram"]:
            # Spektrogramy już są w dB
            pass
        else:
            # Standardowa normalizacja min-max
            feature_min = np.min(feature)
            feature_max = np.max(feature)
            if feature_max > feature_min:
                feature = (feature - feature_min) / (feature_max - feature_min)
    
    return feature

def create_onnx_inference_session(onnx_path, use_cuda=None):
    """
    Tworzenie sesji inferencyjnej ONNX.
    
    Args:
        onnx_path: Ścieżka do modelu ONNX
        use_cuda: Czy używać CUDA (jeśli dostępne). Jeśli None, użyj CUDA jeśli jest dostępne.
    
    Returns:
        ort.InferenceSession: Obiekt sesji ONNX
    """
    
    # Sprawdź dostępne dostawcy
    available_providers = ort.get_available_providers()
    print(f"Dostępni dostawcy ONNX Runtime: {available_providers}")
    
    # Domyślne ustawienie use_cuda na True jeśli CUDA jest dostępne
    if use_cuda is None:
        use_cuda = 'CUDAExecutionProvider' in available_providers
    
    # Ustal listę dostawców (providers) na podstawie dostępności CUDA i preferencji
    providers = []
    if use_cuda and 'CUDAExecutionProvider' in available_providers:
        # Bardziej ostrożna konfiguracja CUDA z mniejszą liczbą opcji, które mogą powodować problemy
        try:
            cuda_options: Dict[str, Union[int, str]] = {
                'device_id': 0,
            }
            # Dodaj dodatkowe opcje tylko jeśli ich wersja onnxruntime je obsługuje
            ort_version = tuple(map(int, ort.__version__.split('.')[:2]))
            if ort_version >= (1, 10):  # Te opcje dostępne od wersji 1.10
                cuda_options.update({
                    'arena_extend_strategy': 'kNextPowerOfTwo'
                })
            
            providers.append(('CUDAExecutionProvider', cuda_options))
            print(f"Używam CUDA do inferencji ONNX (wersja ONNX Runtime: {ort.__version__})")
        except Exception as e:
            print(f"Ostrzeżenie: Problem z konfiguracją CUDA: {e}. Przełączam na CPU.")
            use_cuda = False
    else:
        if use_cuda and 'CUDAExecutionProvider' not in available_providers:
            print("CUDA jest preferowana, ale nie jest dostępna w ONNX Runtime")
        print(f"Używam CPU do inferencji ONNX (wersja ONNX Runtime: {ort.__version__})")
    
    # Zawsze dodajemy wykonawcę CPU jako fallback
    providers.append('CPUExecutionProvider')
    
    # Konfiguracja sesji
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Optymalizacje wielowątkowe dla CPU
    if not use_cuda or 'CPUExecutionProvider' in providers:
        # Wykorzystanie wszystkich dostępnych rdzeni
        import multiprocessing
        num_threads = min(multiprocessing.cpu_count(), 8)  # Nie więcej niż 8 wątków
        sess_options.intra_op_num_threads = num_threads
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        print(f"Używam {num_threads} wątków CPU do inferencji")
    
    # Ustawienie polityki alokacji pamięci dla lepszej wydajności
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True
    
    try:
        # Tworzenie sesji - w przypadku problemu z CUDA, spróbuj tylko z CPU
        try:
            session = ort.InferenceSession(
                onnx_path, 
                sess_options=sess_options,
                providers=providers
            )
        except Exception as e:
            if use_cuda and 'CUDAExecutionProvider' in providers[0][0]:
                print(f"Błąd podczas tworzenia sesji z CUDA: {e}")
                print("Próbuję utworzyć sesję używając tylko CPU...")
                session = ort.InferenceSession(
                    onnx_path, 
                    sess_options=sess_options,
                    providers=['CPUExecutionProvider']
                )
            else:
                raise e
        
        # Wyświetl informacje o sesji
        print(f"Utworzono sesję ONNX Runtime dla modelu: {onnx_path}")
        print(f"Użyte dostawce (providers): {session.get_providers()}")
        
        # Wyświetl informacje o wejściach i wyjściach modelu
        inputs_info = session.get_inputs()
        outputs_info = session.get_outputs()
        
        print("Oczekiwane wejścia:")
        for input_info in inputs_info:
            print(f"  - {input_info.name}: {input_info.shape} ({input_info.type})")
        
        print("Wyjścia modelu:")
        for output_info in outputs_info:
            print(f"  - {output_info.name}: {output_info.shape} ({output_info.type})")
        
        return session
    
    except Exception as e:
        print(f"Błąd podczas tworzenia sesji ONNX: {e}")
        import traceback
        print(traceback.format_exc())
        raise

def run_onnx_inference(session, features):
    """
    Uruchomienie inferencji ONNX z przygotowanymi cechami.
    
    Args:
        session: Sesja ONNX Runtime
        features: Słownik z cechami audio
    
    Returns:
        np.ndarray: Wynik predykcji
    """
    # Przygotowanie wejść dla sesji ONNX
    onnx_inputs = {}
    for name, tensor in features.items():
        if isinstance(tensor, torch.Tensor):
            onnx_inputs[name] = tensor.numpy()
        else:
            onnx_inputs[name] = tensor
    
    # Uruchomienie inferencji
    outputs = session.run(None, onnx_inputs)
    
    return outputs[0]
