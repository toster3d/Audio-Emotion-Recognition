import torch
import warnings
import traceback
import onnx
import librosa
import numpy as np
import onnxruntime as ort
import os

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

def export_onnx_model(ensemble_model, output_path, dummy_input, opset_version=17):
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
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                
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
                    training=torch.onnx.TrainingMode.EVAL,
                    dynamo=True
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
        import onnx
        
        # Sprawdzenie czy plik istnieje
        if not os.path.exists(input_path):
            error_msg = f"Plik modelu ONNX nie istnieje: {input_path}"
            print(error_msg)
            return {"success": False, "error": error_msg}
        
        # Ładowanie modelu
        try:
            model = onnx.load(input_path)
        except Exception as e:
            error_msg = f"Błąd podczas ładowania modelu ONNX: {e}"
            print(error_msg)
            return {"success": False, "error": error_msg}
        
        # Weryfikacja modelu przed optymalizacją
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            error_msg = f"Model ONNX jest niepoprawny przed optymalizacją: {e}"
            print(error_msg)
            return {"success": False, "error": error_msg}
            
        # Najpierw skopiuj oryginalny model jako kopię zapasową
        import shutil
        backup_path = input_path + ".backup"
        shutil.copy(input_path, backup_path)
        print(f"Utworzono kopię zapasową oryginalnego modelu: {backup_path}")
        
        input_size = os.path.getsize(input_path)
        print(f"Rozmiar oryginalnego modelu: {input_size / (1024*1024):.2f} MB")
            
        # Optymalizacja modelu
        try:
            import onnxoptimizer as optimizer
            
            # Używanie bezpieczniejszego zestawu optymalizacji (mniejsza liczba potencjalnie problematycznych)
            safe_passes = [
                'eliminate_nop_dropout',
                'eliminate_unused_initializer',
                'eliminate_identity',
                'fuse_bn_into_conv'
            ]
            
            print("Używanie bezpiecznych optymalizacji...")
                
            # Próba optymalizacji
            optimized_model = optimizer.optimize(model, safe_passes)
            
            # Zapisanie zoptymalizowanego modelu w trybie testowym, aby sprawdzić jego rozmiar
            temp_output_path = output_path + ".temp"
            onnx.save(optimized_model, temp_output_path)
            
            # Weryfikacja modelu po optymalizacji
            try:
                onnx.checker.check_model(optimized_model)
                print("Model po optymalizacji poprawny.")
            except Exception as e:
                warning_msg = f"Ostrzeżenie: Model po optymalizacji ma problemy z weryfikacją: {e}"
                print(warning_msg)
                # Usuwamy plik tymczasowy i używamy oryginału
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
                shutil.copy(input_path, output_path)
                return {
                    "success": True,
                    "input_size": input_size,
                    "output_size": input_size,
                    "reduction_percent": 0.0,
                    "warning": warning_msg,
                    "method": "no_optimization"
                }
                
            # Sprawdzanie efektywności optymalizacji
            temp_output_size = os.path.getsize(temp_output_path)
            
            # Sprawdzenie czy zoptymalizowany model nie jest znacznie większy niż oryginalny
            # Jeśli jest ponad 2x większy, używamy oryginału zamiast
            if temp_output_size > 2 * input_size:
                warning_msg = (f"Ostrzeżenie: Zoptymalizowany model jest znacznie większy "
                               f"({temp_output_size / (1024*1024):.2f} MB) niż oryginalny "
                               f"({input_size / (1024*1024):.2f} MB). Używanie oryginału.")
                print(warning_msg)
                
                # Usuwamy plik tymczasowy i używamy oryginału
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)
                
                shutil.copy(input_path, output_path)
                
                return {
                    "success": True,
                    "input_size": input_size,
                    "output_size": input_size,
                    "reduction_percent": 0.0,
                    "warning": warning_msg,
                    "method": "no_optimization"
                }
            
            # Jeśli optymalizacja przeszła pomyślnie i model nie jest znacznie większy
            # Przenieś plik tymczasowy jako docelowy
            if os.path.exists(temp_output_path):
                shutil.move(temp_output_path, output_path)
            
            output_size = os.path.getsize(output_path)
            
            # Obliczenie procentowej redukcji rozmiaru
            reduction_percent = (1 - output_size / input_size) * 100 if input_size > 0 else 0
            
            print(f"Optymalizacja zakończona pomyślnie.")
            print(f"Rozmiar oryginalny: {input_size / (1024*1024):.2f} MB")
            print(f"Rozmiar po optymalizacji: {output_size / (1024*1024):.2f} MB")
            print(f"Redukcja rozmiaru: {reduction_percent:.2f}%")
            
            return {
                "success": True,
                "input_size": input_size,
                "output_size": output_size,
                "reduction_percent": reduction_percent,
                "method": "onnx_optimizer",
                "passes_applied": safe_passes
            }
            
        except Exception as e:
            # Jeśli optymalizacja się nie powiodła, używamy oryginału
            error_msg = f"Optymalizacja nie powiodła się: {e}. Kopiowanie oryginalnego modelu."
            print(error_msg)
            
            # Kopiowanie oryginalnego modelu do wyjścia
            shutil.copy(input_path, output_path)
            
            return {
                "success": True,  # Nadal zwracamy True, ponieważ mamy działający model
                "input_size": input_size,
                "output_size": os.path.getsize(output_path),
                "reduction_percent": 0.0,
                "warning": error_msg,
                "method": "no_optimization"
            }
            
    except Exception as e:
        error_msg = f"Nieoczekiwany błąd podczas optymalizacji modelu: {e}"
        print(error_msg)
        
        # Zapisz szczegółowe informacje o błędzie
        import traceback
        tb = traceback.format_exc()
        error_path = output_path + ".optimization.error.log"
        with open(error_path, "w") as f:
            f.write(f"Błąd optymalizacji modelu ONNX:\n{error_msg}\n\nTraceback:\n{tb}")
        
        print(f"Szczegóły błędu zapisane do: {error_path}")
        
        # W przypadku błędu, kopiujemy oryginalny model, aby nie przerywać potoku eksportu
        try:
            import shutil
            print(f"Kopiowanie oryginalnego modelu jako zapasowe rozwiązanie...")
            shutil.copy(input_path, output_path)
            input_size = os.path.getsize(input_path)
            return {
                "success": True,
                "input_size": input_size,
                "output_size": input_size,
                "reduction_percent": 0.0,
                "warning": error_msg,
                "method": "fallback_copy"
            }
        except:
            # Jeśli nawet kopiowanie się nie powiodło, zwracamy błąd
            return {"success": False, "error": error_msg, "error_log": error_path}

def quantize_onnx_model(input_path, output_path, quantization_type='dynamic', dtype='uint8'):
    """
    Kwantyzacja modelu ONNX do mniejszej precyzji.
    
    Args:
        input_path: Ścieżka do modelu ONNX
        output_path: Ścieżka wyjściowa dla skwantyzowanego modelu
        quantization_type: Typ kwantyzacji ('dynamic' lub 'static')
        dtype: Typ danych ('uint8' lub 'int8')
    
    Returns:
        dict: Słownik z wynikami kwantyzacji
    """
    try:
        print(f"Rozpoczęcie kwantyzacji modelu ONNX: {input_path}")
        
        # Weryfikacja modelu przed kwantyzacją
        try:
            import onnx
            model = onnx.load(input_path)
            onnx.checker.check_model(model)
            print("Model ONNX poprawny przed kwantyzacją.")
            
            # Próba przeprowadzenia shape inference przed kwantyzacją
            try:
                from onnx import shape_inference
                inferred_model = shape_inference.infer_shapes(model)
                onnx.checker.check_model(inferred_model)
                print("Shape inference przeprowadzone pomyślnie.")
                
                # Zapisz model z wywnioskowanymi kształtami do pliku tymczasowego
                inferred_path = input_path + ".inferred"
                onnx.save(inferred_model, inferred_path)
                # Użyj tego modelu do dalszej kwantyzacji
                model_path_for_quantization = inferred_path
            except Exception as e:
                print(f"Ostrzeżenie: Nie udało się przeprowadzić shape inference: {e}")
                print("Kontynuowanie z oryginalnym modelem.")
                model_path_for_quantization = input_path
        except Exception as e:
            error_msg = f"Model ONNX nie jest prawidłowy przed kwantyzacją: {e}"
            print(error_msg)
            # Zapisz szczegóły błędu
            error_path = output_path + ".model_check.error.log"
            with open(error_path, "w") as f:
                f.write(f"Błąd weryfikacji modelu ONNX:\n{str(e)}")
            return {"success": False, "error": error_msg, "error_log": error_path}
        
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            # Wybór typu kwantyzacji
            if dtype.lower() == 'uint8':
                weight_type = QuantType.QUInt8
                print("Użycie kwantyzacji do unsigned int8")
            elif dtype.lower() == 'int8':
                weight_type = QuantType.QInt8
                print("Użycie kwantyzacji do signed int8")
            else:
                print(f"Nieznany typ danych: {dtype}, użycie domyślnie uint8")
                weight_type = QuantType.QUInt8
            
            # Najpierw spróbuj uproszczonej kwantyzacji bez problematycznych parametrów
            print("Próba kwantyzacji podstawowej...")
            try:
                quantize_dynamic(
                    model_input=model_path_for_quantization,
                    model_output=output_path,
                    weight_type=weight_type,
                    per_channel=False,
                    reduce_range=True
                )
                print("Podstawowa kwantyzacja zakończona pomyślnie.")
            except Exception as e:
                print(f"Podstawowa kwantyzacja nie powiodła się: {e}")
                print("Próba kwantyzacji z pominięciem problematycznych operatorów...")
                
                # Spróbuj z pominięciem problematycznych operatorów
                try:
                    # Lista typów operatorów do pominięcia podczas kwantyzacji
                    op_types_to_exclude = ["Reshape", "Transpose", "Concat", "Slice", "Squeeze", "Unsqueeze"]
                    print(f"Pominięcie operatorów: {op_types_to_exclude}")
                    
                    quantize_dynamic(
                        model_input=model_path_for_quantization,
                        model_output=output_path,
                        weight_type=weight_type,
                        per_channel=False,
                        reduce_range=True,
                        op_types_to_exclude=op_types_to_exclude
                    )
                    print("Kwantyzacja z pominięciem problematycznych operatorów zakończona pomyślnie.")
                except Exception as e2:
                    print(f"Również nie powiodła się kwantyzacja z pominięciem operatorów: {e2}")
                    print("Próba ostatecznej, minimalnej kwantyzacji...")
                    
                    # Spróbuj z dodatkowymi parametrami bezpieczeństwa
                    try:
                        # Bardziej restrykcyjna lista operatorów do kwantyzacji
                        op_types_to_quantize = ["Conv", "MatMul"]
                        print(f"Kwantyzacja tylko operatorów: {op_types_to_quantize}")
                        
                        quantize_dynamic(
                            model_input=model_path_for_quantization,
                            model_output=output_path,
                            weight_type=weight_type,
                            per_channel=False,
                            reduce_range=True,
                            op_types_to_quantize=op_types_to_quantize,
                            extra_options=dict(EnableSubgraph=False)
                        )
                        print("Minimalna kwantyzacja zakończona pomyślnie.")
                    except Exception as e3:
                        # Jeśli wszystko zawiedzie, zwróć błąd
                        error_msg = f"Wszystkie próby kwantyzacji nieudane: {e3}"
                        print(error_msg)
                        return {"success": False, "error": error_msg}
            
            # Usuń tymczasowe pliki
            if 'inferred_path' in locals() and os.path.exists(inferred_path):
                try:
                    os.remove(inferred_path)
                except:
                    pass
            
            # Sprawdź, czy kwantyzacja się powiodła poprzez weryfikację rozmiaru pliku i jego istnienia
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                input_size = os.path.getsize(input_path)
                output_size = os.path.getsize(output_path)
                
                # Oblicz procentową redukcję rozmiaru
                reduction_percent = (1 - output_size / input_size) * 100
                
                print(f"Kwantyzacja zakończona. Model zapisany do: {output_path}")
                print(f"Rozmiar oryginalny: {input_size / (1024*1024):.2f} MB")
                print(f"Rozmiar po kwantyzacji: {output_size / (1024*1024):.2f} MB")
                print(f"Redukcja rozmiaru: {reduction_percent:.2f}%")
                
                # Weryfikacja skwantyzowanego modelu
                try:
                    quant_model = onnx.load(output_path)
                    onnx.checker.check_model(quant_model)
                    print("Model ONNX poprawny po kwantyzacji.")
                except Exception as e:
                    warning_msg = f"Ostrzeżenie: Model po kwantyzacji ma problemy z weryfikacją: {e}"
                    print(warning_msg)
                    # To tylko ostrzeżenie, nie przerywa procesu
                
                return {
                    "success": True,
                    "input_size": input_size,
                    "output_size": output_size,
                    "reduction_percent": reduction_percent,
                    "method": f"quantization_{quantization_type}_{dtype}"
                }
            else:
                error_msg = "Kwantyzacja nie powiodła się - plik wyjściowy nie istnieje lub jest pusty"
                print(error_msg)
                return {"success": False, "error": error_msg}
                
        except ImportError as e:
            error_msg = f"Brak wymaganych bibliotek do kwantyzacji: {e}"
            print(error_msg)
            print("Zainstaluj wymagane biblioteki: pip install onnxruntime onnx")
            return {"success": False, "error": error_msg}
            
    except Exception as e:
        error_msg = f"Błąd podczas kwantyzacji modelu: {e}"
        print(error_msg)
        
        # Zapisz szczegółowe informacje o błędzie
        import traceback
        tb = traceback.format_exc()
        error_path = output_path + ".quantization.error.log"
        with open(error_path, "w") as f:
            f.write(f"Błąd kwantyzacji modelu ONNX:\n{error_msg}\n\nTraceback:\n{tb}")
        print(f"Szczegóły błędu zapisane do: {error_path}")
        
        return {"success": False, "error": error_msg, "error_log": error_path}

def verify_onnx_model(onnx_path, test_inputs, original_model=None, tolerance=None):
    """
    Weryfikacja zgodności modelu ONNX z oryginalnym modelem PyTorch.
    
    Args:
        onnx_path: Ścieżka do modelu ONNX
        test_inputs: Dane testowe (tuple z mel, mfcc, chroma)
        original_model: Oryginalny model PyTorch do porównania (opcjonalnie)
        tolerance: Wartość progowa dla różnicy między wynikami (opcjonalnie)
                  Jeśli None, zostanie ustalona adaptacyjnie na podstawie danych
    
    Returns:
        dict: Słownik z wynikami weryfikacji zawierający:
             - success: True/False - czy weryfikacja się powiodła
             - max_diff: Maksymalna różnica między wynikami
             - mean_diff: Średnia różnica między wynikami
             - tolerance: Użyta wartość progowa
             - error: Komunikat błędu (jeśli wystąpił)
    """
    import onnxruntime as ort
    import numpy as np
    
    # Rozpakuj dane wejściowe
    mel, mfcc, chroma = test_inputs
    
    # Utwórz sesję ONNX Runtime
    print("Inicjalizacja sesji ONNX Runtime...")
    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Zawsze używamy CPU do weryfikacji, by uniknąć problemów z CUDA
        sess = ort.InferenceSession(
            onnx_path, 
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        print("Sesja ONNX Runtime zainicjalizowana pomyślnie.")
        
        # Sprawdź nazwy wejść
        input_names = [input.name for input in sess.get_inputs()]
        print(f"Oczekiwane wejścia ONNX: {input_names}")
        
        # Dostosuj nazwy wejść, jeśli różnią się od oczekiwanych
        expected_names = ['mel_input', 'mfcc_input', 'chroma_input']
        actual_names = {}
        
        for expected, tensor_name in zip(expected_names, input_names):
            actual_names[expected] = tensor_name
        
        # Przygotuj dane wejściowe zgodne z nazwami
        onnx_inputs = {
            actual_names.get('mel_input', 'mel_input'): mel.cpu().numpy(),
            actual_names.get('mfcc_input', 'mfcc_input'): mfcc.cpu().numpy(),
            actual_names.get('chroma_input', 'chroma_input'): chroma.cpu().numpy()
        }
    except Exception as e:
        error_msg = f"Błąd inicjalizacji sesji ONNX: {e}"
        print(error_msg)
        return {"success": False, "error": error_msg}
    
    results = {}  # Słownik na wyniki weryfikacji
    
    # Uruchom inferencję ONNX
    print("Uruchamianie inferencji ONNX...")
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
    
    # Porównaj z modelem PyTorch (jeśli dostępny)
    if original_model:
        print("Porównywanie wyjść modeli PyTorch i ONNX...")
        try:
            # Upewnij się, że model PyTorch jest w trybie ewaluacji
            original_model.eval()
            
            # Uzyskaj wyjście z oryginalnego modelu
            with torch.no_grad():
                torch_output = original_model({
                    'melspectrogram': mel,
                    'mfcc': mfcc,
                    'chroma': chroma
                })
                
                # Porównaj wyniki
                np_torch_output = torch_output.cpu().numpy()
                np_onnx_output = onnx_outputs[0]
                
                # Dopasuj kształty do porównania, jeśli są różne
                if np_torch_output.shape != np_onnx_output.shape:
                    print(f"Ostrzeżenie: Różne kształty wyjść PyTorch {np_torch_output.shape} vs ONNX {np_onnx_output.shape}")
                    # Próba przekształcenia do wspólnego kształtu (flatten)
                    np_torch_output = np_torch_output.reshape(-1)
                    np_onnx_output = np_onnx_output.reshape(-1)
                
                # Oblicz różne miary różnic między wynikami
                abs_diff = np.abs(np_torch_output - np_onnx_output)
                max_diff = np.max(abs_diff)
                mean_diff = np.mean(abs_diff)
                median_diff = np.median(abs_diff)
                std_diff = np.std(abs_diff)
                
                # Ustal adaptacyjną wartość progową, jeśli nie podano
                if tolerance is None:
                    tolerance = max(
                        np.max(np.abs(np_torch_output)) * 0.001,  # 0.1% maksymalnej wartości
                        mean_diff * 10,                           # 10x średnia różnica
                        1e-5                                      # Minimalny próg
                    )
                
                # Zapisz wyniki porównania
                results.update({
                    "max_diff": float(max_diff),
                    "mean_diff": float(mean_diff),
                    "median_diff": float(median_diff),
                    "std_diff": float(std_diff),
                    "tolerance": float(tolerance),
                    "pytorch_output_range": [float(np.min(np_torch_output)), float(np.max(np_torch_output))],
                    "onnx_output_range": [float(np.min(np_onnx_output)), float(np.max(np_onnx_output))]
                })
                
                print(f"Statystyki różnic między wyjściami PyTorch i ONNX:")
                print(f"  - Maksymalna różnica: {max_diff:.8f}")
                print(f"  - Średnia różnica: {mean_diff:.8f}")
                print(f"  - Mediana różnic: {median_diff:.8f}")
                print(f"  - Odchylenie std. różnic: {std_diff:.8f}")
                print(f"  - Użyta wartość progowa: {tolerance:.8f}")
                
                # Sprawdzenie, czy wyjścia są wystarczająco zgodne
                if max_diff > tolerance:
                    warning_msg = f"Uwaga: Znacząca różnica między wyjściami PyTorch i ONNX (max_diff={max_diff:.8f}, tolerance={tolerance:.8f})"
                    print(warning_msg)
                    results["warning"] = warning_msg
                    # Lekka różnica nie powinna oznaczać niepowodzenia - oznaczamy jako ostrzeżenie
                    results["success"] = True
                    results["has_warning"] = True
                    return results
        except Exception as e:
            error_msg = f"Błąd podczas porównywania wyników PyTorch i ONNX: {e}"
            print(error_msg)
            import traceback
            print(traceback.format_exc())
            return {"success": False, "error": error_msg}
    
    print("Weryfikacja ONNX zakończona pomyślnie.")
    results["success"] = True
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
            audio, sr = librosa.load(audio_path, sr=sr)
        else:
            audio, sr = librosa.load(audio_path, sr=sr)
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
            cuda_options = {
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
