# Eksport modelu Ensemble do formatu ONNX

Ten katalog zawiera skrypty do eksportu modelu ensemble do formatu ONNX, który umożliwia uruchamianie modelu na różnych platformach i w różnych językach programowania.

## Najnowsze poprawki (15.05.2025)

Rozwiązano następujące problemy:

1. **Problem z wybuchem rozmiaru przy optymalizacji** - Naprawiono problem, gdy optymalizacja zwiększała rozmiar modelu z ~0.5MB do 128MB. Zaimplementowano zabezpieczenie, które porównuje rozmiar pliku przed i po optymalizacji. Jeśli zoptymalizowany model jest ponad 2x większy od oryginału, system automatycznie używa oryginalnego modelu zamiast zoptymalizowanego.

2. **Problemy z kwantyzacją** - Rozwiązano błąd kwantyzacji `[ShapeInferenceError] Inferred shape and existing shape differ in dimension 0: (512) vs (6)` poprzez:
   - Dodanie procesu shape inference przed kwantyzacją
   - Implementację kaskadowego podejścia do kwantyzacji z trzema poziomami:
     - Podstawowa kwantyzacja z prostymi parametrami
     - Kwantyzacja z pominięciem problematycznych operatorów (Reshape, Transpose, Concat, itp.)
     - Minimalna kwantyzacja tylko dla wybranych operatorów (Conv, MatMul)
   - Usunięcie problematycznych parametrów `optimize_model` niekompatybilnych z onnxruntime 1.19.2

3. **Zakres importów globalnych** - Naprawiono problem, w którym importy wewnątrz bloków try/except mogły prowadzić do błędów "local variable referenced before assignment". Wszystkie niezbędne importy są teraz wykonywane na poziomie modułu.

4. **Ulepszony proces wyboru modelu bazowego** - Logika w `export_onnx_notebook_helper.py` została ulepszona, aby inteligentnie wybierać, który model (oryginalny vs. zoptymalizowany) powinien być użyty jako podstawa do kwantyzacji, szczególnie gdy optymalizacja znacząco zwiększa rozmiar.

5. **Weryfikacja modelu** - Dodano weryfikację modelu przed i po każdym etapie przetwarzania (eksport, optymalizacja, kwantyzacja), aby wcześnie wykrywać problemy.

Te poprawki znacząco zwiększają niezawodność procesu eksportu ONNX, a w szczególności skutecznie:
- Zapobiegają eksplozji rozmiaru modelu z 0.49MB do 128MB
- Umożliwiają skuteczną kwantyzację, która powinna zmniejszyć rozmiar modelu o 60-75%
- Zwiększają kompatybilność z różnymi wersjami onnxruntime

## Dostępne skrypty

1. **export_to_onnx.py** - Główny skrypt do eksportu modelu z wiersza poleceń
2. **export_onnx_notebook_helper.py** - Skrypt pomocniczy do eksportu modelu z notebooka
3. **ensemble_onnx_wrapper.py** - Zawiera funkcje do eksportu, optymalizacji i kwantyzacji modeli ONNX
4. **test_onnx_model.py** - Testuje model ONNX z rzeczywistym plikiem audio
5. **test_without_audio.py** - Testuje model ONNX z losowymi danymi wejściowymi
6. **test_quantization.py** - Testuje różne metody kwantyzacji modelu ONNX
7. **test_onnx_fixes.py** - Testuje poprawki procesu eksportu na sztucznym modelu

## Sposób 1: Eksport z wiersza poleceń

```bash
# Eksport najnowszego modelu ensemble
python src/export_scripts/export_to_onnx.py

# Eksport konkretnego modelu ensemble
python src/export_scripts/export_to_onnx.py path/to/model.pt output_directory

# Eksport z kwantyzacją
python src/export_scripts/export_to_onnx.py --quantize
```

## Sposób 2: Eksport z notebooka

1. Otwórz notebook `ResNet_ensemble.ipynb`
2. Upewnij się, że model ensemble został już utworzony (zmienne `ensemble_model` i `CONFIG` istnieją)
3. Dodaj nową komórkę i wklej poniższy kod:

```python
from src.export_scripts.export_onnx_notebook_helper import export_model_to_onnx

# Eksport modelu do ONNX z kwantyzacją
result = export_model_to_onnx(
    ensemble_model=ensemble_model,
    class_names=CLASS_NAMES,
    export_params={
        "quantize_model": True,
        "quantization_type": "dynamic",
        "quantization_dtype": "uint8",
        "use_cuda": False
    }
)

if result["success"]:
    print(f"Model wyeksportowany do: {result['output_dir']}")
else:
    print(f"Błąd eksportu: {result.get('error', 'Nieznany błąd')}")
```

## Struktura wynikowych plików

Po eksporcie otrzymasz następujące pliki:

- `ensemble_model.onnx` - Oryginalny model w formacie ONNX
- `ensemble_model_optimized.onnx` - Zoptymalizowana wersja modelu ONNX
- `ensemble_model_quantized.onnx` - Skwantyzowana wersja modelu (jeśli kwantyzacja się powiodła)
- `ensemble_model_metadata.json` - Metadane modelu (cechy wejściowe, wymiary, nazwy klas)

## Wymagania

Do eksportu i uruchomienia modelu ONNX potrzebujesz:

```bash
pip install torch>=2.0.0 onnx>=1.15.0 onnxruntime>=1.19.0 onnxoptimizer>=0.3.13 numpy librosa
```

## Przykładowe użycie eksportowanego modelu

```python
import onnxruntime as ort
import numpy as np
import librosa

# Wczytaj model ONNX
session = ort.InferenceSession(
    "ensemble_model_optimized.onnx", 
    providers=['CPUExecutionProvider']
)

# Przygotuj dane wejściowe (przykład)
# W rzeczywistości należy zastosować odpowiednie przetwarzanie audio
input_data = {
    'mel_input': np.random.randn(1, 1, 128, 130).astype(np.float32),
    'mfcc_input': np.random.randn(1, 1, 40, 130).astype(np.float32),
    'chroma_input': np.random.randn(1, 1, 12, 130).astype(np.float32)
}

# Uruchom inferencję
outputs = session.run(None, input_data)

# Interpretuj wyniki
predicted_emotion_index = np.argmax(outputs[0][0])
emotions = ["anger", "fear", "happiness", "neutral", "sadness", "surprised"]
print(f"Przewidziana emocja: {emotions[predicted_emotion_index]}")
```

## Rozwiązywanie problemów

1. **Problem z plikami modeli** - Upewnij się, że plik `ensemble_model.pt` istnieje w katalogu `ensemble_outputs/ensemble_run_*/models/`.
2. **Problem z eksportem ONNX** - Sprawdź logi błędów w katalogu wyjściowym.
3. **Problem z optymalizacją ONNX** - Sprawdź czy używasz najnowszej wersji `ensemble_onnx_wrapper.py` z poprawkami. Możesz też spróbować użyć nieoptymalizowanej wersji modelu.
4. **Problem z kwantyzacją** - Upewnij się, że masz zainstalowany pakiet `onnxruntime` (a nie tylko `onnxruntime-tools`) i `onnx`.
5. **Problem z ładowaniem modelu PyTorch** - Dla PyTorch 2.7+ używaj `torch.load(path, weights_only=False)` i dodaj bezpieczne globalne.
6. **Problem z inferencją** - Upewnij się, że dane wejściowe mają odpowiednie wymiary i typ danych.
