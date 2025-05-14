#!/usr/bin/env python
"""
Skrypt do testowania kwantyzacji modelu ONNX.

Kwantyzacja może znacząco zmniejszyć rozmiar modelu, ale może też
wpłynąć na jego dokładność. Ten skrypt pozwala na sprawdzenie
tego kompromisu.
"""

import os
import sys
import json
from datetime import datetime

# Importy zewnętrzne
import numpy as np
import matplotlib.pyplot as plt

# Import niezbędnych funkcji
from export_scripts.ensemble_onnx_wrapper import (
    quantize_onnx_model,
    create_onnx_inference_session,
    run_onnx_inference
)
from export_scripts.test_without_audio import (
    generate_random_features,
    get_class_names,
    find_latest_model
)
from src.config import N_MELS, N_MFCC, N_CHROMA, CLASS_NAMES

def apply_quantization(input_path, output_dir, quantization_types=None, quantization_dtypes=None):
    """Zastosuj różne techniki kwantyzacji do modelu."""
    if quantization_types is None:
        quantization_types = ["dynamic"]
    
    if quantization_dtypes is None:
        quantization_dtypes = ["uint8", "int8"]
    
    results = {}
    
    # Utwórz katalog wyjściowy, jeśli nie istnieje
    os.makedirs(output_dir, exist_ok=True)
    
    # Kwantyzacja wszystkimi kombinacjami typów i formatów danych
    for q_type in quantization_types:
        for q_dtype in quantization_dtypes:
            try:
                name = f"{q_type}_{q_dtype}"
                output_path = os.path.join(output_dir, f"model_quantized_{name}.onnx")
                
                print(f"\n--- Kwantyzacja {name} ---")
                result = quantize_onnx_model(
                    input_path,
                    output_path,
                    quantization_type=q_type,
                    dtype=q_dtype
                )
                
                if result.get("success", False):
                    result["name"] = name
                    result["output_path"] = output_path
                    result["size_mb"] = os.path.getsize(output_path) / (1024 * 1024)
                    results[name] = result
                    print(f"✅ Kwantyzacja {name} zakończona sukcesem")
                    print(f"   Rozmiar: {result['size_mb']:.2f} MB")
                    print(f"   Redukcja rozmiaru: {result['reduction_percent']:.2f}%")
                else:
                    print(f"❌ Kwantyzacja {name} nie powiodła się: {result.get('error', 'Nieznany błąd')}")
            except Exception as e:
                print(f"❌ Błąd podczas kwantyzacji {q_type}_{q_dtype}: {e}")
    
    return results

def compare_models(original_path, quantized_results, class_names=None):
    """Porównaj wyniki inferencji z różnych wersji modelu."""
    # Generowanie losowych danych wejściowych
    features = generate_random_features(batch_size=5)
    
    # Pobierz nazwy klas (z metadanych lub domyślne)
    if class_names is None:
        class_names = get_class_names(original_path)
    
    # Wyniki dla modelu oryginalnego
    print("\n--- Inferencja na modelu oryginalnym ---")
    try:
        session_original = create_onnx_inference_session(original_path)
        outputs_original = run_onnx_inference(session_original, features)
        print(f"Kształt wyjścia: {outputs_original.shape}")
        
        # Interpretacja wyników dla wszystkich próbek
        probabilities_original = outputs_original
        
        # Sprawdź wymiary wyjścia - może być różnie zależnie od implementacji modelu
        if len(probabilities_original.shape) == 2:
            # Wyjście ma kształt [batch_size, num_classes]
            emotion_indices_original = np.argmax(probabilities_original, axis=1)
        else:
            # Wyjście może mieć inny kształt, np. [batch_size, 1, num_classes]
            # Dostosuj go do oczekiwanego kształtu [batch_size, num_classes]
            if len(probabilities_original.shape) == 3 and probabilities_original.shape[1] == 1:
                probabilities_original = probabilities_original.squeeze(1)
                emotion_indices_original = np.argmax(probabilities_original, axis=1)
            else:
                # Jeśli wymiary są inne, to wykonaj argmax po ostatnim wymiarze
                emotion_indices_original = np.argmax(probabilities_original, axis=-1)
                # Spłaszcz indeksy do tablicy 1D
                emotion_indices_original = emotion_indices_original.flatten()
        
        emotions_original = [class_names[idx] for idx in emotion_indices_original]
        
        print("Przewidywane emocje (oryginał):")
        for i, emotion in enumerate(emotions_original):
            confidence = probabilities_original.flatten()[i * len(class_names) + emotion_indices_original[i]]
            print(f"  Próbka {i+1}: {emotion} ({confidence:.2%})")
        
        # Wyniki dla modeli kwantyzowanych
        comparison_results = {
            "original": {
                "model_path": original_path,
                "predictions": emotions_original,
                "confidences": [float(probabilities_original.flatten()[i * len(class_names) + emotion_indices_original[i]]) for i in range(len(emotions_original))],
                "size_mb": os.path.getsize(original_path) / (1024 * 1024)
            }
        }
        
        # Testowanie każdego skwantyzowanego modelu
        for name, result in quantized_results.items():
            print(f"\n--- Inferencja na modelu {name} ---")
            
            try:
                session = create_onnx_inference_session(result["output_path"])
                outputs = run_onnx_inference(session, features)
                
                # Interpretacja wyników
                probabilities = outputs
                
                # Dostosuj wymiary tak samo jak dla oryginalnego modelu
                if len(probabilities.shape) == 2:
                    emotion_indices = np.argmax(probabilities, axis=1)
                else:
                    if len(probabilities.shape) == 3 and probabilities.shape[1] == 1:
                        probabilities = probabilities.squeeze(1)
                        emotion_indices = np.argmax(probabilities, axis=1)
                    else:
                        emotion_indices = np.argmax(probabilities, axis=-1)
                        emotion_indices = emotion_indices.flatten()
                
                emotions = [class_names[idx] for idx in emotion_indices]
                
                print(f"Przewidywane emocje ({name}):")
                for i, emotion in enumerate(emotions):
                    confidence = probabilities.flatten()[i * len(class_names) + emotion_indices[i]]
                    print(f"  Próbka {i+1}: {emotion} ({confidence:.2%})")
                
                # Oblicz zgodność z modelem oryginalnym
                agreement = sum([1 for i in range(len(emotions)) if emotions[i] == emotions_original[i]]) / len(emotions)
                print(f"Zgodność z modelem oryginalnym: {agreement:.2%}")
                
                # Zapisz wyniki
                comparison_results[name] = {
                    "model_path": result["output_path"],
                    "predictions": emotions,
                    "confidences": [float(probabilities.flatten()[i * len(class_names) + emotion_indices[i]]) for i in range(len(emotions))],
                    "agreement": float(agreement),
                    "size_mb": result["size_mb"],
                    "reduction_percent": result["reduction_percent"]
                }
            except Exception as e:
                print(f"❌ Błąd podczas inferencji na modelu {name}: {e}")
                import traceback
                print(traceback.format_exc())
        
        return comparison_results
    
    except Exception as e:
        print(f"❌ Błąd podczas inferencji na modelu oryginalnym: {e}")
        import traceback
        print(traceback.format_exc())
        return None

def visualize_comparison(comparison_results, output_dir):
    """Wizualizuj porównanie modeli."""
    if not comparison_results:
        print("Brak wyników do wizualizacji")
        return
    
    # Utwórz katalog na wizualizacje
    os.makedirs(output_dir, exist_ok=True)
    
    # Porównanie rozmiarów modeli
    models = list(comparison_results.keys())
    sizes = [comparison_results[model]["size_mb"] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, sizes)
    plt.title('Porównanie rozmiarów modeli')
    plt.ylabel('Rozmiar [MB]')
    plt.xticks(rotation=45)
    
    # Dodaj etykiety z dokładnymi wartościami
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f} MB', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_sizes_comparison.png"))
    
    # Porównanie zgodności z modelem oryginalnym (tylko dla modeli kwantyzowanych)
    quantized_models = [model for model in models if model != "original"]
    if quantized_models:
        agreements = [comparison_results[model].get("agreement", 0) * 100 for model in quantized_models]
        reductions = [comparison_results[model].get("reduction_percent", 0) for model in quantized_models]
        
        # Wykres zgodności i redukcji rozmiaru
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Zgodność (lewa oś Y)
        color = 'tab:blue'
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Zgodność [%]', color=color)
        bars1 = ax1.bar(quantized_models, agreements, color=color, alpha=0.7)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 105)  # Od 0 do 105%
        
        # Etykiety na słupkach
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', color=color)
        
        # Redukcja rozmiaru (prawa oś Y)
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Redukcja rozmiaru [%]', color=color)
        bars2 = ax2.bar(quantized_models, reductions, color=color, alpha=0.4)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, max(reductions) * 1.1)  # Od 0 do maksymalnej wartości + 10%
        
        # Etykiety na słupkach
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', color=color)
        
        plt.title('Porównanie zgodności predykcji i redukcji rozmiaru')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "quantization_comparison.png"))
        
        # Zapisz szczegółowe wyniki do pliku JSON
        results_json_path = os.path.join(output_dir, "quantization_results.json")
        with open(results_json_path, "w") as f:
            json.dump(comparison_results, f, indent=2)
        
        print(f"\nWizualizacje i wyniki zapisane w katalogu: {output_dir}")
    
    plt.close('all')

def main(model_path=None):
    """Główna funkcja."""
    # Znajdź najnowszy model, jeśli nie podano
    if model_path is None:
        model_path = find_latest_model()
        if model_path:
            print(f"Znaleziono najnowszy model: {model_path}")
        else:
            print("Nie znaleziono modelu ONNX.")
            print("Użycie: python test_quantization.py [ścieżka_do_modelu_onnx]")
            return False
    
    # Utwórz katalog na wyniki
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"test_results/quantization_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Zastosuj różne techniki kwantyzacji
    print(f"\n=== Testowanie kwantyzacji modelu: {model_path} ===\n")
    quantization_results = apply_quantization(
        model_path,
        output_dir,
        quantization_types=["dynamic"],
        quantization_dtypes=["uint8", "int8"]
    )
    
    if not quantization_results:
        print("Nie udało się wykonać kwantyzacji modelu.")
        return False
    
    # Porównaj wyniki inferencji
    print("\n=== Porównanie wyników inferencji ===\n")
    comparison_results = compare_models(model_path, quantization_results)
    
    if comparison_results:
        # Wizualizuj porównanie
        visualize_comparison(comparison_results, output_dir)
        return True
    else:
        print("❌ Nie udało się porównać modeli.")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()