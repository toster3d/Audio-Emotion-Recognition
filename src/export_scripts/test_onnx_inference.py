"""
Skrypt do testowania i benchmarkingu modeli ONNX ensemble
"""
import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import onnxruntime as ort

# Dodanie ścieżki do helpers
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config import CLASS_NAMES, N_MELS, N_MFCC


class ONNXModelTester:
    """Klasa do testowania modeli ONNX"""
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.session: Optional[ort.InferenceSession] = None
        self.input_shapes = {}
        self.output_shapes = {}
        
        self._load_session()
        self._analyze_model()
    
    def _load_session(self) -> None:
        """Ładowanie sesji ONNX Runtime"""
        print(f"📦 Ładowanie modelu: {self.model_path.name}")
        
        # Dostępne provider'y
        available_providers = ort.get_available_providers()
        print(f"🔌 Dostępne provider'y: {available_providers}")
        
        # Wybór najlepszego provider'a
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("🚀 Używam CUDA provider'a")
        else:
            providers = ['CPUExecutionProvider']
            print("🖥️ Używam CPU provider'a")
        
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers
            )
            print(f"✅ Model załadowany pomyślnie")
            print(f"   Provider: {self.session.get_providers()[0]}")
        except Exception as e:
            raise RuntimeError(f"❌ Błąd ładowania modelu: {e}")
    
    def _analyze_model(self) -> None:
        """Analiza struktury modelu"""
        if not self.session:
            raise RuntimeError("Model nie został załadowany")
            
        print(f"\n🔍 Analiza modelu:")
        
        # Wejścia
        for input_meta in self.session.get_inputs():
            name = input_meta.name
            shape = input_meta.shape
            dtype = input_meta.type
            self.input_shapes[name] = shape
            print(f"   📥 Wejście '{name}': {shape} ({dtype})")
        
        # Wyjścia
        for output_meta in self.session.get_outputs():
            name = output_meta.name
            shape = output_meta.shape
            dtype = output_meta.type
            self.output_shapes[name] = shape
            print(f"   📤 Wyjście '{name}': {shape} ({dtype})")
    
    def generate_test_data(self, batch_size: int = 1, time_steps: int = 300) -> Dict[str, np.ndarray]:
        """Generowanie danych testowych"""
        test_data = {}
        
        for input_name, shape in self.input_shapes.items():
            # Zastąpienie dynamicznych wymiarów
            test_shape = []
            for dim in shape:
                if isinstance(dim, str) or dim == -1:
                    if 'batch' in str(dim).lower():
                        test_shape.append(batch_size)
                    elif 'time' in str(dim).lower():
                        test_shape.append(time_steps)
                    else:
                        test_shape.append(300)  # Domyślny rozmiar
                else:
                    test_shape.append(dim)
            
            # Generowanie losowych danych
            random_data = np.random.randn(*test_shape)
            test_data[input_name] = random_data.astype(np.float32)
            print(f"   🎲 Wygenerowano dane dla '{input_name}': {test_shape}")
        
        return test_data
    
    def run_inference(self, input_data: Dict[str, np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """Wykonanie inference z pomiarem czasu"""
        if not self.session:
            raise RuntimeError("Model nie został załadowany")
            
        start_time = time.perf_counter()
        
        try:
            outputs = self.session.run(None, input_data)
            inference_time = time.perf_counter() - start_time
            return outputs, inference_time
        except Exception as e:
            raise RuntimeError(f"❌ Błąd inference: {e}")
    
    def benchmark_performance(self, 
                            batch_sizes: List[int] = [1, 4, 8, 16],
                            num_iterations: int = 100,
                            warmup_iterations: int = 10) -> Dict:
        """Benchmark wydajności modelu"""
        
        print(f"\n⚡ Benchmark wydajności:")
        print(f"   Iteracje: {num_iterations} (+ {warmup_iterations} warmup)")
        print(f"   Batch sizes: {batch_sizes}")
        
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\n📊 Testowanie batch_size = {batch_size}")
            
            # Generowanie danych
            test_data = self.generate_test_data(batch_size=batch_size)
            
            # Warmup
            for _ in range(warmup_iterations):
                self.run_inference(test_data)
            
            # Właściwy benchmark
            times = []
            for i in range(num_iterations):
                if (i + 1) % 20 == 0:
                    print(f"   Iteracja {i + 1}/{num_iterations}")
                
                _, inference_time = self.run_inference(test_data)
                times.append(inference_time)
            
            # Statystyki
            times_array = np.array(times)
            stats = {
                'batch_size': batch_size,
                'mean_time': float(np.mean(times_array)),
                'std_time': float(np.std(times_array)),
                'min_time': float(np.min(times_array)),
                'max_time': float(np.max(times_array)),
                'p50_time': float(np.percentile(times_array, 50)),
                'p95_time': float(np.percentile(times_array, 95)),
                'p99_time': float(np.percentile(times_array, 99)),
                'throughput': batch_size / np.mean(times_array),  # próbek/sekundę
                'latency_per_sample': np.mean(times_array) / batch_size * 1000  # ms/próbka
            }
            
            results[batch_size] = stats
            
            print(f"   ✅ Średni czas: {stats['mean_time']*1000:.2f} ± {stats['std_time']*1000:.2f} ms")
            print(f"   📈 Throughput: {stats['throughput']:.1f} próbek/s")
            print(f"   ⏱️ Latency: {stats['latency_per_sample']:.2f} ms/próbka")
        
        return results
    
    def test_accuracy(self, test_data: Dict[str, np.ndarray], expected_shape: Optional[Tuple] = None) -> Dict:
        """Test poprawności wyjść"""
        print(f"\n🎯 Test poprawności wyjść:")
        
        outputs, inference_time = self.run_inference(test_data)
        
        results = {
            'inference_time': inference_time,
            'outputs': []
        }
        
        for i, (output_name, expected_shape) in enumerate(zip(self.output_shapes.keys(), [expected_shape] if expected_shape else [None])):
            output = outputs[i]
            
            output_info = {
                'name': output_name,
                'shape': output.shape,
                'dtype': str(output.dtype),
                'min_value': float(np.min(output)),
                'max_value': float(np.max(output)),
                'mean_value': float(np.mean(output)),
                'std_value': float(np.std(output))
            }
            
            # Sprawdzenie czy wyjście ma prawidłowy kształt
            if expected_shape and output.shape != expected_shape:
                output_info['shape_error'] = f"Oczekiwano {expected_shape}, otrzymano {output.shape}"
            
            # Sprawdzenie czy to są prawdopodobieństwa (suma ~1.0)
            if len(output.shape) == 2 and output.shape[1] == len(CLASS_NAMES):
                # Sprawdź czy suma prawdopodobieństw jest bliska 1.0
                prob_sums = np.sum(output, axis=1)
                output_info['probability_check'] = {
                    'is_softmax': bool(np.allclose(prob_sums, 1.0, atol=1e-3)),
                    'prob_sum_mean': float(np.mean(prob_sums)),
                    'prob_sum_std': float(np.std(prob_sums))
                }
                
                # Przewidywane klasy
                predicted_classes = np.argmax(output, axis=1)
                output_info['predictions'] = {
                    'predicted_classes': predicted_classes.tolist(),
                    'max_probabilities': np.max(output, axis=1).tolist(),
                    'predicted_emotions': [CLASS_NAMES[idx] for idx in predicted_classes]
                }
            
            results['outputs'].append(output_info)
            
            print(f"   📤 Wyjście '{output_name}':")
            print(f"      Kształt: {output.shape}")
            print(f"      Zakres: [{output_info['min_value']:.4f}, {output_info['max_value']:.4f}]")
            print(f"      Średnia: {output_info['mean_value']:.4f} ± {output_info['std_value']:.4f}")
            
            if 'probability_check' in output_info:
                prob_check = output_info['probability_check']
                print(f"      Softmax: {prob_check['is_softmax']} (suma: {prob_check['prob_sum_mean']:.4f})")
                
            if 'predictions' in output_info:
                pred = output_info['predictions']
                print(f"      Przewidywane emocje: {pred['predicted_emotions']}")
        
        return results


def compare_models(model_paths: List[str], 
                  batch_sizes: List[int] = [1, 4, 8],
                  num_iterations: int = 50) -> None:
    """Porównanie wydajności wielu modeli"""
    
    print(f"🏁 Porównanie modeli:")
    print(f"   Modele: {len(model_paths)}")
    print(f"   Batch sizes: {batch_sizes}")
    print(f"   Iteracje: {num_iterations}")
    print("=" * 60)
    
    all_results = {}
    
    for model_path in model_paths:
        model_name = Path(model_path).stem
        print(f"\n🧪 Testowanie: {model_name}")
        
        try:
            tester = ONNXModelTester(model_path)
            results = tester.benchmark_performance(
                batch_sizes=batch_sizes,
                num_iterations=num_iterations,
                warmup_iterations=10
            )
            all_results[model_name] = results
            
        except Exception as e:
            print(f"❌ Błąd testowania {model_name}: {e}")
            all_results[model_name] = None
    
    # Podsumowanie porównania
    print(f"\n📊 PODSUMOWANIE PORÓWNANIA")
    print("=" * 60)
    
    for batch_size in batch_sizes:
        print(f"\n📈 Batch size = {batch_size}:")
        print(f"{'Model':<25} {'Latency (ms)':<15} {'Throughput (sps)':<20} {'P95 (ms)':<12}")
        print("-" * 72)
        
        for model_name, results in all_results.items():
            if results and batch_size in results:
                stats = results[batch_size]
                latency = stats['latency_per_sample']
                throughput = stats['throughput']
                p95 = stats['p95_time'] * 1000
                
                print(f"{model_name:<25} {latency:<15.2f} {throughput:<20.1f} {p95:<12.2f}")
            else:
                print(f"{model_name:<25} {'ERROR':<15} {'-':<20} {'-':<12}")


def main():
    parser = argparse.ArgumentParser(description='Test i benchmark modeli ONNX')
    parser.add_argument('--model_path', type=str, help='Ścieżka do modelu ONNX')
    parser.add_argument('--model_dir', type=str, help='Katalog z modelami do porównania')
    parser.add_argument('--batch_sizes', nargs='+', type=int, 
                       default=[1, 4, 8], help='Rozmiary batch do testowania')
    parser.add_argument('--iterations', type=int, default=100, 
                       help='Liczba iteracji benchmarku')
    parser.add_argument('--compare', action='store_true', 
                       help='Porównaj wszystkie modele w katalogu')
    parser.add_argument('--output', type=str, help='Plik wyjściowy JSON z wynikami')
    
    args = parser.parse_args()
    
    if args.compare and args.model_dir:
        # Porównanie modeli
        model_dir = Path(args.model_dir)
        model_paths = list(model_dir.glob("*.onnx"))
        
        if not model_paths:
            print(f"❌ Nie znaleziono modeli ONNX w {model_dir}")
            return
        
        compare_models(
            [str(p) for p in model_paths],
            batch_sizes=args.batch_sizes,
            num_iterations=args.iterations
        )
        
    elif args.model_path:
        # Test pojedynczego modelu
        print(f"🧪 Test modelu: {args.model_path}")
        
        try:
            tester = ONNXModelTester(args.model_path)
            
            # Test poprawności
            test_data = tester.generate_test_data(batch_size=2)
            accuracy_results = tester.test_accuracy(test_data)
            
            # Benchmark wydajności
            performance_results = tester.benchmark_performance(
                batch_sizes=args.batch_sizes,
                num_iterations=args.iterations
            )
            
            # Zapis wyników
            if args.output:
                results = {
                    'model_path': args.model_path,
                    'accuracy_test': accuracy_results,
                    'performance_benchmark': performance_results,
                    'test_config': {
                        'batch_sizes': args.batch_sizes,
                        'iterations': args.iterations
                    }
                }
                
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"\n💾 Wyniki zapisane do: {args.output}")
            
        except Exception as e:
            print(f"❌ Błąd testowania: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        # Automatyczne wyszukiwanie modeli
        print("🔍 Wyszukiwanie modeli ONNX...")
        
        # Sprawdź folder exported_models
        possible_dirs = [
            "exported_models",
            "../exported_models",
            ".",
        ]
        
        found_models = []
        for directory in possible_dirs:
            if os.path.exists(directory):
                models = list(Path(directory).rglob("*.onnx"))
                found_models.extend(models)
        
        if found_models:
            print(f"📦 Znaleziono {len(found_models)} modeli:")
            for i, model in enumerate(found_models, 1):
                size_mb = model.stat().st_size / (1024 * 1024)
                print(f"   {i}. {model} ({size_mb:.2f} MB)")
            
            # Test najnowszego modelu
            latest_model = max(found_models, key=lambda p: p.stat().st_mtime)
            print(f"\n🎯 Testowanie najnowszego modelu: {latest_model}")
            
            tester = ONNXModelTester(str(latest_model))
            test_data = tester.generate_test_data(batch_size=1)
            accuracy_results = tester.test_accuracy(test_data)
            
        else:
            print("❌ Nie znaleziono żadnych modeli ONNX")
            print("   Uruchom najpierw eksport modelu")


if __name__ == "__main__":
    main() 