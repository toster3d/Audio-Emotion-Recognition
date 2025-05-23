#!/usr/bin/env python3
"""
Nowoczesny eksporter modelu ensemble do ONNX z obsługą 5 typów cech.
Zgodny z PyTorch 2.7+ i najlepszymi praktykami ONNX.

Obsługuje:
- chroma, tempogram, mfcc, melspectrogram, hpss (5 typów cech)
- Bezpieczne ładowanie modeli
- Optymalizację ONNX
- Kwantyzację
- Weryfikację modeli
"""

import os
import sys
import json
import warnings
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort

# Dodaj ścieżkę projektu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CLASS_NAMES, MAX_LENGTH, N_FFT, HOP_LENGTH, N_MELS, N_MFCC, N_CHROMA
from helpers.ensemble_model import WeightedEnsembleModel
from helpers.resnet_model_definition import AudioResNet

# Stałe dla cech audio
N_CQT = 84
N_TEMPOGRAM = 384  # Typowy rozmiar tempogram

@dataclass
class ExportConfig:
    """Konfiguracja eksportu modelu"""
    model_path: Optional[str] = None
    output_dir: Optional[str] = None
    sample_rate: int = 22050
    max_length: float = MAX_LENGTH
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    opset_version: int = 19  # Najnowsza stabilna wersja ONNX opset
    enable_optimization: bool = True
    enable_quantization: bool = False
    quantization_type: str = "dynamic"  # "dynamic" lub "static"
    fp16_conversion: bool = False
    verify_model: bool = True
    export_metadata: bool = True


class ModernEnsembleONNXWrapper(torch.nn.Module):
    """Wrapper modelu ensemble dla ONNX z obsługą 5 typów cech"""
    
    def __init__(self, ensemble_model: WeightedEnsembleModel):
        super().__init__()
        self.ensemble = ensemble_model
        self.feature_types = ensemble_model.feature_types
        
        # Mapowanie rozmiarów cech
        self.feature_sizes = self._get_feature_sizes()
        
    def _get_feature_sizes(self) -> Dict[str, int]:
        """Określa rozmiary każdego typu cechy"""
        sizes = {}
        for feature_type in self.feature_types:
            if feature_type == 'chroma':
                sizes[feature_type] = N_CHROMA
            elif feature_type == 'tempogram':
                sizes[feature_type] = N_TEMPOGRAM
            elif feature_type == 'mfcc':
                sizes[feature_type] = N_MFCC
            elif feature_type == 'melspectrogram':
                sizes[feature_type] = N_MELS
            elif feature_type == 'hpss':
                sizes[feature_type] = N_MELS
            elif feature_type == 'cqt':
                sizes[feature_type] = N_CQT
            else:
                # Domyślny rozmiar dla nieznanych typów
                sizes[feature_type] = N_MELS
        return sizes
        
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - dzieli tensor na poszczególne cechy
        
        Args:
            audio_features: Tensor [batch, 1, total_features, time]
            
        Returns:
            output: Tensor z prawdopodobieństwami klas [batch, num_classes]
        """
        # Podział tensora na poszczególne cechy
        inputs_dict = {}
        start_idx = 0
        
        for feature_type in self.feature_types:
            size = self.feature_sizes[feature_type]
            end_idx = start_idx + size
            
            # Wyciągnięcie fragmentu tensora dla danej cechy
            feature_tensor = audio_features[:, :, start_idx:end_idx, :]
            inputs_dict[feature_type] = feature_tensor
            
            start_idx = end_idx
        
        # Przetworzenie przez model ensemble
        return self.ensemble(inputs_dict)


class ModernONNXExporter:
    """Nowoczesny eksporter modelu ensemble do ONNX"""
    
    def __init__(self, config: ExportConfig):
        self.config = config
        self.export_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Automatyczne tworzenie katalogu wyjściowego jeśli nie podano
        if not config.output_dir:
            self.config.output_dir = f"exported_models/onnx_{self.export_timestamp}"
        
        # Bezpieczne tworzenie katalogu
        output_path = Path(self.config.output_dir) if self.config.output_dir else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
        
    def load_ensemble_model(self) -> Tuple[WeightedEnsembleModel, List[str]]:
        """Ładowanie modelu ensemble z obsługą PyTorch 2.7+"""
        if not self.config.model_path:
            # Automatyczne wyszukanie najnowszego modelu
            import glob
            model_files = glob.glob("ensemble_outputs/ensemble_run_*/models/ensemble_model.pt")
            if model_files:
                self.config.model_path = sorted(model_files)[-1]
                print(f"🔍 Automatycznie wykryto model: {self.config.model_path}")
            else:
                raise FileNotFoundError("❌ Nie znaleziono modeli ensemble")
        
        # Ładowanie modelu
        print(f"📥 Ładowanie modelu: {self.config.model_path}")
        
        # Strategia ładowania zgodna z PyTorch 2.7+
        state = None
        try:
            # Próba 1: Standardowe bezpieczne ładowanie
            state = torch.load(
                self.config.model_path, 
                map_location='cpu',
                weights_only=True
            )
        except Exception as e1:
            try:
                # Próba 2: Z dodanym bezpiecznym globalnym obiektem
                import torch.version as torch_version
                torch.serialization.add_safe_globals([torch_version.TorchVersion]) # type: ignore
                state = torch.load(
                    self.config.model_path, 
                    map_location='cpu',
                    weights_only=True
                )
            except Exception as e2:
                # Próba 3: Ostateczny fallback - ładowanie bez ograniczeń
                print(f"⚠️ Używam fallback ładowania (weights_only=False)")
                print(f"   Błąd bezpiecznego ładowania: {str(e1)[:100]}...")
                state = torch.load(
                    self.config.model_path, 
                    map_location='cpu',
                    weights_only=False
                )
        
        # Ekstraktowanie informacji o modelu
        if isinstance(state, dict) and 'model_state_dict' in state:
            # Obsługa wszystkich 5 typów cech
            feature_types = state.get('feature_types', 
                                     ['chroma', 'tempogram', 'mfcc', 'melspectrogram', 'hpss'])
            class_names = state.get('class_names', CLASS_NAMES)
        else:
            raise ValueError("❌ Nieprawidłowy format pliku modelu")
        
        # Rekonstrukcja modeli składowych
        models_dict = {}
        for feature_type in feature_types:
            # Tworzenie modelu składowego
            model = AudioResNet(num_classes=len(class_names))
            models_dict[feature_type] = model
        
        # Tworzenie modelu ensemble
        weights = state.get('normalized_weights')
        temperature = state.get('temperature', 1.0)
        reg_strength = state.get('regularization_strength', 0.01)
        
        ensemble_model = WeightedEnsembleModel(
            models_dict=models_dict,
            weights=weights,
            temperature=temperature,
            regularization_strength=reg_strength
        )
        
        # Ładowanie wag
        ensemble_model.load_state_dict(state['model_state_dict'])
        ensemble_model.eval()
        
        print(f"✅ Model załadowany z {len(feature_types)} typami cech: {feature_types}")
        return ensemble_model, feature_types
    
    def generate_dummy_input(self, feature_types: List[str]) -> torch.Tensor:
        """Generowanie przykładowych danych wejściowych dla wszystkich 5 typów cech"""
        # Obliczanie wymiarów
        num_frames = int(self.config.sample_rate * self.config.max_length / self.config.hop_length) + 1
        
        # Obliczanie całkowitego rozmiaru cech
        total_features = 0
        feature_info = {}
        
        for feature_type in feature_types:
            if feature_type == 'chroma':
                size = N_CHROMA
            elif feature_type == 'tempogram':
                size = N_TEMPOGRAM
            elif feature_type == 'mfcc':
                size = N_MFCC
            elif feature_type == 'melspectrogram':
                size = N_MELS
            elif feature_type == 'hpss':
                size = N_MELS
            elif feature_type == 'cqt':
                size = N_CQT
            else:
                # Domyślny rozmiar dla nieznanych typów
                size = N_MELS
                print(f"⚠️ Nieznany typ cechy '{feature_type}', używam domyślnego rozmiaru {size}")
                
            feature_info[feature_type] = size
            total_features += size
        
        # Tworzenie tensora z wszystkimi cechami
        dummy_input = torch.randn(1, 1, total_features, num_frames)
        
        print(f"📊 Wymiary wejścia: {dummy_input.shape}")
        print(f"📋 Informacje o cechach: {feature_info}")
        print(f"🔢 Całkowity rozmiar cech: {total_features}")
        
        return dummy_input
    
    def export_to_onnx(self, model: WeightedEnsembleModel, feature_types: List[str]) -> Dict[str, Any]:
        """Główna funkcja eksportu do ONNX"""
        print(f"\n🚀 Rozpoczynam eksport do ONNX...")
        
        # Tworzenie wrapper'a
        wrapper = ModernEnsembleONNXWrapper(model)
        wrapper.eval()
        
        # Generowanie danych wejściowych
        dummy_input = self.generate_dummy_input(feature_types)
        
        # Testowanie forward pass
        print("🔍 Test forward pass...")
        with torch.no_grad():
            output = wrapper(dummy_input)
            print(f"✅ Test ukończony. Wymiary wyjścia: {output.shape}")
        
        # Ścieżki plików
        if not self.config.output_dir:
            raise ValueError("❌ Brak określonego katalogu wyjściowego")
            
        onnx_path = Path(self.config.output_dir) / "ensemble_model.onnx"
        
        # Eksport do ONNX
        try:
            print("📦 Eksport do ONNX...")
            
            # Konfiguracja dynamicznych osi
            dynamic_axes = {
                'audio_features': {0: 'batch_size', 3: 'time_steps'},
                'output': {0: 'batch_size'}
            }
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore")
                
                torch.onnx.export(
                    wrapper,
                    (dummy_input,),  # Tuple z argumentami
                    str(onnx_path),
                    input_names=['audio_features'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes,
                    opset_version=self.config.opset_version,
                    do_constant_folding=True,
                    export_params=True,
                    training=torch.onnx.TrainingMode.EVAL,
                    verbose=False
                )
            
            print(f"✅ Eksport ukończony: {onnx_path}")
            
            # Weryfikacja modelu
            if self.config.verify_model:
                self._verify_onnx_model(onnx_path, dummy_input, wrapper)
            
            result = {
                "success": True,
                "path": str(onnx_path),
                "size_mb": onnx_path.stat().st_size / (1024 * 1024),
                "feature_types": feature_types
            }
            
            # Optymalizacja
            if self.config.enable_optimization:
                optimized_path = self._optimize_model(onnx_path)
                result["optimized_path"] = str(optimized_path)
            
            # Kwantyzacja
            if self.config.enable_quantization:
                quantized_path = self._quantize_model(onnx_path, dummy_input)
                result["quantized_path"] = str(quantized_path)
            
            # Eksport metadanych
            if self.config.export_metadata:
                self._export_metadata(feature_types, result)
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Błąd eksportu ONNX: {e}"
            print(error_msg)
            traceback.print_exc()
            return {"success": False, "error": error_msg}
    
    def _verify_onnx_model(self, onnx_path: Path, dummy_input: torch.Tensor, wrapper: torch.nn.Module) -> bool:
        """Weryfikacja modelu ONNX"""
        print("🔍 Weryfikacja modelu ONNX...")
        
        try:
            # Ładowanie i sprawdzanie modelu
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # Test inference
            ort_session = ort.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider']
            )
            
            # Porównanie wyników PyTorch vs ONNX
            with torch.no_grad():
                pytorch_output = wrapper(dummy_input)
            
            onnx_output = ort_session.run(
                None,
                {'audio_features': dummy_input.numpy()}
            )[0]
            
            # Sprawdzenie różnicy
            max_diff = np.max(np.abs(pytorch_output.numpy() - onnx_output))
            print(f"✅ Weryfikacja zakończona. Max różnica: {max_diff:.6f}")
            
            if max_diff > 1e-4:
                print(f"⚠️ Ostrzeżenie: Duża różnica między modelami ({max_diff:.6f})")
            
            return True
            
        except Exception as e:
            print(f"❌ Błąd weryfikacji: {e}")
            return False
    
    def _optimize_model(self, onnx_path: Path) -> Path:
        """Optymalizacja modelu ONNX"""
        optimized_path = onnx_path.parent / f"{onnx_path.stem}_optimized.onnx"
        
        try:
            print("⚡ Optymalizacja modelu ONNX...")
            
            # Podstawowa optymalizacja z onnx
            model = onnx.load(str(onnx_path))
            
            # Zastosuj podstawowe optymalizacje dostępne w onnx
            from onnx import optimizer as onnx_optimizer # type: ignore
            passes = ['eliminate_unused_initializer', 'eliminate_identity']
            optimized_model = onnx_optimizer.optimize(model, passes)
            
            onnx.save(optimized_model, str(optimized_path))
            print(f"✅ Model zoptymalizowany: {optimized_path}")
            
            return optimized_path
            
        except Exception as e:
            print(f" Optymalizacja nieudana: {e}")
            # Zwróć oryginalną ścieżkę jako fallback
            return onnx_path
    
    def _quantize_model(self, onnx_path: Path, dummy_input: torch.Tensor) -> Path:
        """Kwantyzacja modelu ONNX"""
        quantized_path = onnx_path.parent / f"{onnx_path.stem}_quantized.onnx"
        
        try:
            print(f"🔢 Kwantyzacja modelu ({self.config.quantization_type})...")
            
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            if self.config.quantization_type == "dynamic":
                quantize_dynamic(
                    str(onnx_path),
                    str(quantized_path),
                    weight_type=QuantType.QInt8
                )
            else:
                # Statyczna kwantyzacja wymaga danych kalibracyjnych
                print("⚠️ Statyczna kwantyzacja wymaga implementacji danych kalibracyjnych")
                return onnx_path
            
            print(f"✅ Model skwantyzowany: {quantized_path}")
            return quantized_path
            
        except Exception as e:
            print(f"⚠️ Kwantyzacja nieudana: {e}")
            return onnx_path
    
    def _export_metadata(self, feature_types: List[str], export_result: Dict[str, Any]) -> None:
        """Eksport metadanych"""
        if not self.config.output_dir:
            return
            
        metadata = {
            "export_info": {
                "timestamp": self.export_timestamp,
                "exporter_version": "2.0",
                "pytorch_version": torch.__version__,
                "onnx_version": getattr(onnx, '__version__', 'unknown'),
                "onnxruntime_version": ort.__version__
            },
            "model_info": {
                "feature_types": feature_types,
                "num_features": len(feature_types),
                "class_names": CLASS_NAMES,
                "num_classes": len(CLASS_NAMES)
            },
            "config": {
                "opset_version": self.config.opset_version,
                "sample_rate": self.config.sample_rate,
                "max_length": self.config.max_length,
                "optimization_enabled": self.config.enable_optimization,
                "quantization_enabled": self.config.enable_quantization,
                "quantization_type": self.config.quantization_type
            },
            "export_result": export_result
        }
        
        metadata_path = Path(self.config.output_dir) / "export_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Metadane zapisane: {metadata_path}")


def main():
    """Główna funkcja CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Nowoczesny eksport modelu ensemble do ONNX')
    parser.add_argument('--model-path', type=str, help='Ścieżka do modelu ensemble')
    parser.add_argument('--output-dir', type=str, help='Katalog wyjściowy')
    parser.add_argument('--optimize', action='store_true', help='Włącz optymalizację')
    parser.add_argument('--quantize', action='store_true', help='Włącz kwantyzację')
    parser.add_argument('--quantization-type', choices=['dynamic', 'static'], 
                       default='dynamic', help='Typ kwantyzacji')
    parser.add_argument('--opset-version', type=int, default=19, help='Wersja ONNX opset')
    
    args = parser.parse_args()
    
    # Tworzenie konfiguracji
    config = ExportConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        enable_optimization=args.optimize,
        enable_quantization=args.quantize,
        quantization_type=args.quantization_type,
        opset_version=args.opset_version
    )
    
    # Eksport
    exporter = ModernONNXExporter(config)
    
    try:
        model, feature_types = exporter.load_ensemble_model()
        result = exporter.export_to_onnx(model, feature_types)
        
        if result["success"]:
            print(f"\ Eksport zakończony sukcesem!")
            print(f"📁 Pliki w: {config.output_dir}")
        else:
            print(f"\nEksport nieudany: {result.get('error', 'Nieznany błąd')}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nKrytyczny błąd: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 