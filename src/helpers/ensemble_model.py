import torch
import torch.nn as nn
import torch.nn.functional as F
import json

class WeightedEnsembleModel(nn.Module):
    """
    Ważony model zespołowy, który łączy prognozy z wielu modeli różnych typów cech.
    
    Argumenty:
        models_dict (dict): Słownik modeli w formacie {typ_cechy: model}
        weights (dict, optional): Początkowe wagi dla każdego typu cechy
        temperature (float, optional): Parametr temperatury dla kalibracji prawdopodobieństw
        regularization_strength (float, optional): Siła regularyzacji L1 dla wag
    """
    def __init__(self, models_dict, weights=None, temperature=1.0, regularization_strength=0.01):
        super(WeightedEnsembleModel, self).__init__()
        self.models = nn.ModuleDict(models_dict)
        self.feature_types = list(models_dict.keys())
        
        # Inicjalizacja wag
        if weights is None:
            # Równomierne przypisanie wag
            weights_tensor = torch.ones(len(self.feature_types)) / len(self.feature_types)
        else:
            # Wykorzystanie podanego słownika wag
            weights_tensor = torch.tensor([weights[ft] for ft in self.feature_types])
        
        # Umożliwienie uczenia się wag jako parametrów, jeśli to konieczne
        self.weights = nn.Parameter(weights_tensor, requires_grad=False)
        
        # Parametr temperatury do regulacji prognoz
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=False)
        
        # Ustalenie siły regulacji
        self.regularization_strength = regularization_strength
        
        # Generowanie znormalizowanych wag do wnioskowania
        self._update_normalized_weights()
            
    def _update_normalized_weights(self):
        """Aktualizacja znormalizowanych wag"""
        normalized = F.softmax(self.weights, dim=0)
        self.normalized_weights = {ft: normalized[i].item() 
                                  for i, ft in enumerate(self.feature_types)}
        return self.normalized_weights
    
    def forward(self, inputs):
        """
        Przechodzenie do przodu modelu zespołowego.
        
        Argumenty:
            inputs (dict): Słownik tensorów wejściowych w formacie {typ_cechy: tensor}
            
        Zwraca:
            torch.Tensor: Ważona suma prawdopodobieństw z modeli
        """
        outputs = []
        available_features = []
        
        for i, ft in enumerate(self.feature_types):
            if ft in inputs:
                # Uzyskiwanie wyjścia modelu
                model_output = self.models[ft](inputs[ft])
                # Skalowanie wyjścia za pomocą temperatury
                scaled_output = model_output / self.temperature
                # Zastosowanie softmax w celu uzyskania prawdopodobieństw
                probs = F.softmax(scaled_output, dim=1)
                outputs.append(probs)
                available_features.append(i)
        
        if not outputs:
            raise ValueError("Brak danych wejściowych dla modeli")
        
        # Uzyskiwanie wag dla dostępnych cech i ich normalizacja
        available_weights = self.weights[available_features]
        normalized_weights = F.softmax(available_weights, dim=0)
        
        # Zastosowanie wag do wyjścia każdego modelu
        weighted_sum = torch.zeros_like(outputs[0])
        for output, weight in zip(outputs, normalized_weights):
            weighted_sum += output * weight
            
        return weighted_sum
    
    def get_weights(self):
        """Zwracanie aktualnych znormalizowanych wag"""
        return self._update_normalized_weights()
    
    def set_weights(self, weights_dict):
        """
        Ustawianie nowych wag na podstawie słownika
        
        Argumenty:
            weights_dict (dict): Słownik wag w formacie {typ_cechy: waga}
        """
        for i, ft in enumerate(self.feature_types):
            if ft in weights_dict:
                self.weights.data[i] = weights_dict[ft]
        self._update_normalized_weights()
    
    def l1_regularization(self):
        """Zastosowanie regulacji L1 w celu promowania rzadkich wag"""
        return self.regularization_strength * torch.norm(self.weights, p=1)
        
    def save(self, path, class_names=None, version="1.0"):
        """
        Zapis modelu wraz z wagami i parametrami
        """
        # Przygotuj stan do zapisu
        state = {
            'model_state_dict': self.state_dict(),
            'feature_types': list(self.feature_types),
            'normalized_weights': {k: float(v) for k, v in self.normalized_weights.items()},
            'temperature': float(self.temperature.item()),
            'regularization_strength': float(self.regularization_strength),
            'class_names': [str(name) for name in class_names] if class_names is not None else None,
            'model_version': version,
            'pytorch_version': torch.__version__
        }
        
        # Zapisz z obsługą różnych wersji PyTorch
        torch.save(state, path)
        
        # Przygotowanie metadanych w formacie JSON (tylko serializowalne typy)
        json_metadata = {
            'feature_types': list(self.feature_types),
            'normalized_weights': {k: float(v) for k, v in self.normalized_weights.items()},
            'temperature': float(self.temperature.item()),
            'regularization_strength': float(self.regularization_strength),
            'class_names': [str(name) for name in class_names] if class_names is not None else None,
            'model_version': version,
            'pytorch_version': torch.__version__
        }
        
        # Zapisz również metadane oddzielnie dla łatwiejszego dostępu
        metadata_path = path.replace('.pt', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(json_metadata, f, indent=2)
    
        
    @classmethod
    def load(cls, path, models_dict):
        """
        Ładowanie modelu z pliku z obsługą różnych wersji PyTorch
        """
        # Dodanie bezpiecznych globali dla PyTorch 2.6+
        try:
            import torch.serialization
            import numpy as np
            safe_globals = [
                torch.torch_version.TorchVersion,
                np.core.multiarray._reconstruct,
                np.ndarray,
                np.dtype
            ]
            for safe_global in safe_globals:
                try:
                    torch.serialization.add_safe_globals([safe_global])
                except Exception:
                    pass
        except Exception:
            pass
        
        try:
            # Próba z weights_only=True (dla PyTorch 2.6+)
            state = torch.load(path, weights_only=True)
        except Exception:
            try:
                # Jeśli nie działa, próba z weights_only=False (mniej bezpieczna)
                state = torch.load(path, weights_only=False)
            except Exception as e:
                # Jeśli ładowanie dalej nie działa, wyświetl bardziej szczegółowy błąd
                raise RuntimeError(f"Nie można załadować modelu: {str(e)}")
        
        model = cls(
            models_dict=models_dict,
            temperature=state.get('temperature', 1.0),
            regularization_strength=state.get('regularization_strength', 0.01)
        )
        
        # Ładuj stan modelu bezpiecznie
        if 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)  # Próba bezpośredniego ładowania, jeśli brak klucza
        
        model.eval()
        return model, state.get('class_names')
