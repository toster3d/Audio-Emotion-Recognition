import torch
import torch.nn as nn
import torch.nn.functional as F

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
            # Inicjalizacja równymi wagami
            weights_tensor = torch.ones(len(self.feature_types)) / len(self.feature_types)
        else:
            # Użyj podanego słownika wag
            weights_tensor = torch.tensor([weights[ft] for ft in self.feature_types])
        
        # Uczyń wagi parametrami uczącymi się, jeśli to konieczne
        self.weights = nn.Parameter(weights_tensor, requires_grad=False)
        
        # Parametr temperatury do ostrzenia/łagodzenia prognoz
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=False)
        
        # Siła regularizacji
        self.regularization_strength = regularization_strength
        
        # Generowanie znormalizowanych wag do wnioskowania
        self._update_normalized_weights()
            
    def _update_normalized_weights(self):
        """Aktualizuje właściwość znormalizowanych wag"""
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
                # Uzyskaj wyjście modelu
                model_output = self.models[ft](inputs[ft])
                # Zastosuj skalowanie temperatury
                scaled_output = model_output / self.temperature
                # Zastosuj softmax, aby uzyskać prawdopodobieństwa
                probs = F.softmax(scaled_output, dim=1)
                outputs.append(probs)
                available_features.append(i)
        
        if not outputs:
            raise ValueError("Nie podano żadnych danych wejściowych dla żadnego modelu")
        
        # Uzyskaj wagi dla dostępnych cech i znormalizuj je
        available_weights = self.weights[available_features]
        normalized_weights = F.softmax(available_weights, dim=0)
        
        # Zastosuj wagi do wyjścia każdego modelu
        weighted_sum = torch.zeros_like(outputs[0])
        for output, weight in zip(outputs, normalized_weights):
            weighted_sum += output * weight
            
        return weighted_sum
    
    def get_weights(self):
        """Zwraca aktualne znormalizowane wagi"""
        return self._update_normalized_weights()
    
    def set_weights(self, weights_dict):
        """
        Ustaw nowe wagi ze słownika
        
        Argumenty:
            weights_dict (dict): Słownik wag w formacie {typ_cechy: waga}
        """
        for i, ft in enumerate(self.feature_types):
            if ft in weights_dict:
                self.weights.data[i] = weights_dict[ft]
        self._update_normalized_weights()
    
    def l1_regularization(self):
        """Zastosuj regularizację L1, aby zachęcić do rzadkich wag"""
        return self.regularization_strength * torch.norm(self.weights, p=1)
        
    def save(self, path):
        """
        Zapisz model wraz z wagami i parametrami
        
        Argumenty:
            path (str): Ścieżka do zapisu modelu
        """
        state = {
            'model_state_dict': self.state_dict(),
            'feature_types': self.feature_types,
            'normalized_weights': self.normalized_weights,
            'temperature': self.temperature.item(),
            'regularization_strength': self.regularization_strength
        }
        torch.save(state, path)
        
    @classmethod
    def load(cls, path, models_dict):
        """
        Załaduj model z pliku
        
        Argumenty:
            path (str): Ścieżka do pliku modelu
            models_dict (dict): Słownik modeli w formacie {typ_cechy: model}
            
        Zwraca:
            WeightedEnsembleModel: Załadowany model
        """
        state = torch.load(path)
        # Utwórz model z tymi samymi parametrami
        model = cls(
            models_dict=models_dict,
            temperature=state['temperature'],
            regularization_strength=state['regularization_strength']
        )
        # Załaduj stan
        model.load_state_dict(state['model_state_dict'])
        return model