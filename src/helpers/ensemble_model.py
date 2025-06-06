import json

import numpy as np
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

    def __init__(
        self, models_dict, weights=None, temperature=1.0, regularization_strength=0.01
    ):
        super().__init__()
        self.models = nn.ModuleDict(models_dict)
        self.feature_types = list(models_dict.keys())

        # Inicjalizacja wag
        if weights is None:
            # Równomierne przypisanie wag
            weights_tensor = torch.ones(len(self.feature_types)) / len(
                self.feature_types
            )
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
        self.normalized_weights = {
            ft: normalized[i].item() for i, ft in enumerate(self.feature_types)
        }
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
        # Poprawka: Unikanie konwersji tensora wag na listę, zamiast tego używamy indeksowania
        weighted_sum = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            weight = normalized_weights[i]  # Bezpośrednie indeksowanie tensora wag
            weighted_sum = (
                weighted_sum + output * weight
            )  # Używamy dodawania zamiast += dla kompatybilności z tracerem

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
            "model_state_dict": self.state_dict(),
            "feature_types": list(self.feature_types),
            "normalized_weights": {
                k: float(v) for k, v in self.normalized_weights.items()
            },
            "temperature": float(self.temperature.item()),
            "regularization_strength": float(self.regularization_strength),
            "class_names": [str(name) for name in class_names]
            if class_names is not None
            else None,
            "model_version": version,
            "pytorch_version": torch.__version__,
        }

        # Zapisz z obsługą różnych wersji PyTorch
        torch.save(state, path)

        # Przygotowanie metadanych w formacie JSON (tylko serializowalne typy)
        json_metadata = {
            "feature_types": list(self.feature_types),
            "normalized_weights": {
                k: float(v) for k, v in self.normalized_weights.items()
            },
            "temperature": float(self.temperature.item()),
            "regularization_strength": float(self.regularization_strength),
            "class_names": [str(name) for name in class_names]
            if class_names is not None
            else None,
            "model_version": version,
            "pytorch_version": torch.__version__,
        }

        # Zapisz również metadane oddzielnie dla łatwiejszego dostępu
        metadata_path = path.replace(".pt", "_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(json_metadata, f, indent=2)

    @classmethod
    def load(cls, path, models_dict):
        """
        Ładowanie modelu z pliku z obsługą różnych wersji PyTorch

        Args:
            path: Ścieżka do pliku modelu
            models_dict: Słownik modeli bazowych

        Returns:
            tuple: (załadowany_model, nazwy_klas)
        """
        # Konfiguracja bezpiecznego ładowania dla PyTorch
        try:
            # Dodaj bezpieczne typy danych dla PyTorch
            import torch.serialization

            try:
                safe_globals_exist = True
            except ImportError:
                safe_globals_exist = False

            # Tylko jeśli safe_globals jest dostępne, dodaj dodatkowe globale
            if safe_globals_exist:
                try:
                    from torch import torch_version

                    safe_types = [
                        torch_version.TorchVersion,
                        np.ndarray,
                        np.dtype,
                    ]

                    for safe_type in safe_types:
                        torch.serialization.add_safe_globals([safe_type])
                except Exception as e:
                    print(f"Ostrzeżenie: Nie można dodać bezpiecznych globali: {e}")
        except Exception as e:
            print(f"Ostrzeżenie: Problem z konfiguracją bezpiecznego ładowania: {e}")

        # Stopniowo zwiększaj poziom bezpieczeństwa przy ładowaniu
        load_exceptions = []

        # Próba 1: Użyj weights_only=True (zalecane dla PyTorch 2.6+)
        try:
            state = torch.load(path, weights_only=True, map_location="cpu")
            print("Załadowano model z weights_only=True")
        except Exception as e:
            load_exceptions.append(f"Próba z weights_only=True nie powiodła się: {e}")

            # Próba 2: Użyj pickle_module=None
            try:
                state = torch.load(path, map_location="cpu", pickle_module=None)
                print("Załadowano model z pickle_module=None")
            except Exception as e:
                load_exceptions.append(
                    f"Próba z pickle_module=None nie powiodła się: {e}"
                )

                # Próba 3: Standardowe ładowanie (mniej bezpieczne)
                try:
                    state = torch.load(path, map_location="cpu")
                    print("Załadowano model standardową metodą")
                except Exception as e:
                    error_msg = (
                        "Wszystkie metody ładowania nie powiodły się:\n"
                        + "\n".join(load_exceptions)
                        + f"\nOstatni błąd: {e}"
                    )
                    raise RuntimeError(error_msg) from e

        # Tworzenie modelu
        try:
            # Sprawdź, czy w stanie modelu są zapisane wagi
            weights = None
            if "normalized_weights" in state:
                weights = state["normalized_weights"]
            elif (
                isinstance(state, dict)
                and "model_state_dict" in state
                and "weights" in state
            ):
                weights = state["weights"]

            # Parametry temperatury i regularyzacji
            temperature = (
                state.get("temperature", 1.0) if isinstance(state, dict) else 1.0
            )
            reg_strength = (
                state.get("regularization_strength", 0.01)
                if isinstance(state, dict)
                else 0.01
            )

            # Tworzenie modelu ensemble
            model = cls(
                models_dict=models_dict,
                weights=weights,
                temperature=temperature,
                regularization_strength=reg_strength,
            )

            # Wczytaj stan modelu
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                try:
                    model.load_state_dict(state)
                except Exception as e:
                    print(
                        f"Ostrzeżenie: Nie udało się bezpośrednio załadować stanu modelu: {e}"
                    )

            # Pobierz nazwy klas, jeśli są dostępne
            class_names = None
            if isinstance(state, dict) and "class_names" in state:
                class_names = state["class_names"]

            model.eval()  # Przełącz model w tryb ewaluacji
            return model, class_names

        except Exception as e:
            raise RuntimeError(
                f"Błąd podczas inicjalizacji modelu z wczytanych danych: {e}"
            ) from e
