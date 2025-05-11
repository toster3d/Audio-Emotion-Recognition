import torch
import torch.serialization
import numpy as np
import os

# Dodajemy bezpieczne globale (wymagane dla PyTorch 2.6+)
def add_safe_globals_for_pytorch():
    """
    Dodaje niezbędne klasy/funkcje do bezpiecznej listy PyTorch 2.6+
    """
    try:
        # Lista potencjalnie wymaganych klas/funkcji 
        safe_globals = [
            torch.torch_version.TorchVersion,
            np.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype
        ]
        
        # Dodanie wszystkich klas/funkcji do bezpiecznej listy
        for safe_global in safe_globals:
            try:
                torch.serialization.add_safe_globals([safe_global])
                print(f"Dodano {safe_global.__module__}.{safe_global.__name__} do bezpiecznych globali")
            except Exception as e:
                print(f"Nie można dodać do bezpiecznych globali: {e}")
                
    except Exception as e:
        print(f"Ostrzeżenie: Nie można dodać niektórych klas do bezpiecznych globali: {e}")

def load_model_safe(path, map_location="cpu"):
    """
    Bezpieczne ładowanie modelu PyTorch z obsługą różnych wersji
    
    Args:
        path (str): Ścieżka do pliku modelu
        map_location: Urządzenie, na które załadować model
        
    Returns:
        dict: Słownik stanu modelu lub None w przypadku niepowodzenia
    """
    print(f"Ładowanie modelu: {path}")
    print(f"Wersja PyTorch: {torch.__version__}")
    
    # Dodanie bezpiecznych globali
    add_safe_globals_for_pytorch()
    
    # Próba różnych metod ładowania
    try:
        # 1. Najpierw spróbuj z weights_only=False (mniej bezpieczne, ale działa)
        print("Próba ładowania z weights_only=False...")
        state = torch.load(path, map_location=map_location, weights_only=False)
        print("✓ Sukces z weights_only=False")
        return state
    except Exception as e1:
        print(f"✗ Nie udało się z weights_only=False: {str(e1)}")
        
        try:
            # 2. Spróbuj z weights_only=True po dodaniu bezpiecznych globali
            print("Próba ładowania z weights_only=True...")
            state = torch.load(path, map_location=map_location, weights_only=True)
            print("✓ Sukces z weights_only=True")
            return state
        except Exception as e2:
            print(f"✗ Nie udało się z weights_only=True: {str(e2)}")
            
            # 3. Ostatnia próba - wczytaj metadane JSON, gdy dostępne
            metadata_path = path.replace('.pt', '_metadata.json')
            if os.path.exists(metadata_path):
                import json
                print("Próba wczytania metadanych z JSON...")
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    print("✓ Sukces z wczytaniem metadanych JSON")
                    return metadata
                except Exception as e3:
                    print(f"✗ Nie udało się wczytać metadanych: {str(e3)}")
    
    print("❌ Wszystkie metody ładowania zawiodły!")
    return None

def inspect_model_keys(model_state):
    """
    Wypisuje dostępne klucze i ich wartości w stanie modelu
    
    Args:
        model_state (dict): Słownik stanu modelu
    """
    if model_state is None:
        print("Brak danych modelu do inspekcji")
        return
    
    print("\nDostępne klucze w stanie modelu:")
    for key in model_state.keys():
        print(f"- {key}")
        
    print("\nSzczegółowe informacje:")
    for key, value in model_state.items():
        if isinstance(value, (str, int, float, bool)):
            print(f"{key}: {value}")
        elif isinstance(value, (list, tuple)) and len(value) < 10:
            print(f"{key}: {value}")
        elif isinstance(value, dict) and len(value) < 10:
            print(f"{key}: {value}")
        elif isinstance(value, dict):
            print(f"{key}: {type(value)} z {len(value)} elementami")
            # Wyświetl kilka przykładów
            print("  Przykłady:")
            for i, (k, v) in enumerate(value.items()):
                if i >= 3: break
                print(f"  - {k}: {type(v)}")
        else:
            print(f"{key}: {type(value)}")

# Przykład użycia
if __name__ == "__main__":
    # Ścieżka do ostatnio zapisanego modelu
    model_path = os.path.join("src", "ensemble_outputs", "ensemble_run_20250510_201804", "models", "ensemble_model.pt")
    
    # Wczytaj model
    model_state = load_model_safe(model_path)
    
    # Wyświetl informacje
    inspect_model_keys(model_state) 