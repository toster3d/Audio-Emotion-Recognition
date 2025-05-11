# Komórka bezpiecznej inspekcji modeli PyTorch 2.6+
import torch
import torch.serialization
import numpy as np
import os

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

def safe_inspect_model(path):
    """
    Bezpieczna funkcja do inspekcji zawartości modelu PyTorch 2.6+
    
    Args:
        path (str): Ścieżka do pliku .pt z zapisanym modelem
    """
    print(f"Inspekcja pliku modelu: {path}")
    print(f"Wersja PyTorch: {torch.__version__}")
    
    # Dodanie bezpiecznych globali
    add_safe_globals_for_pytorch()
    
    # Próba ładowania - najpierw weights_only=False (działa, ale mniej bezpieczne)
    try:
        print("\nPróba ładowania z weights_only=False...")
        state = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
        print("✓ Sukces z weights_only=False!")
        
        print("\nKlucze dostępne w pliku stanu modelu:")
        for key in state.keys():
            print(f"- {key}")
            
        print("\nSzczegółowe informacje:")
        for key, value in state.items():
            if isinstance(value, (str, int, float, bool)):
                print(f"{key}: {value}")
            elif isinstance(value, (list, tuple)) and len(value) < 10:
                print(f"{key}: {value}")
            elif isinstance(value, dict) and len(value) < 10:
                print(f"{key}: {value}")
            elif isinstance(value, dict):
                print(f"{key}: {type(value)} z {len(value)} elementami")
                print("  Przykłady:")
                for i, (k, v) in enumerate(value.items()):
                    if i >= 3: break
                    print(f"  - {k}: {type(v)}")
            else:
                print(f"{key}: {type(value)}")
        
        return state
                
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku: {str(e)}\n")
        
        # Jeśli jednak się nie udało, sprawdź metadane JSON
        try:
            metadata_path = path.replace('.pt', '_metadata.json')
            print(f"Próba wczytania metadanych z: {metadata_path}")
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print("\nMetadane modelu:")
            for key, value in metadata.items():
                print(f"- {key}: {value}")
            
            return metadata
        except Exception as e_meta:
            print(f"Nie udało się wczytać metadanych: {str(e_meta)}")
            
        return None

# Przykładowe użycie:
model_path = "src/ensemble_outputs/ensemble_run_20250510_201804/models/ensemble_model.pt"
model_state = safe_inspect_model(model_path) 