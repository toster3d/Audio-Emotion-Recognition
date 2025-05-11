import torch
import torch.serialization
import numpy as np
import sys
import os

def inspect_model_state(path):
    """
    Wyświetla zawartość pliku stanu modelu z obsługą bezpieczeństwa PyTorch 2.6+
    
    Args:
        path (str): Ścieżka do pliku .pt z zapisanym modelem
    """
    print(f"Inspecting model file: {path}")
    print(f"PyTorch version: {torch.__version__}")
    print("-" * 60)
    
    # Dodanie TorchVersion do bezpiecznych globali
    try:
        # Dodajemy różne potencjalnie wymagane klasy/funkcje do bezpiecznych globali
        safe_globals = [
            torch.torch_version.TorchVersion,
            np.core.multiarray._reconstruct,
            np.ndarray,
            np.dtype,
            np._globals._NoValue
        ]
        
        # Dodaj wszystkie klasy/funkcje zabezpieczając przed brakującymi
        for safe_global in safe_globals:
            try:
                torch.serialization.add_safe_globals([safe_global])
                print(f"Added {safe_global.__module__}.{safe_global.__name__} to safe globals")
            except (AttributeError, ImportError, NameError) as e:
                print(f"Could not add to safe globals: {e}")
    except Exception as e:
        print(f"Warning: Could not add some classes to safe globals: {e}")
    
    state = None
    
    # Strategia 1: Używamy kontekstu safe_globals
    try:
        print("\nStrategy 1: Using safe_globals context manager...")
        with torch.serialization.safe_globals(safe_globals):
            state = torch.load(path, map_location="cpu")
        print("✓ Success with safe_globals context manager!")
    except Exception as e:
        print(f"✗ Failed with safe_globals context: {str(e)}")
    
    # Strategia 2: Próba z weights_only=True (bezpieczniejsza)
    if state is None:
        try:
            print("\nStrategy 2: Using weights_only=True...")
            state = torch.load(path, map_location="cpu", weights_only=True)
            print("✓ Success with weights_only=True!")
        except Exception as e:
            print(f"✗ Failed with weights_only=True: {str(e)}")
    
    # Strategia 3: Próba z weights_only=False (mniej bezpieczna)
    if state is None:
        try:
            print("\nStrategy 3: Using weights_only=False (less secure)...")
            state = torch.load(path, map_location="cpu", weights_only=False)
            print("✓ Success with weights_only=False!")
        except Exception as e:
            print(f"✗ Failed with weights_only=False: {str(e)}")
    
    # Jeśli wszystkie podejścia zawiodły
    if state is None:
        print("\n❌ All loading strategies failed")
        return
    
    # Wyświetlenie zawartości
    print("\nKeys available in model state:")
    for key in state.keys():
        print(f"- {key}")
        
    print("\nDetailed information:")
    for key, value in state.items():
        if isinstance(value, (str, int, float, bool)):
            print(f"{key}: {value}")
        elif isinstance(value, (list, tuple)) and len(value) < 10:
            print(f"{key}: {value}")
        elif isinstance(value, dict) and len(value) < 10:
            print(f"{key}: {value}")
        elif isinstance(value, dict):
            print(f"{key}: {type(value)} with {len(value)} elements")
            # Show a few examples
            print("  Examples:")
            for i, (k, v) in enumerate(value.items()):
                if i >= 3: break
                print(f"  - {k}: {type(v)}")
        else:
            print(f"{key}: {type(value)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Poprawiona ścieżka do ostatnio zapisanego modelu
        model_path = os.path.join("src", "ensemble_outputs", "ensemble_run_20250510_201804", "models", "ensemble_model.pt")
    
    inspect_model_state(model_path) 