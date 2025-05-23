import numpy as np
import torch

class LRP:
    def __init__(self, model):
        """
        Inicjalizuje Layer-wise Relevance Propagation dla modelu
        
        Args:
            model: Model PyTorch, który chcemy wyjaśnić
        """
        self.model = model
        self.model.eval()
        
    def generate_relevance_map(self, input_tensor, target_class=None):
        """
        Generuje mapę istotności LRP dla danego spektrogramu.
        
        Args:
            input_tensor: Tensor zawierający spektrogram (1, 1, H, W)
            target_class: Klasa dla której generujemy wyjaśnienie (jeśli None, używamy predykcji)
            
        Returns:
            Mapa istotności o tym samym rozmiarze co wejściowy spektrogram
        """
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Jeśli nie podano klasy docelowej, użyj klasy z najwyższym prawdopodobieństwem
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        # Inicjalizacja relevance dla ostatniej warstwy (one-hot dla wybranej klasy)
        relevance = torch.zeros_like(output)
        relevance[0, target_class] = 1.0
        
        # Propagacja relevance przez sieć (implementacja LRP-ε)
        # Używamy gradientów pomnożonych przez wejście jako przybliżenie
        self.model.zero_grad()
        output.backward(gradient=relevance)
        
        # Gradient * Input jako przybliżenie LRP
        relevance_map = input_tensor * input_tensor.grad
        
        return relevance_map.detach().cpu().numpy()[0, 0]  # (H, W)

def verify_lrp_explanation(model, sample, relevance_map, device, reverse_label_mapping, percentile=80):
    """
    Weryfikuje wyjaśnienie LRP porównując spadek prawdopodobieństwa
    po zakryciu ważnych i losowych obszarów.
    
    Args:
        model: model, który chcemy wyjaśnić
        sample: próbka do wyjaśnienia
        relevance_map: mapa istotności LRP
        device: urządzenie (CPU/GPU)
        reverse_label_mapping: odwrotne mapowanie etykiet
        percentile: percentyl, powyżej którego wartości uznajemy za ważne
        
    Returns:
        Dict zawierający wyniki weryfikacji
    """
    # Przygotowanie próbki
    if sample.shape[0] == 1:
        sample = sample.squeeze(0)
    
    # Konwersja do formatu wymaganego przez model
    sample_tensor = torch.tensor(sample).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # 1. Klasyfikacja oryginalnej próbki
    with torch.no_grad():
        original_output = model(sample_tensor)
        original_probs = torch.nn.functional.softmax(original_output, dim=1).cpu().numpy()[0]
        pred_class = np.argmax(original_probs)
        pred_prob = original_probs[pred_class]
        pred_emotion = reverse_label_mapping[pred_class]
    
    # 2. Tworzenie maski binarnej z mapy istotności LRP
    abs_relevance = np.abs(relevance_map)
    threshold = np.percentile(abs_relevance, percentile)
    binary_mask = abs_relevance > threshold
    
    # 3. Tworzenie zaciemnionej próbki (zakrywanie ważnych obszarów)
    masked_sample = sample.copy()
    mean_val = np.mean(sample)
    masked_sample[binary_mask] = mean_val
    
    # 4. Klasyfikacja zaciemnionej próbki
    masked_tensor = torch.tensor(masked_sample).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        masked_output = model(masked_tensor)
        masked_probs = torch.nn.functional.softmax(masked_output, dim=1).cpu().numpy()[0]
        masked_pred = np.argmax(masked_probs)
        masked_pred_prob = masked_probs[pred_class]
        masked_emotion = reverse_label_mapping[masked_pred]
    
    # 5. Tworzenie losowo zaciemnionej próbki
    random_mask = np.zeros_like(binary_mask)
    num_pixels = np.sum(binary_mask)
    random_indices = np.random.choice(binary_mask.size, size=int(num_pixels), replace=False)
    random_mask.flat[random_indices] = 1
    
    # Zastosowanie losowej maski
    random_masked_sample = sample.copy()
    random_masked_sample[random_mask] = mean_val
    
    # 6. Klasyfikacja losowo zaciemnionej próbki
    random_tensor = torch.tensor(random_masked_sample).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        random_output = model(random_tensor)
        random_probs = torch.nn.functional.softmax(random_output, dim=1).cpu().numpy()[0]
        random_pred = np.argmax(random_probs)
        random_pred_prob = random_probs[pred_class]
        random_emotion = reverse_label_mapping[random_pred]

    # Zwracamy wyniki weryfikacji
    return {
        'pred_class': pred_class,
        'pred_emotion': pred_emotion,
        'pred_prob': pred_prob,
        'masked_pred': masked_pred,
        'masked_emotion': masked_emotion,
        'masked_pred_prob': masked_pred_prob,
        'prob_drop_important': pred_prob - masked_pred_prob,
        'random_pred': random_pred,
        'random_emotion': random_emotion,
        'random_pred_prob': random_pred_prob,
        'prob_drop_random': pred_prob - random_pred_prob,
        'binary_mask': binary_mask,
        'random_mask': random_mask,
        'masked_sample': masked_sample,
        'random_masked_sample': random_masked_sample
    }
