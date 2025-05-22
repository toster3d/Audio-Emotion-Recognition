import numpy as np
import torch

def create_smooth_silence_map(sample, n_samples=20, noise_level=0.05, percentile=20):
    """
    Tworzy wygładzoną mapę ciszy dla spektrogramu mela przy użyciu metody SmoothGrad.
    
    Args:
        sample: Spektrogram mela (numpy array)
        n_samples: Liczba próbek szumu do wygenerowania
        noise_level: Poziom szumu (odchylenie standardowe)
        percentile: Percentyl, poniżej którego wartości są uznawane za ciche
        
    Returns:
        Wygładzona mapa ciszy
    """
    # Normalizacja próbki
    sample_norm = (sample - sample.min()) / (sample.max() - sample.min() + 1e-10)
    
    # Inicjalizacja mapy ciszy
    silence_maps = []
    
    # Generowanie map ciszy dla zaszumionych próbek
    for _ in range(n_samples):
        # Dodaj szum do znormalizowanej próbki
        noise = np.random.normal(0, noise_level, sample_norm.shape)
        noisy_sample = np.clip(sample_norm + noise, 0, 1)  # Ograniczenie do zakresu [0,1]
        
        # Tworzenie maski dla obszarów cichych
        threshold = np.percentile(noisy_sample, percentile)
        silence_mask = (noisy_sample <= threshold).astype(float)
        silence_maps.append(silence_mask)
    
    # Uśrednianie map ciszy
    smooth_silence_map = np.mean(silence_maps, axis=0)
    
    return smooth_silence_map

def verify_silence_map(model, sample, silence_mask, device, reverse_label_mapping, threshold=0.5):
    """
    Weryfikuje mapę ciszy porównując spadek prawdopodobieństwa
    po zakryciu cichych i losowych obszarów.
    
    Args:
        model: model, który chcemy wyjaśnić
        sample: próbka do wyjaśnienia
        silence_mask: mapa ciszy
        device: urządzenie (CPU/GPU)
        reverse_label_mapping: odwrotne mapowanie etykiet
        threshold: próg, powyżej którego uznajemy obszar za cichy
        
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
    
    # 2. Tworzenie maski binarnej z mapy ciszy
    binary_mask = silence_mask > threshold
    
    # 3. Tworzenie zaciemnionej próbki (zakrywanie cichych obszarów)
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
        'prob_drop_silence': pred_prob - masked_pred_prob,
        'random_pred': random_pred,
        'random_emotion': random_emotion,
        'random_pred_prob': random_pred_prob,
        'prob_drop_random': pred_prob - random_pred_prob,
        'binary_mask': binary_mask,
        'random_mask': random_mask,
        'masked_sample': masked_sample,
        'random_masked_sample': random_masked_sample
    }