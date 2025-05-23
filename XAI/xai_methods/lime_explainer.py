import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries

def explain_with_lime(image, predict_fn, num_samples=5000, num_features=30):
    """
    Funkcja LIME specjalnie dla spektrogramów audio.
    
    Args:
        image: spektrogram do wyjaśnienia
        predict_fn: funkcja predykcyjna, przewidująca klasy dla danych wejściowych
        num_samples: liczba perturbowanych próbek do wygenerowania
        num_features: liczba cech do uwzględnienia w wyjaśnieniu
        
    Returns:
        temp: obraz z nałożonym wyjaśnieniem
        mask: maska wyjaśnienia
        segments: segmenty użyte do wyjaśnienia
    """
    # LIME wymaga (H, W, 3)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    
    # Specjalna segmentacja dla spektrogramów - siatka segmentów
    h, w = image.shape[:2]
    segments = np.zeros((h, w), dtype=np.int32)
    
    # Tworzenie siatki 8x12 = 96 segmentów
    # Więcej segmentów w osi czasu (pozioma), mniej w osi częstotliwości (pionowa)
    segments_h = 8  # Oś częstotliwości (pionowa)
    segments_w = 12  # Oś czasu (pozioma)
    
    # Utwórz siatkę segmentów
    for i in range(segments_h):
        for j in range(segments_w):
            segments[i*h//segments_h:(i+1)*h//segments_h, 
                     j*w//segments_w:(j+1)*w//segments_w] = i * segments_w + j
    
    # Stały segmentator, który zawsze zwróci naszą siatkę
    segmentation_fn = lambda x: segments
    
    # Inicjalizacja eksploratora LIME ze stałym ziarnem
    explainer = lime_image.LimeImageExplainer(random_state=42)
    
    # Generowanie wyjaśnień
    explanation = explainer.explain_instance(
        image,
        classifier_fn=predict_fn,
        segmentation_fn=segmentation_fn,
        top_labels=1,
        num_samples=num_samples,  # Zwiększona liczba próbek
        batch_size=50  # Parametr przyspieszający obliczenia
    )
    
    # Pobierz etykietę z najwyższym prawdopodobieństwem
    top_label = explanation.top_labels[0]
    
    # Pobierz wartości istotności dla wszystkich segmentów
    importances = dict(explanation.local_exp[top_label])
    
    # Sortuj segmenty według ich wagi (absolutnej wartości)
    sorted_importances = sorted(
        importances.items(), 
        key=lambda x: abs(x[1]), 
        reverse=True
    )
    
    # Pobranie maski uwzględniającej zarówno pozytywne jak i negatywne cechy
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=False,  # Pokaż WSZYSTKIE istotne cechy (zarówno + jak i -)
        num_features=num_features,  # Więcej cech
        hide_rest=False,
        min_weight=0.0  # Bez progu istotności
    )
    
    # Jeśli maska jest pusta, spróbuj jeszcze raz z innymi parametrami
    if np.sum(mask) == 0:
        print("Maska pusta, zwiększam liczbę cech...")
        temp, mask = explanation.get_image_and_mask(
            top_label,
            positive_only=True,  # Tym razem tylko pozytywne
            num_features=segments_h * segments_w,  # Wszystkie segmenty
            hide_rest=False
        )
    
    return temp, mask, segments

def verify_lime_explanation(model, sample, mask, device, reverse_label_mapping):
    """
    Weryfikuje wyjaśnienie LIME porównując spadek prawdopodobieństwa
    po zakryciu ważnych i losowych obszarów.
    
    Args:
        model: model, który chcemy wyjaśnić
        sample: próbka do wyjaśnienia
        mask: maska wyjaśnienia LIME
        device: urządzenie (CPU/GPU)
        reverse_label_mapping: odwrotne mapowanie etykiet
        
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
    
    # 2. Przygotowanie maski binarnej z maski LIME
    if mask.ndim > 2:
        binary_mask = mask[:,:,0] > 0
    else:
        binary_mask = mask > 0
    
    # 3. Tworzenie zaciemnionej próbki (zakrywanie ważnych obszarów)
    masked_sample = sample.copy()
    mean_val = np.mean(sample)
    indices = np.where(binary_mask)
    masked_sample[indices] = mean_val
    
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
    if num_pixels > 0:
        random_indices = np.random.choice(binary_mask.size, size=int(num_pixels), replace=False)
        random_mask.flat[random_indices] = 1
        
        # Zastosowanie losowej maski
        random_masked_sample = sample.copy()
        random_masked_sample[random_mask] = mean_val
    else:
        random_masked_sample = sample.copy()
    
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
