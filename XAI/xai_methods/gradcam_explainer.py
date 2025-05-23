import numpy as np
import torch

class GradCAM:
    def __init__(self, model, target_layer_name="layer4"):
        """
        Inicjalizuje GradCAM dla modelu AudioResNet.
        
        Args:
            model: Model AudioResNet
            target_layer_name: Nazwa warstwy, z której będziemy pobierać mapy aktywacji
                              (zwykle ostatnia warstwa konwolucyjna)
        """
        self.model = model
        self.model.eval()
        
        # Warstwy są zagnieżdżone w komponencie 'resnet', więc odpowiednio dostosowujemy ścieżki
        if target_layer_name == "layer1":
            self.target_layer = model.resnet.layer1
        elif target_layer_name == "layer2":
            self.target_layer = model.resnet.layer2
        elif target_layer_name == "layer3":
            self.target_layer = model.resnet.layer3
        elif target_layer_name == "layer4":
            self.target_layer = model.resnet.layer4
        else:
            raise ValueError(f"Nieznana nazwa warstwy: {target_layer_name}")
        
        # Zmienne do przechowywania gradientów i aktywacji
        self.gradients = None
        self.activations = None
        
        # Zarejestruj hooki dla forward i backward
        self.register_hooks()
        
    def register_hooks(self):
        """Rejestruje hooki do przechwytywania aktywacji i gradientów."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
            
        # Zarejestruj hooki
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
    
    def remove_hooks(self):
        """Usuwa zarejestrowane hooki."""
        self.forward_handle.remove()
        self.backward_handle.remove()
        
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generuje mapę GradCAM dla podanego wejścia.
        
        Args:
            input_tensor: Tensor wejściowy (1, 1, H, W)
            target_class: Klasa docelowa (jeśli None, używa predykcji modelu)
            
        Returns:
            Mapa cieplna GradCAM przeskalowana do rozmiaru wejścia
        """
        # Upewnij się, że tensor ma gradient
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Jeśli nie podano klasy docelowej, użyj predykcji modelu
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        # Zerowanie gradientów
        self.model.zero_grad()
        
        # Wybierz wynik dla klasy docelowej
        target_output = output[0, target_class]
        
        # Backward pass
        target_output.backward()
        
        # Pobierz gradienty i aktywacje
        gradients = self.gradients
        activations = self.activations
        
        if gradients is None or activations is None:
            raise ValueError("Gradienty lub aktywacje są None - sprawdź czy hooki działają poprawnie")
        
        # Oblicz wagi dla kanałów
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Ważona suma aktywacji
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = torch.relu(cam)
        
        # Normalizacja
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Przeskalowanie do rozmiaru wejścia
        cam = torch.nn.functional.interpolate(
            cam, 
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear', 
            align_corners=False
        )
        
        return cam.squeeze().cpu().detach().numpy()

def verify_gradcam_explanation(model, sample, gradcam_map, device, reverse_label_mapping, percentile=80):
    """
    Weryfikuje wyjaśnienie GradCAM porównując spadek prawdopodobieństwa
    po zakryciu ważnych i losowych obszarów.
    
    Args:
        model: model, który chcemy wyjaśnić
        sample: próbka do wyjaśnienia
        gradcam_map: mapa istotności GradCAM
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
    
    # 2. Tworzenie maski binarnej z mapy GradCAM
    threshold = np.percentile(gradcam_map, percentile)
    binary_mask = gradcam_map > threshold
    
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
