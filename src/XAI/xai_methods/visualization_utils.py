import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap


def normalize_sample(sample):
    """
    Normalizuje spektrogram do przedziału [0, 1].

    Args:
        sample: Spektrogram do normalizacji

    Returns:
        Znormalizowany spektrogram
    """
    sample_min = np.min(sample)
    sample_max = np.max(sample)

    if sample_min == sample_max:
        return np.zeros_like(sample)

    return (sample - sample_min) / (sample_max - sample_min)


def predict_proba(model, image, device):
    """
    Przewiduje prawdopodobieństwa klas dla podanego obrazu.

    Args:
        model: Model PyTorch
        image: Obraz wejściowy (NumPy array)
        device: Urządzenie obliczeniowe (CPU/GPU)

    Returns:
        Prawdopodobieństwa klas i indeks przewidywanej klasy
    """
    if len(image.shape) == 2:  # Pojedynczy kanał
        image = image[np.newaxis, np.newaxis, :, :]
    elif len(image.shape) == 3:  # Z kanałami
        if image.shape[0] == 1:  # Batch size = 1
            pass  # Już w odpowiednim formacie [1, C, H, W]
        else:  # Kanały na końcu [H, W, C]
            image = np.transpose(image, (2, 0, 1))
            image = image[np.newaxis, :, :, :]

    image_tensor = torch.tensor(image).float().to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probas = F.softmax(output, dim=1)[0].cpu().numpy()
        pred_class = np.argmax(probas)

    return probas, pred_class


def spectro_cmap():
    """
    Zwraca ładną mapę kolorów dla spektrogramów audio.

    Returns:
        LinearSegmentedColormap: Mapa kolorów dla wizualizacji spektrogramów
    """
    # Tworzenie niestandardowej mapy kolorów dla spektrogramów
    return LinearSegmentedColormap.from_list(
        "spectro_cmap",
        [
            (0.0, "#000033"),  # Ciemny niebieski dla cichych (niskich) wartości
            (0.2, "#000099"),  # Niebieski
            (0.4, "#0099FF"),  # Jasny niebieski
            (0.6, "#00CC99"),  # Morski/zielony
            (0.75, "#FFCC00"),  # Żółty
            (1.0, "#FF3300"),  # Czerwony dla głośnych (wysokich) wartości
        ],
        N=256,
    )


def plot_sample_with_explanation(sample, explanation, method="gradcam", title=None):
    """
    Wyświetla próbkę z nałożonym wyjaśnieniem wybranej metody XAI.

    Args:
        sample: Próbka audio (spektrogram)
        explanation: Wyjaśnienie metody XAI
        method: Rodzaj metody XAI ('gradcam', 'lrp', 'lime', 'silence')
        title: Opcjonalny tytuł wykresu

    Returns:
        fig, ax: Obiekty Figure i Axes matplotlib
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalizacja próbki
    sample_norm = normalize_sample(sample)

    if method == "gradcam":
        # Wyświetlenie oryginalnego spektrogramu
        ax.imshow(sample_norm, cmap=spectro_cmap())
        # Nałożenie mapy ciepła GradCAM
        cam = ax.imshow(explanation, cmap="hot", alpha=0.6)
        plt.colorbar(cam, ax=ax, label="Istotność", fraction=0.046, pad=0.04)

    elif method == "lrp":
        # Wyświetlenie oryginalnego spektrogramu
        ax.imshow(sample_norm, cmap=spectro_cmap())
        # Ustalenie zakresu min/max dla mapy LRP
        vmax = np.max(np.abs(explanation))
        vmin = -vmax
        # Nałożenie mapy istotności LRP
        lrp_map = ax.imshow(
            explanation, cmap="seismic", alpha=0.6, vmin=vmin, vmax=vmax
        )
        plt.colorbar(lrp_map, ax=ax, label="Istotność", fraction=0.046, pad=0.04)

    elif method == "lime":
        # LIME zwraca tuple (lime_temp, lime_mask), więc musimy to obsłużyć
        if isinstance(explanation, tuple):
            lime_temp, lime_mask = explanation
            ax.imshow(lime_temp, cmap="viridis")
        else:
            # Jeśli to pojedyncza mapa, wyświetl ją bezpośrednio
            ax.imshow(explanation, cmap="viridis")

    elif method == "silence":
        # Wyświetlenie mapy ciszy
        silence_map = ax.imshow(explanation, cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(
            silence_map,
            ax=ax,
            label="Prawdopodobieństwo ciszy",
            fraction=0.046,
            pad=0.04,
        )

    if title:
        ax.set_title(title, color="white", fontsize=14)

    ax.set_axis_off()
    ax.set_facecolor("black")

    return fig, ax


def display_xai_comparison(
    sample, xai_results, emotion, pred_emotion=None, figsize=(25, 6)
):
    """
    Wyświetla porównanie różnych metod XAI dla tej samej próbki audio.

    Args:
        sample: próbka audio (spektrogram)
        xai_results: słownik z wynikami różnych metod XAI
        emotion: prawdziwa emocja
        pred_emotion: przewidywana emocja (opcjonalnie)
        figsize: rozmiar wykresu

    Returns:
        Figure matplotlib
    """
    n_methods = len(xai_results) + 1  # +1 dla oryginalnego spektrogramu
    fig, axs = plt.subplots(1, n_methods, figsize=figsize)

    # Upewnij się, że axs jest listą/tablicą nawet jeśli n_methods=1
    if n_methods == 1:
        axs = [axs]

    # Normalizacja próbki do wyświetlenia
    sample_disp = normalize_sample(sample)

    # 1. Oryginalny spektrogram
    axs[0].set_title("Spektrogram Mela", color="white", fontsize=14)
    axs[0].imshow(sample_disp, cmap="viridis")
    axs[0].set_axis_off()
    axs[0].set_facecolor("black")

    # Licznik indeksu dla innych metod
    idx = 1

    # 2. LIME (jeśli dostępne)
    if "lime" in xai_results:
        lime_temp, lime_mask = xai_results["lime"]
        axs[idx].set_title("LIME", color="white", fontsize=14)
        axs[idx].imshow(lime_temp, cmap="viridis")
        axs[idx].set_axis_off()
        axs[idx].set_facecolor("black")
        idx += 1

    # 3. GradCAM (jeśli dostępne)
    if "gradcam" in xai_results:
        gradcam_map = xai_results["gradcam"]
        axs[idx].set_title("GradCAM", color="white", fontsize=14)
        # Najpierw wyświetl oryginalny spektrogram
        axs[idx].imshow(sample_disp, cmap=spectro_cmap())
        # Nałożenie mapy ciepła GradCAM
        axs[idx].imshow(gradcam_map, cmap="jet", alpha=0.7)
        axs[idx].set_axis_off()
        axs[idx].set_facecolor("black")
        idx += 1

    # 4. LRP (jeśli dostępne)
    if "lrp" in xai_results:
        lrp_map = xai_results["lrp"]
        axs[idx].set_title("LRP", color="white", fontsize=14)
        axs[idx].imshow(sample_disp, cmap=spectro_cmap())
        # Nałożenie mapy istotności LRP
        axs[idx].imshow(
            lrp_map,
            cmap="seismic",
            alpha=0.7,
            vmin=-np.max(np.abs(lrp_map)),
            vmax=np.max(np.abs(lrp_map)),
        )
        axs[idx].set_axis_off()
        axs[idx].set_facecolor("black")
        idx += 1

    # 5. Silence Maps (jeśli dostępne)
    if "silence" in xai_results:
        silence_map = xai_results["silence"]
        axs[idx].set_title("Smooth Saliency Maps", color="white", fontsize=14)
        # Wyświetl mapę ciszy
        img = axs[idx].imshow(silence_map, cmap="Blues", vmin=0, vmax=1)
        axs[idx].set_axis_off()
        axs[idx].set_facecolor("black")

        # Dodanie paska kolorów dla Silence Maps
        cbar = plt.colorbar(img, ax=axs[idx], fraction=0.046, pad=0.04)
        cbar.set_label("Cisza (prawdopodobieństwo)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax, "yticklabels"), color="white")

    plt.suptitle(f"Porównanie metod XAI: {emotion}", color="white", fontsize=16)
    plt.tight_layout()

    return fig


def plot_verification_results(results, title="Weryfikacja metody XAI"):
    """
    Wizualizuje wyniki weryfikacji metody XAI.

    Args:
        results: Lista słowników z wynikami dla różnych emocji
        title: Tytuł wykresu

    Returns:
        Figure matplotlib
    """
    emotions = [r["emotion"] for r in results]
    important_drops = [
        r.get("prob_drop_important", r.get("prob_drop_silence", 0)) for r in results
    ]
    random_drops = [r["prob_drop_random"] for r in results]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(emotions))
    width = 0.35

    ax.bar(x - width / 2, important_drops, width, label="Ważne obszary")
    ax.bar(x + width / 2, random_drops, width, label="Losowe obszary")

    ax.set_ylabel("Spadek prawdopodobieństwa", color="white")
    ax.set_title(title, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, color="white")
    ax.legend(loc="upper left")
    ax.set_facecolor("black")

    plt.tight_layout()
    return fig
