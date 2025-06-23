import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import VGG16_EMOTION_NAMES as EMOTION_NAMES


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_dir: Optional[str] = None,
    normalize: bool = True,
) -> Tuple[float, float, np.ndarray]:
    """
    Ewaluacja modelu na zbiorze testowym

    Args:
        model: Model PyTorch
        test_loader: DataLoader dla zbioru testowego
        device: Urządzenie (CPU/GPU)
        save_dir: Opcjonalna ścieżka do zapisu macierzy pomyłek
        normalize: NOrmalizacja macierzy pomyłek

    Returns:
        test_acc: Dokładność na zbiorze testowym
        test_loss: Strata na zbiorze testowym
        conf_matrix: Macierz pomyłek
    """
    # Ustaw model w trybie ewaluacji
    model.eval()

    # Inicjalizacja zmiennych
    test_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    # Kryterium straty
    criterion = nn.CrossEntropyLoss()

    # Ewaluacja bez obliczania gradientów
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Ewaluacja"):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Aktualizacja statystyk
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Zapisanie predykcji i celów do macierzy pomyłek
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Obliczenie końcowych metryk
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = correct / total

    # Obliczenie macierzy pomyłek
    conf_matrix = confusion_matrix(all_targets, all_preds)

    # Wizualizacja macierzy pomyłek
    plt.style.use("dark_background")
    plt.figure(figsize=(10, 8))

    # Wizualizacja standardowej lub znormalizowanej macierzy
    if normalize:
        # Normalizacja macierzy pomyłek (po wierszach)
        row_sums = conf_matrix.sum(axis=1)
        norm_conf_matrix = conf_matrix / row_sums[:, np.newaxis]

        sns.heatmap(
            norm_conf_matrix,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=EMOTION_NAMES,
            yticklabels=EMOTION_NAMES,
        )
        plt.title(f"Znormalizowana macierz pomyłek - dokładność: {test_acc:.4f}")

        # Zapisz znormalizowaną macierz jeśli podano ścieżkę
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(
                os.path.join(save_dir, "normalized_confusion_matrix.png"), dpi=300
            )
    else:
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=EMOTION_NAMES,
            yticklabels=EMOTION_NAMES,
        )
        plt.title(f"Macierz pomyłek - dokładność: {test_acc:.4f}")

        # Zapisz standardową macierz jeśli podano ścieżkę
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)

    plt.xlabel("Predykcja")
    plt.ylabel("Prawdziwa wartość")
    plt.tight_layout()
    plt.show()

    # Wyświetlenie wyników
    print(f"Strata testowa: {test_loss:.4f}")
    print(f"Dokładność testowa: {test_acc:.4f}")

    # Zapisanie wyników do pliku tekstowego
    if save_dir:
        with open(os.path.join(save_dir, "test_results.txt"), "w") as f:
            f.write(f"DOkładność testowa: {test_acc:.4f}\n")
            f.write(f"Strata testowa: {test_loss:.4f}\n")

    return test_acc, test_loss, conf_matrix


def plot_training_history(history, save_dir=None):
    """
    Wizualizacja historii trenowania

    Args:
        history: Historia trenowania (słownik)
        save_dir: Opcjonalna ścieżka do zapisu wykresu
    """
    plt.style.use("dark_background")
    plt.figure(figsize=(12, 5))

    # Wykres dokładności
    plt.subplot(1, 2, 1)
    plt.plot(history["train_acc"], label="Dokładność treningu")
    plt.plot(history["val_acc"], label="Dokładność walidacji")
    plt.title("Dokładność modelu")
    plt.xlabel("Epoka")
    plt.ylabel("Dokładność")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Wykres straty
    plt.subplot(1, 2, 2)
    plt.plot(history["train_loss"], label="Strata treningu")
    plt.plot(history["val_loss"], label="Strata walidacji")
    plt.title("Funkcja straty modelu")
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Zapisz wykres jeśli podano ścieżkę
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, "training_history.png"), dpi=300)

        # Zapisz też plik tekstowy z kluczowymi metrykami
        with open(os.path.join(save_dir, "training_metrics.txt"), "w") as f:
            f.write(f"Liczba epok: {len(history['train_acc'])}\n")
            f.write(f"Najlepsza dokładność walidacji: {max(history['val_acc']):.4f}\n")
            f.write(f"Końcowa dokładność walidacji: {history['val_acc'][-1]:.4f}\n")
            f.write(f"Najlepsza dokładność treningu: {max(history['train_acc']):.4f}\n")

    plt.show()
