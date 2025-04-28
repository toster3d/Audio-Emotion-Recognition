import os
import pandas as pd
import re
import pickle
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold

from config import DEVICE, CLASS_NAMES, SEED, CV_FOLDS

def load_pretrained_model(model_path, model_class, num_classes=6, device=None):
    """
    Załaduj wstępnie wytrenowany model.
    
    Argumenty:
        model_path (str): Ścieżka do zapisanego modelu
        model_class (class): Klasa modelu do załadowania
        num_classes (int, optional): Liczba klas w modelu. Domyślnie 6.
        device (torch.device, optional): Urządzenie do umieszczenia modelu.
            Jeśli None, używa DEVICE z konfiguracji.
    
    Zwraca:
        model_class: Załadowany model lub None w przypadku błędu
    """
    if device is None:
        device = DEVICE
        
    model = model_class(num_classes=num_classes)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Obsłuż modele opakowane w DataParallel
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model.eval()  # Ustaw tryb ewaluacji
        return model
    except Exception as e:
        print(f"Błąd podczas ładowania modelu z {model_path}: {e}")
        return None

def load_features(feature_file):
    """
    Załaduj metadane cech z pliku pickle bez ładowania wszystkich cech naraz.
    
    Argumenty:
        feature_file (str): Ścieżka do pliku cech w formacie pickle
    
    Zwraca:
        tuple: (shape_cech, próbka_cech, etykiety, koder_etykiet) lub (None, None, None, None) w przypadku błędu
    """
    try:
        with open(feature_file, 'rb') as f:
            data = pickle.load(f)
            
        # Sprawdź strukturę załadowanych danych
        if isinstance(data, dict) and 'features' in data and 'labels' in data:
            # Po prostu uzyskaj kształt i kilka pierwszych próbek, aby zweryfikować strukturę
            features_shape = data['features'].shape
            features_sample = data['features'][:5]  # Tylko pierwsze 5 próbek
            labels = data['labels']
            label_encoder = data.get('label_encoder', None)
            return features_shape, features_sample, labels, label_encoder
        else:
            print(f"Nieoczekiwana struktura danych w {feature_file}")
            return None, None, None, None
    except Exception as e:
        print(f"Błąd podczas ładowania cech z {feature_file}: {e}")
        return None, None, None, None

def stratified_kfold_split(labels, n_splits=None, random_state=None):
    """
    Utwórz złożone podziały krzyżowe z warstwowaniem, zwracając tylko indeksy.
    
    Argumenty:
        labels (array-like): Etykiety dla stratyfikacji
        n_splits (int, optional): Liczba foldów. Jeśli None, używa CV_FOLDS z konfiguracji.
        random_state (int, optional): Ziarno losowości. Jeśli None, używa RANDOM_SEED z konfiguracji.
    
    Zwraca:
        list: Lista tupli (indeksy_treningowe, indeksy_walidacyjne)
    """
    if n_splits is None:
        n_splits = CV_FOLDS
    if random_state is None:
        random_state = SEED
    
    # Utwórz StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Utwórz podziały
    folds = []
    for train_indices, val_indices in skf.split(np.zeros(len(labels)), labels):
        folds.append((train_indices, val_indices))
    
    return folds

def evaluate_model(model, dataloader, device=None, class_names=None, return_probs=False):
    """
    Oceń model zespołowy i zwróć szczegółowe metryki.
    
    Argumenty:
        model (torch.nn.Module): Model do oceny
        dataloader (torch.utils.data.DataLoader): DataLoader z danymi testowymi
        device (torch.device, optional): Urządzenie do umieszczenia modelu i danych.
            Jeśli None, używa DEVICE z konfiguracji.
        class_names (list, optional): Nazwy klas. Jeśli None, używa CLASS_NAMES z konfiguracji.
        return_probs (bool, optional): Czy zwrócić prawdopodobieństwa. Domyślnie False.
    
    Zwraca:
        dict: Słownik metryk i wyników
    """
    if device is None:
        device = DEVICE
    if class_names is None:
        class_names = CLASS_NAMES
        
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Przenieś dane wejściowe na urządzenie
            labels = labels.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()} if isinstance(inputs, dict) else inputs.to(device)
            
            # Przepływ do przodu
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Zbieraj prognozy, etykiety i prawdopodobieństwa
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    # Oblicz metryki
    results = calculate_metrics(all_labels, all_preds, all_probs, class_names, return_probs)
    return results

def calculate_metrics(true_labels, predictions, probabilities=None, class_names=None, return_probs=False):
    """
    Oblicz metryki dla wyników klasyfikacji.
    
    Argumenty:
        true_labels (array-like): Prawdziwe etykiety
        predictions (array-like): Przewidywane etykiety
        probabilities (array-like, optional): Prawdopodobieństwa klas
        class_names (list, optional): Nazwy klas. Jeśli None, używa CLASS_NAMES z konfiguracji.
        return_probs (bool, optional): Czy zwrócić prawdopodobieństwa. Domyślnie False.
    
    Zwraca:
        dict: Słownik metryk
    """
    if class_names is None:
        class_names = CLASS_NAMES
        
    # Oblicz podstawowe metryki
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    # Generuj raport klasyfikacji i macierz pomyłek
    report = classification_report(true_labels, predictions, target_names=class_names, output_dict=True)
    cm = confusion_matrix(true_labels, predictions)
    
    # Przygotuj wyniki
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report,
        'cm': cm,
        'preds': predictions,
        'true': true_labels
    }
    
    # Dodaj prawdopodobieństwa, jeśli wymagane
    if return_probs and probabilities is not None:
        results['probabilities'] = np.array(probabilities)
    
    return results

def find_results_directory(base_dir='feature_comparison_results'):
    """
    Znajduje katalog z wynikami.
    
    Args:
        base_dir: Proponowany katalog bazowy
        
    Returns:
        Ścieżka do katalogu z wynikami lub None
    """
    if os.path.exists(base_dir):
        return base_dir
    
    # Spróbuj alternatywne ścieżki
    alternatives = [
        'feature_comparison_results',
        'src/feature_comparison_results',
        '../feature_comparison_results',
        './feature_comparison_results'
    ]
    
    for alt_path in alternatives:
        if os.path.exists(alt_path):
            print(f"Znaleziono katalog: {alt_path}")
            return alt_path
    
    print(f"Nie można znaleźć katalogu wyników. Sprawdź ścieżkę: {base_dir}")
    return None

def read_results_from_files(base_dir=None):
    """
    Odczytuje wyniki dokładności i czas treningu z plików txt i csv w folderach cech.
    
    Args:
        base_dir: Katalog bazowy z folderami cech
        
    Returns:
        DataFrame z wynikami
    """
    # Znajdź katalog z wynikami
    if base_dir is None:
        base_dir = find_results_directory()
    
    if base_dir is None:
        return None
    
    results = []
    
    # Przeglądaj foldery cech
    for feature_dir in os.listdir(base_dir):
        feature_path = os.path.join(base_dir, feature_dir)
        
        # Pomijaj pliki (szukamy tylko folderów cech)
        if not os.path.isdir(feature_path):
            continue
            
        # Inicjalizuj zmienne dla cechy
        accuracy = None
        training_time = None
        feature_type = feature_dir
        
        # Szukaj plików wyników txt
        txt_files = [f for f in os.listdir(feature_path) if f.startswith('results_') and f.endswith('.txt')]
        for txt_file in txt_files:
            try:
                with open(os.path.join(feature_path, txt_file), 'r') as f:
                    content = f.read()
                    
                    # Szukaj dokładności
                    accuracy_match = re.search(r'test_accuracy:\s*([\d.]+)', content)
                    if accuracy_match:
                        accuracy = float(accuracy_match.group(1))
                    
                    # Szukaj czasu treningu
                    time_match = re.search(r'training_time:\s*([\d.]+)', content)
                    if time_match:
                        training_time = float(time_match.group(1))
            except Exception as e:
                print(f"Błąd podczas odczytu pliku {os.path.join(feature_path, txt_file)}: {str(e)}")
        
        # Jeśli nie znaleziono wyników w txt, spróbuj plik CSV
        if accuracy is None:
            csv_files = [f for f in os.listdir(feature_path) if f.startswith('classification_report_') and f.endswith('.csv')]
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(os.path.join(feature_path, csv_file))
                    # Szukaj wiersza z dokładnością
                    if 'accuracy' in df.iloc[:, 0].values:
                        accuracy_row = df[df.iloc[:, 0] == 'accuracy']
                        if not accuracy_row.empty:
                            accuracy = float(accuracy_row.iloc[0, 1]) * 100  # Konwersja na procenty
                except Exception as e:
                    print(f"Błąd podczas odczytu pliku {os.path.join(feature_path, csv_file)}: {str(e)}")
        
        # Jeśli znaleziono dokładność lub czas, dodaj do wyników
        if accuracy is not None or training_time is not None:
            results.append({
                'Feature Type': feature_type,
                'Test Accuracy (%)': accuracy if accuracy is not None else 0,
                'Training Time (s)': training_time if training_time is not None else 0
            })
    
    # Utwórz DataFrame z wyników
    if results:
        df = pd.DataFrame(results)
        return df
    else:
        print("Nie znaleziono wyników w podanych folderach.")
        return None