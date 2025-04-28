from .utils import find_results_directory
import os
import pandas as pd
import re

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

def read_emotion_results(base_dir=None):
    """
    Odczytuje wyniki dla poszczególnych emocji z plików classification_report.
    
    Args:
        base_dir: Katalog bazowy z folderami cech
        
    Returns:
        DataFrame z wynikami dla wszystkich emocji i typów cech
    """
    # Znajdź katalog z wynikami
    if base_dir is None:
        base_dir = find_results_directory()
    
    if base_dir is None:
        return None
    
    all_emotion_results = []
    
    # Przeglądaj foldery cech
    for feature_dir in os.listdir(base_dir):
        feature_path = os.path.join(base_dir, feature_dir)
        
        # Pomijaj pliki (szukamy tylko folderów cech)
        if not os.path.isdir(feature_path):
            continue
        
        # Szukaj plików classification_report
        csv_files = [f for f in os.listdir(feature_path) if f.startswith('classification_report_') and f.endswith('.csv')]
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(os.path.join(feature_path, csv_file))
                
                # Wybierz tylko wiersze z emocjami (bez accuracy, macro avg, weighted avg)
                emotions_df = df[~df.iloc[:, 0].isin(['accuracy', 'macro avg', 'weighted avg'])]
                
                # Dla każdej emocji dodaj wyniki
                for _, row in emotions_df.iterrows():
                    emotion = row.iloc[0]
                    precision = float(row.iloc[1])
                    recall = float(row.iloc[2])
                    f1 = float(row.iloc[3])
                    
                    all_emotion_results.append({
                        'Feature Type': feature_dir,
                        'Emotion': emotion,
                        'Precision': precision * 100,  # Konwersja na procenty
                        'Recall': recall * 100,
                        'F1-score': f1 * 100
                    })
                    
            except Exception as e:
                print(f"Błąd podczas odczytu pliku {os.path.join(feature_path, csv_file)}: {str(e)}")
    
    # Utwórz DataFrame z wyników
    if all_emotion_results:
        emotions_df = pd.DataFrame(all_emotion_results)
        return emotions_df
    else:
        print("Nie znaleziono wyników dla emocji w podanych folderach.")
        return None