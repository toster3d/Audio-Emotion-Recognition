import os
import re

import pandas as pd

from .utils import find_results_directory


def read_results_from_files(base_dir=None):
    """
    Odczytuje wyniki dokładności i czas treningu z plików txt i csv w folderach cech.

    Args:
        base_dir: Katalog bazowy z folderami cech

    Returns:
        DataFrame z wynikami
    """
    # Ustalenie katalogu z wynikami
    if base_dir is None:
        base_dir = find_results_directory()

    if base_dir is None:
        return None

    results = []

    # Iteracja przez foldery cech
    for feature_dir in os.listdir(base_dir):
        feature_path = os.path.join(base_dir, feature_dir)

        # Pomijanie plików (skupienie na folderach cech)
        if not os.path.isdir(feature_path):
            continue

        # Inicjalizacja zmiennych dla cechy
        accuracy = None
        training_time = None
        feature_type = feature_dir

        # Wyszukiwanie plików wyników txt
        txt_files = [
            f
            for f in os.listdir(feature_path)
            if f.startswith("results_") and f.endswith(".txt")
        ]
        for txt_file in txt_files:
            try:
                with open(os.path.join(feature_path, txt_file)) as f:
                    content = f.read()

                    # Wydobywanie dokładności
                    accuracy_match = re.search(r"test_accuracy:\s*([\d.]+)", content)
                    if accuracy_match:
                        accuracy = float(accuracy_match.group(1))

                    # Wydobywanie czasu treningu
                    time_match = re.search(r"training_time:\s*([\d.]+)", content)
                    if time_match:
                        training_time = float(time_match.group(1))
            except Exception as e:
                print(
                    f"Błąd podczas odczytu pliku {os.path.join(feature_path, txt_file)}: {e!s}"
                )

        # W przypadku braku wyników w txt, przeszukiwanie plików CSV
        if accuracy is None:
            csv_files = [
                f
                for f in os.listdir(feature_path)
                if f.startswith("classification_report_") and f.endswith(".csv")
            ]
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(os.path.join(feature_path, csv_file))
                    # Wydobywanie wiersza z dokładnością
                    if "accuracy" in df.iloc[:, 0].values:
                        accuracy_row = df[df.iloc[:, 0] == "accuracy"]
                        if not accuracy_row.empty:
                            accuracy = (
                                float(accuracy_row.iloc[0, 1]) * 100
                            )  # Konwersja na procenty
                except Exception as e:
                    print(
                        f"Błąd podczas odczytu pliku {os.path.join(feature_path, csv_file)}: {e!s}"
                    )

        # Dodawanie wyników, jeśli dokładność lub czas zostały znalezione
        if accuracy is not None or training_time is not None:
            results.append(
                {
                    "Feature Type": feature_type,
                    "Test Accuracy (%)": accuracy if accuracy is not None else 0,
                    "Training Time (s)": training_time
                    if training_time is not None
                    else 0,
                }
            )

    # Tworzenie DataFrame z wyników
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
    # Ustalenie katalogu z wynikami
    if base_dir is None:
        base_dir = find_results_directory()

    if base_dir is None:
        return None

    all_emotion_results = []

    # Iteracja przez foldery cech
    for feature_dir in os.listdir(base_dir):
        feature_path = os.path.join(base_dir, feature_dir)

        # Pomijanie plików (skupienie na folderach cech)
        if not os.path.isdir(feature_path):
            continue

        # Wyszukiwanie plików classification_report
        csv_files = [
            f
            for f in os.listdir(feature_path)
            if f.startswith("classification_report_") and f.endswith(".csv")
        ]

        for csv_file in csv_files:
            try:
                df = pd.read_csv(os.path.join(feature_path, csv_file))

                # Selekcja wierszy z emocjami (bez accuracy, macro avg, weighted avg)
                emotions_df = df[
                    ~df.iloc[:, 0].isin(["accuracy", "macro avg", "weighted avg"])
                ]

                # Dodawanie wyników dla każdej emocji
                for _, row in emotions_df.iterrows():
                    emotion = row.iloc[0]
                    precision = float(row.iloc[1])
                    recall = float(row.iloc[2])
                    f1 = float(row.iloc[3])

                    all_emotion_results.append(
                        {
                            "Feature Type": feature_dir,
                            "Emotion": emotion,
                            "Precision": precision * 100,  # Konwersja na procenty
                            "Recall": recall * 100,
                            "F1-score": f1 * 100,
                        }
                    )

            except Exception as e:
                print(
                    f"Błąd podczas odczytu pliku {os.path.join(feature_path, csv_file)}: {e!s}"
                )

    # Tworzenie DataFrame z wyników
    if all_emotion_results:
        emotions_df = pd.DataFrame(all_emotion_results)
        return emotions_df
    else:
        print("Nie znaleziono wyników dla emocji w podanych folderach.")
        return None
