import os
import pickle
import yaml
from datetime import datetime

import mlflow
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from config import (
    DEVICE, CLASS_NAMES, BATCH_SIZE, SEED, 
    CV_FOLDS, TEST_SPLIT, OPTUNA_TRIALS, OPTUNA_TIMEOUT
)
from helpers.utils import load_pretrained_model, evaluate_model, stratified_kfold_split
from helpers.ensemble_model import WeightedEnsembleModel
from helpers.ensemble_dataset import EnsembleDatasetIndexed, ensemble_collate_fn

class EnsembleModelTrainer:
    """
    Klasa odpowiedzialna za trenowanie i optymalizację modeli ensemble.
    
    Argumenty:
        model_paths (dict): Słownik ścieżek do modeli w formacie {typ_cechy: ścieżka}
        feature_files (dict): Słownik ścieżek do plików cech w formacie {typ_cechy: ścieżka}
        output_dir (str): Katalog wyjściowy do zapisywania artefaktów
    """
    
    def __init__(self, model_paths, feature_files, output_dir, model_class, class_names=None, device=None):
        self.model_paths = model_paths
        self.feature_files = feature_files
        self.output_dir = output_dir
        self.model_class = model_class

        # Parametry konfiguracyjne
        self.device = device if device is not None else DEVICE
        self.class_names = class_names if class_names is not None else CLASS_NAMES
        self.feature_types = list(model_paths.keys())
        
        # Stan wewnętrzny
        self.base_models = {}
        self.dataset = None
        
        # Ładowanie modeli bazowych
        self._load_base_models()
        
        # Tworzenie katalogów wyjściowych
        self._create_output_directories()
        
    def _create_output_directories(self):
        """Tworzy wymagane katalogi wyjściowe"""
        dirs = [
            os.path.join(self.output_dir, "models"),
            os.path.join(self.output_dir, "evaluation"),
            os.path.join(self.output_dir, "optimization_plots"),
            os.path.join(self.output_dir, "error_analysis"),
            os.path.join(self.output_dir, "logs")
        ]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
    
    def _load_base_models(self):
        """Ładowanie wszystkich modeli bazowych"""
        for feature_type, model_path in self.model_paths.items():
            model = load_pretrained_model(
                model_path, 
                self.model_class  # Użyj przekazanej klasy modelu
            )
            if model is not None:
                self.base_models[feature_type] = model.to(self.device)
            
        if not self.base_models:
            raise RuntimeError("Nie udało się załadować żadnych modeli!")
        
        return len(self.base_models)
    
    def _create_dataset(self):
        """
        Tworzenie zbioru danych, jeśli nie został jeszcze utworzony.
        
        Zwraca:
            list: Etykiety dla próbek w zbiorze danych
        """
        if self.dataset is None:
            try:
                # Ładowanie etykiet z pierwszego pliku cech, aby skonfigurować zbiór danych
                feature_file = next(iter(self.feature_files.values()))
                with open(feature_file, 'rb') as f:
                    data = pickle.load(f)
                labels = data['labels']
                
                # Pobierz nazwy klas, jeśli są dostępne
                if 'label_encoder' in data and hasattr(data['label_encoder'], 'classes_'):
                    self.class_names = data['label_encoder'].classes_
                    
                self.dataset = EnsembleDatasetIndexed(self.feature_files, labels)
                return labels
            except Exception as e:
                raise RuntimeError(f"Błąd podczas tworzenia zbioru danych: {e}")
        return self.dataset.labels
    
    def _create_dataloaders(self, dataset, train_indices, val_indices, batch_size=None):
        """
        Tworzenie dataloaderów dla treningu i walidacji.
        
        Argumenty:
            dataset (Dataset): Zbiór danych
            train_indices (list): Indeksy próbek treningowych
            val_indices (list): Indeksy próbek walidacyjnych
            batch_size (int, optional): Rozmiar batcha. Jeśli None, użyj wartości z konfiguracji.
            
        Zwraca:
            tuple: (train_loader, val_loader)
        """
        if batch_size is None:
            batch_size = BATCH_SIZE
            
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=ensemble_collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=ensemble_collate_fn
        )
        
        return train_loader, val_loader
    
    def optimize_weights(self, n_trials=None, timeout=None, n_folds=None, test_size=None):
        """
        Optymalizacja wag ensemble przy użyciu Optuna.
        
        Argumenty:
            n_trials (int, optional): Liczba prób Optuna. Jeśli None, użyj wartości z konfiguracji.
            timeout (int, optional): Limit czasu optymalizacji w sekundach. Jeśli None, użyj wartości z konfiguracji.
            n_folds (int, optional): Liczba foldów walidacji krzyżowej. Jeśli None, użyj wartości z konfiguracji.
            test_size (float, optional): Frakcja danych testowych. Jeśli None, użyj wartości z konfiguracji.
            
        Zwraca:
            dict: Zoptymalizowane wagi dla każdego typu cechy
        """
        # Parametry z konfiguracji lub podane
        n_trials = n_trials if n_trials is not None else OPTUNA_TRIALS
        timeout = timeout if timeout is not None else OPTUNA_TIMEOUT
        n_folds = n_folds if n_folds is not None else CV_FOLDS
        test_size = test_size if test_size is not None else TEST_SPLIT
        
        # Tworzenie zbioru danych, jeśli nie został jeszcze utworzony
        labels = self._create_dataset()
        
        # Tworzenie podziału na trening/test, aby zapobiec wyciekom danych
        indices = np.arange(len(labels))
        train_val_indices, _ = train_test_split(
            indices, 
            test_size=test_size,
            random_state=SEED,
            stratify=labels
        )
        
        # Pobierz etykiety dla zbioru treningowego/walidacyjnego
        train_val_labels = [labels[i] for i in train_val_indices]
        
        # Tworzenie foldów dla CV
        cv_folds = stratified_kfold_split(train_val_labels, n_splits=n_folds)
        
        # Mapowanie indeksów foldów CV na oryginalne indeksy zbioru danych
        original_folds = []
        for fold_train_idx, fold_val_idx in cv_folds:
            original_train_idx = [train_val_indices[i] for i in fold_train_idx]
            original_val_idx = [train_val_indices[i] for i in fold_val_idx]
            original_folds.append((original_train_idx, original_val_idx))
        
        # Ustawienie eksperymentu MLflow do optymalizacji
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        study_name = f"ensemble_optimization_{timestamp}"
        mlflow.set_experiment(study_name)
        
        # Tworzenie studium Optuna
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        sampler = optuna.samplers.TPESampler(seed=SEED)
        
        study = optuna.create_study(
            direction="maximize", 
            pruner=pruner,
            sampler=sampler,
            study_name=study_name
        )
        
        # Definicja funkcji obiektywnej
        def objective(trial):
            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                # Generowanie wag przy użyciu interfejsu sugestii Optuna
                weights = {}
                for ft in self.feature_types:
                    weights[ft] = trial.suggest_float(f"weight_{ft}", 0.0, 1.0)
                
                # Normalizacja wag, aby suma wynosiła 1
                total = sum(weights.values())
                normalized_weights = {ft: w/total for ft, w in weights.items()}
                
                # Logowanie parametrów do MLflow
                mlflow.log_param("trial_number", trial.number)
                for ft, weight in normalized_weights.items():
                    mlflow.log_param(f"weight_{ft}", weight)
                
                # Ocena modeli ensemble w różnych foldach walidacyjnych
                val_accuracies = []
                
                # Użycie tylko podzbioru foldów dla szybszej oceny podczas optymalizacji
                eval_folds = original_folds[:min(3, len(original_folds))]
                
                for i, (train_indices, val_indices) in enumerate(eval_folds):
                    # Tworzenie dataloaderów dla tego folda
                    train_loader, val_loader = self._create_dataloaders(
                        self.dataset, train_indices, val_indices
                    )
                    
                    # Tworzenie modelu ensemble z sugerowanymi wagami
                    ensemble = WeightedEnsembleModel(self.base_models, normalized_weights).to(self.device)
                    
                    # Ocena na zbiorze walidacyjnym
                    results = evaluate_model(ensemble, val_loader, device=self.device)
                    val_accuracies.append(results['accuracy'])
                    
                    # Logowanie metryk dla tego folda
                    mlflow.log_metric(f"fold_{i}_accuracy", results['accuracy'])
                    mlflow.log_metric(f"fold_{i}_f1", results['f1'])
                    
                    # Czyszczenie pamięci
                    del ensemble, train_loader, val_loader
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.dataset.clear_cache()
                
                # Obliczanie i logowanie średnich metryk
                mean_accuracy = np.mean(val_accuracies)
                mlflow.log_metric("mean_val_accuracy", mean_accuracy)
                
                return mean_accuracy
        
        # Uruchomienie optymalizacji
        with mlflow.start_run(run_name=f"optuna_optimization_{timestamp}"):
            # Logowanie parametrów studium
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("timeout", timeout)
            mlflow.log_param("feature_types", self.feature_types)
            
            # Optymalizacja
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            
            # Pobranie najlepszych parametrów
            best_weights = {ft: study.best_params[f"weight_{ft}"] for ft in self.feature_types}
            
            # Normalizacja wag, aby suma wynosiła 1
            total = sum(best_weights.values())
            best_weights = {ft: weight / total for ft, weight in best_weights.items()}
            
            # Logowanie najlepszych parametrów i metryk
            for ft, weight in best_weights.items():
                mlflow.log_param(f"best_weight_{ft}", weight)
            mlflow.log_metric("best_val_accuracy", study.best_value)
            
            # Zapisanie wykresów optymalizacji
            fig_dir = os.path.join(self.output_dir, "optimization_plots")
            
            # Wykres historii optymalizacji
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.title('Historia optymalizacji Optuna')
            plt.tight_layout()
            history_path = os.path.join(fig_dir, "optimization_history.png")
            plt.savefig(history_path)
            mlflow.log_artifact(history_path)
            plt.close()
            
            # Generowanie wykresu ważności
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.title('Ważności parametrów')
            plt.tight_layout()
            importance_path = os.path.join(fig_dir, "parameter_importance.png")
            plt.savefig(importance_path)
            mlflow.log_artifact(importance_path)
            plt.close()
            
        return best_weights
    
    def train_and_evaluate(self, weights, test_size=None, batch_size=None):
        """
        Trenowanie i ocena modelu ensemble z podanymi wagami.
        
        Argumenty:
            weights (dict): Słownik wag dla każdego typu cechy
            test_size (float, optional): Frakcja danych testowych. Jeśli None, użyj wartości z konfiguracji.
            batch_size (int, optional): Rozmiar batcha. Jeśli None, użyj wartości z konfiguracji.
            
        Zwraca:
            tuple: (model, wyniki_testowe)
        """
        # Parametry z konfiguracji lub podane
        test_size = test_size if test_size is not None else TEST_SPLIT
        batch_size = batch_size if batch_size is not None else BATCH_SIZE
        
        # Tworzenie zbioru danych, jeśli nie został jeszcze utworzony
        labels = self._create_dataset()
        
        # Tworzenie podziału na trening/test
        indices = np.arange(len(labels))
        train_indices, test_indices = train_test_split(
            indices, 
            test_size=test_size,
            random_state=SEED,
            stratify=labels
        )
        
        # Logowanie podstawowych parametrów
        with mlflow.start_run():
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("test_samples", len(test_indices))
            mlflow.log_param("train_samples", len(train_indices))
            
            for ft, weight in weights.items():
                mlflow.log_param(f"weight_{ft}", weight)
            
            # Tworzenie modelu ensemble z podanymi wagami
            ensemble_model = WeightedEnsembleModel(self.base_models, weights).to(self.device)
            
            # Tworzenie dataloadera testowego
            test_dataset = Subset(self.dataset, test_indices)
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=ensemble_collate_fn
            )
            
            # Ocena modelu na zbiorze testowym
            test_results = evaluate_model(
                ensemble_model, 
                test_loader, 
                device=self.device,
                class_names=self.class_names, 
                return_probs=True
            )
            
            # Logowanie metryk testowych
            mlflow.log_metric("test_accuracy", test_results['accuracy'])
            mlflow.log_metric("test_precision", test_results['precision'])
            mlflow.log_metric("test_recall", test_results['recall'])
            mlflow.log_metric("test_f1", test_results['f1'])
            
            # Dla każdej klasy logowanie precyzji, recall i F1
            for i, class_name in enumerate(self.class_names):
                if class_name in test_results['report']:
                    class_metrics = test_results['report'][class_name]
                    mlflow.log_metric(f"test_{class_name}_precision", class_metrics['precision'])
                    mlflow.log_metric(f"test_{class_name}_recall", class_metrics['recall'])
                    mlflow.log_metric(f"test_{class_name}_f1-score", class_metrics['f1-score'])
            
            # Tworzenie wykresu macierzy pomyłek
            plt.figure(figsize=(10, 8))
            cm_normalized = test_results['cm'].astype('float') / test_results['cm'].sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title('Znormalizowana macierz pomyłek')
            plt.ylabel('Prawdziwa etykieta')
            plt.xlabel('Przewidywana etykieta')
            plt.tight_layout()
            
            # Zapis wykresu
            cm_dir = os.path.join(self.output_dir, "evaluation")
            cm_path = os.path.join(cm_dir, "confusion_matrix.png")
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close()
            
            # Zapis modelu
            model_dir = os.path.join(self.output_dir, "models")
            model_path = os.path.join(model_dir, "ensemble_model.pt")
            torch.save(ensemble_model.state_dict(), model_path)
            mlflow.log_artifact(model_path)
            
            # Zapis wag
            weights_path = os.path.join(self.output_dir, "optimized_weights.yaml")
            with open(weights_path, "w") as f:
                yaml.dump(weights, f)
            mlflow.log_artifact(weights_path)
        
        return ensemble_model, test_results
    
    def analyze_errors(self, model, test_size=None, batch_size=None):
        """Analizuje przypadki, które model sklasyfikował niepoprawnie"""
        # Parametry z konfiguracji lub podane
        test_size = test_size if test_size is not None else TEST_SPLIT
        batch_size = batch_size if batch_size is not None else BATCH_SIZE
        
        # Tworzenie zbioru danych i podziału
        labels = self._create_dataset()
        indices = np.arange(len(labels))
        _, test_indices = train_test_split(
            indices, 
            test_size=test_size,
            random_state=SEED,
            stratify=labels
        )
        
        # Ewaluacja modelu i znalezienie błędów
        test_dataset = Subset(self.dataset, test_indices)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Batch size 1 dla łatwiejszej analizy
            shuffle=False,
            collate_fn=ensemble_collate_fn
        )
        
        model.eval()
        errors = []
        
        with torch.no_grad():
            for i, (inputs, label) in enumerate(test_loader):
                # Przenieś dane wejściowe na urządzenie
                label = label.to(self.device)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Przepływ do przodu
                output = model(inputs)
                _, pred = torch.max(output, 1)
                
                # Sprawdź, czy klasyfikacja jest błędna
                if pred.item() != label.item():
                    errors.append({
                        'index': test_indices[i],
                        'true_label': label.item(),
                        'pred_label': pred.item(),
                        'confidence': torch.softmax(output, dim=1)[0][pred].item(),
                        'true_label_confidence': torch.softmax(output, dim=1)[0][label].item()
                    })
        
        # Zapisz wyniki analizy
        error_dir = os.path.join(self.output_dir, "error_analysis")
        error_path = os.path.join(error_dir, "classification_errors.csv")
        
        error_df = pd.DataFrame(errors)
        error_df['true_class'] = error_df['true_label'].apply(lambda x: self.class_names[x])
        error_df['pred_class'] = error_df['pred_label'].apply(lambda x: self.class_names[x])
        error_df.to_csv(error_path, index=False)
        
        return error_df
    
    def save_full_model(self, model, path=None):
        """Zapisuje pełny model, nie tylko stan parametrów"""
        if path is None:
            path = os.path.join(self.output_dir, "models", "full_ensemble_model.pt")
            
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_types': model.feature_types,
            'weights': model.get_weights(),
            'temperature': model.temperature.item(),
            'class_names': self.class_names
        }, path)
        return path
        
    @staticmethod
    def load_full_model(path, base_models):
        """Ładuje pełny model z pliku"""
        checkpoint = torch.load(path)
        model = WeightedEnsembleModel(
            base_models, 
            weights=checkpoint['weights'], 
            temperature=checkpoint['temperature']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint['class_names']