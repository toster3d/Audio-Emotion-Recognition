# visualizer.py - wizualizacja wyników z zapisem do folderu
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import json

class ResultVisualizer:
    def __init__(self, output_dir="model_outputs"):
        # Tworzenie katalogu dla wyników, jeśli nie istnieje
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Wyniki będą zapisywane w katalogu: {os.path.abspath(self.output_dir)}")
    
    def plot_confusion_matrix(self, true_labels, predictions, class_names):
        cm = confusion_matrix(true_labels, predictions)
        
        # Normalizacja macierzy konfuzji
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Utworzenie tylko znormalizowanej macierzy konfuzji
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Purples', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Przewidywana etykieta')
        plt.ylabel('Rzeczywista etykieta')
        plt.title('Znormalizowana macierz konfuzji')
        
        # Zapisanie wykresu - tylko plik PNG, bez JSON
        confusion_matrix_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(confusion_matrix_path)
        plt.close()  # Zamknij wykres zamiast plt.show() aby nie wyświetlać go w notebooku
    
    def plot_training_history(self, history):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        
        # Zapisanie wykresu - tylko plik PNG, bez JSON
        history_path = os.path.join(self.output_dir, 'training_history.png')
        plt.tight_layout()
        plt.savefig(history_path)
        plt.close()  # Zamknij wykres zamiast plt.show() aby nie wyświetlać go w notebooku
    
    def print_classification_report(self, true_labels, predictions, class_names):
        # Generowanie raportu klasyfikacji
        report_text = classification_report(true_labels, predictions, target_names=class_names)
        
        # Wyświetlenie raportu
        print("\nClassification Report:")
        print(report_text)
        
        # Zapisanie raportu jako results.txt
        results_path = os.path.join(self.output_dir, 'results.txt')
        with open(results_path, 'w') as f:
            f.write("Classification Report:\n")
            f.write(report_text)
        
        # Dodatkowo tworzenie wykresu raportu klasyfikacji
        report_dict = classification_report(true_labels, predictions, 
                                          target_names=class_names, 
                                          output_dict=True)
        
        # Przygotowanie danych do wykresu
        classes = []
        precision = []
        recall = []
        f1_score = []
        
        for cls in class_names:
            if cls in report_dict:
                classes.append(cls)
                precision.append(report_dict[cls]['precision'])
                recall.append(report_dict[cls]['recall'])
                f1_score.append(report_dict[cls]['f1-score'])
        
        # Tworzenie wykresu
        plt.figure(figsize=(12, 6))
        x = np.arange(len(classes))
        width = 0.25
        
        plt.bar(x - width, precision, width, label='Precision')
        plt.bar(x, recall, width, label='Recall')
        plt.bar(x + width, f1_score, width, label='F1-score')
        
        plt.xlabel('Klasy')
        plt.ylabel('Wartość')
        plt.title('Metryki klasyfikacji')
        plt.xticks(x, classes, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Zapisanie wykresu raportu klasyfikacji
        report_plot_path = os.path.join(self.output_dir, 'classification_report_plot.png')
        plt.savefig(report_plot_path)
        plt.close()
    
    def save_model_summary(self, model, test_loss, test_acc):
        """Zapisuje podsumowanie modelu i wyniki testu do pliku results.txt"""
        # Dopisz do istniejącego pliku results.txt
        results_path = os.path.join(self.output_dir, 'results.txt')
        
        with open(results_path, 'a') as f:
            f.write("\n\nModel Summary:\n")
            f.write(f"Model: {model.__class__.__name__}\n")
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
            f.write(f"Test Loss: {test_loss:.4f}\n\n")
            f.write("Model Structure:\n")
            f.write(str(model))
        
        print(f"Zapisano podsumowanie modelu do pliku: {results_path}")
