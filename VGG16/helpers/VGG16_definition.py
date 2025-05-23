"""
Moduł zawierający definicję modelu VGG16
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import *

def build_vgg16_model(num_classes=NUM_CLASSES):
    """
    Budowanie modelu VGG16 w PyTorch
    
    Args:
        num_classes: Liczba klas wyjściowych
        
    Returns:
        model: Model PyTorch
    """
    print("🔄 Budowanie modelu VGG16 w PyTorch...")

    # Ładowanie pre-trenowanego modelu VGG16
    model = models.vgg16(pretrained=True)
    
    # Zamrożenie wag
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Modyfikacja ostatniej warstwy dla naszej liczby klas
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, num_classes)
    
    print(f"✅ Model VGG16 został pomyślnie zbudowany z {num_classes} klasami wyjściowymi")
    
    return model

def train_model(model, train_loader, val_loader, device, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    """
    Trenowanie modelu VGG16 w PyTorch
    
    Args:
        model: Model do trenowania
        train_loader: DataLoader z danymi treningowymi
        val_loader: DataLoader z danymi walidacyjnymi
        device: Urządzenie (cuda/cpu)
        epochs: Liczba epok
        learning_rate: Współczynnik uczenia
        
    Returns:
        model: Wytrenowany model
        history: Historia treningu (słownik z metrykami)
    """
    print("🚀 Rozpoczynam trenowanie modelu...")
    
    # Kryterium i optymalizator
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Scheduler zmniejszający learning rate
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=REDUCE_LR_FACTOR, 
                                  patience=REDUCE_LR_PATIENCE, min_lr=MIN_LR, verbose=True)
    
    # Historia treningu
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    # Najlepszy model
    best_val_acc = 0
    best_model_weights = None
    
    # Czas rozpoczęcia
    start_time = time.time()
    print(f"   Trenowanie rozpoczęte, może to potrwać dłuższą chwilę...")
    
    # Trenowanie
    for epoch in range(epochs):
        print(f"\nEpoka {epoch+1}/{epochs}")
        
        # --- Tryb treningu ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Pasek postępu dla zbioru treningowego
        train_pbar = tqdm(train_loader, desc=f"Trening")
        
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zerowanie gradientów
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statystyki
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Aktualizacja paska postępu
            train_pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_correct/train_total:.4f}"
            })
        
        # Średnia strata i dokładność treningowa
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # --- Tryb walidacji ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Wyłączamy obliczanie gradientów
        with torch.no_grad():
            # Pasek postępu dla zbioru walidacyjnego
            val_pbar = tqdm(val_loader, desc=f"Walidacja")
            
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statystyki
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Aktualizacja paska postępu
                val_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{val_correct/val_total:.4f}"
                })
        
        # Średnia strata i dokładność walidacyjna
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Aktualizacja schedulera
        scheduler.step(val_loss)
        
        # Wyświetlenie wyników epoki
        print(f"   Epoka {epoch+1}/{epochs}: " + 
              ("🔼" if val_acc > best_val_acc else "🔽") + 
              f" val_acc={val_acc:.4f}, val_loss={val_loss:.4f}, " + 
              f"train_acc={train_acc:.4f}, train_loss={train_loss:.4f}")
        
        # Zapisanie najlepszego modelu
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict().copy()
            print(f"   Znaleziono lepszy model! Zapisuję wagi.")
    
    # Koniec treningu
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"✅ Trenowanie zakończone! Czas: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"   Najlepsza dokładność walidacyjna: {best_val_acc:.4f}")
    
    # Przywrócenie najlepszego modelu
    if best_model_weights:
        model.load_state_dict(best_model_weights)
    
    return model, history

def save_model(model, history, filename='VGG16_model.pt'):
    """
    Zapisywanie modelu PyTorch i historii treningu
    
    Args:
        model: Model do zapisania
        history: Historia treningu
        filename: Nazwa pliku
    """
    print(f"💾 Zapisywanie modelu do pliku {filename}...")
    try:
        # Zapisanie modelu
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history
        }, filename)
        print(f"   Model zapisany pomyślnie!")
    except Exception as e:
        print(f"❌ Wystąpił błąd podczas zapisywania modelu: {str(e)}")
