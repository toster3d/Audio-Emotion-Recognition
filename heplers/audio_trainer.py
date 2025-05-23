# trainer.py - funkcje do treningu i ewaluacji modelu z zapisem do folderu
import torch
from tqdm import tqdm
import os

class ModelTrainer:
    def __init__(self, model, criterion, optimizer, device="cuda", output_dir="model_outputs"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)
        self.scheduler = None
        self.output_dir = output_dir
        
        # Tworzenie katalogu dla wyników, jeśli nie istnieje
        os.makedirs(self.output_dir, exist_ok=True)
    
    def setup_scheduler(self, scheduler):
        self.scheduler = scheduler
    
    def train_model(self, train_loader, val_loader, num_epochs=20, patience=5):
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        
        for epoch in range(num_epochs):
            # Tryb treningowy
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zerowanie gradientów
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Backward pass i optymalizacja
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Aktualizacja statystyk
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Aktualizacja paska postępu
                pbar.set_postfix({"loss": loss.item(), "acc": train_correct/train_total})
            
            train_loss = train_loss / len(train_loader.dataset)
            train_acc = train_correct / train_total
            
            # Ewaluacja na zbiorze walidacyjnym
            val_loss, val_acc = self.evaluate_model(val_loader)
            
            # Zapisywanie historii
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            # Aktualizacja learning rate na podstawie straty walidacyjnej
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Zapisywanie najlepszego modelu i early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                
                # Ścieżka do zapisania modelu
                model_path = os.path.join(self.output_dir, "best_model.pth")
                torch.save(self.model.state_dict(), model_path)
                print(f"Model saved with validation accuracy: {val_acc:.4f} to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}. Best model was at epoch {best_epoch+1}.")
                    break
        
        return history
    
    def evaluate_model(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        avg_loss = total_loss / len(data_loader.dataset)
        
        return avg_loss, accuracy
    
    def get_predictions(self, data_loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return all_labels, all_preds
    
    def load_best_model(self):
        """Ładuje najlepszy model z katalogu wyjściowego"""
        model_path = os.path.join(self.output_dir, "best_model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded best model from {model_path}")
        else:
            print(f"Warning: Could not find best model at {model_path}")