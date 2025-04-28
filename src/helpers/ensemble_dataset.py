import gc
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path

class EnsembleDatasetIndexed(Dataset):
    """
    Zestaw danych oszczędzający pamięć, który ładuje cechy na żądanie za pomocą indeksów.
    
    Argumenty:
        feature_files (dict): Słownik ścieżek do plików cech w formacie {typ_cechy: ścieżka}
        labels (list): Lista etykiet dla próbek
    """
    def __init__(self, feature_files, labels):
        self.feature_files = {k: Path(v) for k, v in feature_files.items()}
        self.labels = labels
        self.feature_types = list(feature_files.keys())
        self.length = len(labels)
        self.feature_data = {}  # Pamięć podręczna dla załadowanych danych cech
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        features = {}
        for feature_type in self.feature_types:
            # Ładuj plik cech, jeśli nie został jeszcze załadowany
            if feature_type not in self.feature_data:
                try:
                    with open(self.feature_files[feature_type], 'rb') as f:
                        data = pickle.load(f)
                        self.feature_data[feature_type] = data['features']
                except (FileNotFoundError, KeyError) as e:
                    raise RuntimeError(f"Błąd ładowania pliku cech {self.feature_files[feature_type]}: {e}")
            
            # Pobierz cechę dla tego indeksu
            feature = self.feature_data[feature_type][idx]
            
            # Upewnij się, że cecha ma odpowiedni kształt (dodaj wymiar kanału, jeśli to konieczne)
            if feature.ndim == 2:
                feature = np.expand_dims(feature, 0)
            features[feature_type] = feature
            
        label = self.labels[idx]
        return features, label
    
    def clear_cache(self):
        """Wyczyść pamięć podręczną danych cech, aby zwolnić pamięć"""
        self.feature_data = {}
        gc.collect()
        
class EnsembleDataset(Dataset):
    """
    Klasa zestawu danych dla modelu zespołowego obsługująca wiele typów cech.
    
    Argumenty:
        features_dict (dict): Słownik cech w formacie {typ_cechy: [cechy]}
        labels (list): Lista etykiet dla próbek
    """
    def __init__(self, features_dict, labels):
        self.features_dict = features_dict
        self.labels = labels
        self.feature_types = list(features_dict.keys())
        self.length = len(labels)
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        features = {}
        for feature_type in self.feature_types:
            feature = self.features_dict[feature_type][idx]
            # Upewnij się, że cecha ma odpowiedni kształt (dodaj wymiar kanału, jeśli to konieczne)
            if feature.ndim == 2:
                feature = np.expand_dims(feature, 0)
            features[feature_type] = feature
            
        label = self.labels[idx]
        return features, label

def ensemble_collate_fn(batch):
    """
    Funkcja zbierająca do obsługi partii próbek z wieloma cechami.
    
    Argumenty:
        batch (list): Lista tupli (cechy, etykieta)
        
    Zwraca:
        tuple: (Słownik tensorów cech, tensor etykiet)
    """
    features_dict = {}
    labels = []
    
    # Pobierz wszystkie typy cech z pierwszej próbki
    if not batch:
        return {}, torch.tensor([])
        
    feature_types = list(batch[0][0].keys())
    
    # Zbieraj cechy i etykiety ze wszystkich próbek
    for sample_features, sample_label in batch:
        for ft in feature_types:
            if ft not in features_dict:
                features_dict[ft] = []
            features_dict[ft].append(torch.FloatTensor(sample_features[ft]))
        labels.append(sample_label)
    
    # Konwertuj listy na tensory
    for ft in features_dict:
        try:
            features_dict[ft] = torch.stack(features_dict[ft])
        except:
            # Obsłuż tensory o zmiennym rozmiarze (jeśli występują)
            shapes = [f.shape for f in features_dict[ft]]
            max_shape = [max(dim) for dim in zip(*[s for s in shapes])]
            
            padded_features = []
            for feature in features_dict[ft]:
                if feature.shape != tuple(max_shape):
                    padding = []
                    for i, (dim, max_dim) in enumerate(zip(feature.shape, max_shape)):
                        padding.extend([0, max_dim - dim])
                    
                    if any(p > 0 for p in padding):
                        padded_feature = F.pad(feature, tuple(reversed(padding)))
                        padded_features.append(padded_feature)
                    else:
                        padded_features.append(feature)
                else:
                    padded_features.append(feature)
            
            features_dict[ft] = torch.stack(padded_features)
    
    labels = torch.LongTensor(labels)
    return features_dict, labels