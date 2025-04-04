import numpy as np
import torch


class AudioAugmentation:
    @staticmethod
    def add_noise(spectrogram, noise_level=0.005):
        noise = np.random.randn(*spectrogram.shape) * noise_level
        return spectrogram + noise
    
    @staticmethod
    def time_shift(spectrogram, shift_range=5):
        shift = np.random.randint(-shift_range, shift_range)
        if shift > 0:
            return np.pad(spectrogram, ((0, 0), (0, shift)), mode='constant')[:, shift:]
        else:
            return np.pad(spectrogram, ((0, 0), (-shift, 0)), mode='constant')[:, :shift]
    
    @staticmethod
    def frequency_mask(spectrogram, max_mask_width=10, num_masks=1):
        result = spectrogram.copy()
        n_mels, n_steps = spectrogram.shape
        
        for i in range(num_masks):
            width = np.random.randint(1, max_mask_width)
            start = np.random.randint(0, n_mels - width)
            result[start:start+width, :] = result.min()
        
        return result
    
    @staticmethod
    def time_mask(spectrogram, max_mask_width=20, num_masks=1):
        result = spectrogram.copy()
        n_mels, n_steps = spectrogram.shape
        
        for i in range(num_masks):
            width = np.random.randint(1, max_mask_width)
            start = np.random.randint(0, n_steps - width)
            result[:, start:start+width] = result.min()
        
        return result
    
    @staticmethod
    def apply_augmentation(spectrogram, augmentation_prob=0.5):
        aug_spectrogram = spectrogram.copy()
        
        # Losowo wybierz i zastosuj augmentacje
        if np.random.random() < augmentation_prob:
            aug_spectrogram = AudioAugmentation.add_noise(aug_spectrogram)
        if np.random.random() < augmentation_prob:
            aug_spectrogram = AudioAugmentation.frequency_mask(aug_spectrogram)
        if np.random.random() < augmentation_prob:
            aug_spectrogram = AudioAugmentation.time_mask(aug_spectrogram)
        
        return aug_spectrogram
    
    # Niestandardowy zestaw danych z augmentacją
class AugmentedAudioDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, transform=None, augment=False):
        self.features = features
        self.labels = labels
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = self.features[idx].squeeze(0)  # Usunięcie wymiaru kanału
        label = self.labels[idx]
        
        # Zastosuj augmentację, jeśli włączona
        if self.augment and np.random.random() < 0.5:
            feature = AudioAugmentation.apply_augmentation(feature)
        
        # Dodaj z powrotem wymiar kanału
        feature = feature[np.newaxis, :, :]
        
        # Konwersja na tensor
        feature = torch.FloatTensor(feature)
        
        return feature, label

