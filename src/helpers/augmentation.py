import numpy as np
import torch


class AudioAugmentation:
    def add_noise(self, noise_level=0.005):
        noise = np.random.randn(*self.shape) * noise_level
        return self + noise

    def time_shift(self, shift_range=5):
        shift = np.random.randint(-shift_range, shift_range)
        if shift > 0:
            return np.pad(self, ((0, 0), (0, shift)), mode="constant")[:, shift:]
        else:
            return np.pad(self, ((0, 0), (-shift, 0)), mode="constant")[
                :, :shift
            ]

    def frequency_mask(self, max_mask_width=10, num_masks=1):
        result = self.copy()
        n_mels, n_steps = self.shape

        for _i in range(num_masks):
            width = np.random.randint(1, max_mask_width)
            start = np.random.randint(0, n_mels - width)
            result[start : start + width, :] = result.min()

        return result

    def time_mask(self, max_mask_width=20, num_masks=1):
        result = self.copy()
        n_mels, n_steps = self.shape

        for _i in range(num_masks):
            width = np.random.randint(1, max_mask_width)
            start = np.random.randint(0, n_steps - width)
            result[:, start : start + width] = result.min()

        return result

    def apply_augmentation(self, augmentation_prob=0.5):
        aug_spectrogram = self.copy()

        # Augmentacja jest stosowana losowo na podstawie prawdopodobieństwa
        if np.random.random() < augmentation_prob:
            aug_spectrogram = AudioAugmentation.add_noise(aug_spectrogram)
        if np.random.random() < augmentation_prob:
            aug_spectrogram = AudioAugmentation.frequency_mask(aug_spectrogram)
        if np.random.random() < augmentation_prob:
            aug_spectrogram = AudioAugmentation.time_mask(aug_spectrogram)

        return aug_spectrogram

    # Zestaw danych audio z zastosowaną augmentacją


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

        # Augmentacja jest stosowana, gdy jest włączona
        if self.augment and np.random.random() < 0.5:
            feature = AudioAugmentation.apply_augmentation(feature)

        # Przywrócenie wymiaru kanału
        feature = feature[np.newaxis, :, :]

        # Konwersja na tensor
        feature = torch.FloatTensor(feature)

        return feature, label
