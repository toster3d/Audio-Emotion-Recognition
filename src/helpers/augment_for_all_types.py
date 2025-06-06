from abc import ABC, abstractmethod

import numpy as np
import torch


# Abstrakcyjna klasa bazowa dla strategii augmentacji
class AudioAugmentationStrategy(ABC):
    @abstractmethod
    def apply_augmentation(self, feature, augmentation_prob=0.5):
        """
        Wykonanie augmentacji dla danego typu reprezentacji audio.

        Args:
            feature: Cecha audio do augmentacji
            augmentation_prob: Prawdopodobieństwo zastosowania każdej augmentacji

        Returns:
            Zaugmentowana cecha audio
        """
        pass

    @abstractmethod
    def is_applicable(self, feature):
        """
        Ocena możliwości zastosowania augmentacji do danego typu cech.

        Args:
            feature: Cecha audio do sprawdzenia

        Returns:
            bool: Czy augmentacja jest odpowiednia dla tej cechy
        """
        pass


# Konkretna implementacja strategii dla spektrogramów
class SpectrogramAugmentation(AudioAugmentationStrategy):
    def apply_augmentation(self, feature, augmentation_prob=0.5):
        aug_feature = feature.copy()

        if np.random.random() < augmentation_prob:
            aug_feature = self._add_noise(aug_feature)
        if np.random.random() < augmentation_prob:
            aug_feature = self._frequency_mask(aug_feature)
        if np.random.random() < augmentation_prob:
            aug_feature = self._time_mask(aug_feature)
        if np.random.random() < augmentation_prob:
            aug_feature = self._time_shift(aug_feature)

        return aug_feature

    def is_applicable(self, feature):
        # Weryfikacja, czy dane mają 2D i wystarczający rozmiar
        return feature.ndim == 2 and min(feature.shape) > 1

    def _add_noise(self, spectrogram, noise_level=0.005):
        noise = np.random.randn(*spectrogram.shape) * noise_level
        return spectrogram + noise

    def _time_shift(self, spectrogram, shift_range=5):
        shift = np.random.randint(-shift_range, shift_range)
        if shift > 0:
            return np.pad(spectrogram, ((0, 0), (0, shift)), mode="constant")[:, shift:]
        else:
            return np.pad(spectrogram, ((0, 0), (-shift, 0)), mode="constant")[
                :, :shift
            ]

    def _frequency_mask(self, spectrogram, max_mask_width=10, num_masks=1):
        result = spectrogram.copy()
        n_mels, n_steps = spectrogram.shape

        # Dostosowanie szerokości maski, jeśli jest zbyt duża
        max_mask_width = min(max_mask_width, n_mels - 1)

        if max_mask_width >= 1:
            for _i in range(num_masks):
                width = np.random.randint(1, max_mask_width + 1)
                start = np.random.randint(0, n_mels - width + 1)
                result[start : start + width, :] = result.min()

        return result

    def _time_mask(self, spectrogram, max_mask_width=20, num_masks=1):
        result = spectrogram.copy()
        n_mels, n_steps = spectrogram.shape

        # Dostosowanie szerokości maski, jeśli jest zbyt duża
        max_mask_width = min(max_mask_width, n_steps - 1)

        if max_mask_width >= 1:
            for _i in range(num_masks):
                width = np.random.randint(1, max_mask_width + 1)
                start = np.random.randint(0, n_steps - width + 1)
                result[:, start : start + width] = result.min()

        return result


# Konkretna implementacja strategii dla MFCC
class MFCCAugmentation(AudioAugmentationStrategy):
    def apply_augmentation(self, feature, augmentation_prob=0.5):
        aug_feature = feature.copy()

        if np.random.random() < augmentation_prob:
            aug_feature = self._add_noise(
                aug_feature, noise_level=0.003
            )  # Zastosowanie mniejszego poziomu szumu dla MFCC
        if np.random.random() < augmentation_prob:
            aug_feature = self._time_mask(
                aug_feature, max_mask_width=5
            )  # Zastosowanie węższych masek czasowych

        return aug_feature

    def is_applicable(self, feature):
        # MFCC również ma reprezentację 2D, ale często ma mniej pasm mel
        return feature.ndim == 2

    def _add_noise(self, feature, noise_level=0.003):
        noise = np.random.randn(*feature.shape) * noise_level
        return feature + noise

    def _time_mask(self, feature, max_mask_width=5, num_masks=1):
        result = feature.copy()
        _, n_steps = feature.shape

        max_mask_width = min(max_mask_width, n_steps - 1)

        if max_mask_width >= 1:
            for _i in range(num_masks):
                width = np.random.randint(1, max_mask_width + 1)
                start = np.random.randint(0, n_steps - width + 1)
                result[:, start : start + width] = result.min()

        return result


# Konkretna implementacja strategii dla jednowymiarowych cech (ZCR, RMS)
class OneDimensionalAugmentation(AudioAugmentationStrategy):
    def apply_augmentation(self, feature, augmentation_prob=0.5):
        aug_feature = feature.copy()

        if np.random.random() < augmentation_prob:
            aug_feature = self._add_noise(aug_feature)

        return aug_feature

    def is_applicable(self, feature):
        # Weryfikacja dla cech jednowymiarowych, które zostały rozszerzone
        return feature.ndim == 2 and feature.shape[0] >= 1

    def _add_noise(self, feature, noise_level=0.001):
        # Zastosowanie mniejszego poziomu szumu dla jednowymiarowych cech
        noise = np.random.randn(*feature.shape) * noise_level
        return feature + noise


class ChromaAugmentation(AudioAugmentationStrategy):
    def apply_augmentation(self, feature, augmentation_prob=0.5):
        aug_feature = feature.copy()

        if np.random.random() < augmentation_prob:
            aug_feature = self._add_noise(aug_feature)
        if np.random.random() < augmentation_prob:
            aug_feature = self._frequency_mask(aug_feature)

        return aug_feature

    def is_applicable(self, feature):
        # Chroma i Tonnetz to cechy 2D z określonymi wymiarami
        return feature.ndim == 2 and min(feature.shape) > 1

    def _add_noise(self, feature, noise_level=0.002):
        # Zastosowanie niższego poziomu szumu dla cech harmonicznych
        noise = np.random.randn(*feature.shape) * noise_level
        return feature + noise

    def _frequency_mask(self, feature, max_mask_width=5, num_masks=1):
        result = feature.copy()
        n_chroma, n_steps = feature.shape

        # Dostosowanie szerokości maski
        max_mask_width = min(max_mask_width, n_chroma - 1)

        if max_mask_width >= 1:
            for _i in range(num_masks):
                width = np.random.randint(1, max_mask_width + 1)
                start = np.random.randint(0, n_chroma - width + 1)
                result[start : start + width, :] = result.min()

        return result


class TempogramAugmentation(AudioAugmentationStrategy):
    def apply_augmentation(self, feature, augmentation_prob=0.5):
        aug_feature = feature.copy()

        if np.random.random() < augmentation_prob:
            aug_feature = self._add_noise(aug_feature)
        if np.random.random() < augmentation_prob:
            aug_feature = self._time_mask(aug_feature)

        return aug_feature

    def is_applicable(self, feature):
        # Tempogram i Delta Tempogram to cechy 2D
        return feature.ndim == 2 and min(feature.shape) > 1

    def _add_noise(self, feature, noise_level=0.002):
        # Zastosowanie niższego poziomu szumu dla cech rytmicznych
        noise = np.random.randn(*feature.shape) * noise_level
        return feature + noise

    def _time_mask(self, feature, max_mask_width=10, num_masks=1):
        result = feature.copy()
        n_tempogram, n_steps = feature.shape

        # Dostosowanie szerokości maski
        max_mask_width = min(max_mask_width, n_steps - 1)

        if max_mask_width >= 1:
            for _i in range(num_masks):
                width = np.random.randint(1, max_mask_width + 1)
                start = np.random.randint(0, n_steps - width + 1)
                result[:, start : start + width] = result.min()

        return result


class SpectralContrastAugmentation(AudioAugmentationStrategy):
    def apply_augmentation(self, feature, augmentation_prob=0.5):
        aug_feature = feature.copy()

        if np.random.random() < augmentation_prob:
            aug_feature = self._add_noise(aug_feature)
        if np.random.random() < augmentation_prob:
            aug_feature = self._frequency_mask(aug_feature)

        return aug_feature

    def is_applicable(self, feature):
        # Spectral Contrast to cecha 2D
        return feature.ndim == 2 and min(feature.shape) > 1

    def _add_noise(self, feature, noise_level=0.002):
        # Zastosowanie niższego poziomu szumu dla kontrastu spektralnego
        noise = np.random.randn(*feature.shape) * noise_level
        return feature + noise

    def _frequency_mask(self, feature, max_mask_width=5, num_masks=1):
        result = feature.copy()
        n_bands, n_steps = feature.shape

        # Dostosowanie szerokości maski
        max_mask_width = min(max_mask_width, n_bands - 1)

        if max_mask_width >= 1:
            for _i in range(num_masks):
                width = np.random.randint(1, max_mask_width + 1)
                start = np.random.randint(0, n_bands - width + 1)
                result[start : start + width, :] = result.min()

        return result


class HPSSAugmentation(AudioAugmentationStrategy):
    def apply_augmentation(self, feature, augmentation_prob=0.5):
        aug_feature = feature.copy()

        if np.random.random() < augmentation_prob:
            aug_feature = self._add_noise(aug_feature)
        if np.random.random() < augmentation_prob:
            aug_feature = self._frequency_mask(aug_feature)
        if np.random.random() < augmentation_prob:
            aug_feature = self._time_mask(aug_feature)
        if np.random.random() < augmentation_prob:
            aug_feature = self._time_shift(aug_feature)

        return aug_feature

    def is_applicable(self, feature):
        # Weryfikacja, czy dane mają 2D i wystarczający rozmiar
        return feature.ndim == 2 and min(feature.shape) > 1

    def _add_noise(self, spectrogram, noise_level=0.005):
        noise = np.random.randn(*spectrogram.shape) * noise_level
        return spectrogram + noise

    def _time_shift(self, spectrogram, shift_range=5):
        shift = np.random.randint(-shift_range, shift_range)
        if shift > 0:
            return np.pad(spectrogram, ((0, 0), (0, shift)), mode="constant")[:, shift:]
        else:
            return np.pad(spectrogram, ((0, 0), (-shift, 0)), mode="constant")[
                :, :shift
            ]

    def _frequency_mask(self, spectrogram, max_mask_width=10, num_masks=1):
        result = spectrogram.copy()
        n_mels, n_steps = spectrogram.shape

        # Dostosowanie szerokości maski, jeśli jest zbyt duża
        max_mask_width = min(max_mask_width, n_mels - 1)

        if max_mask_width >= 1:
            for _i in range(num_masks):
                width = np.random.randint(1, max_mask_width + 1)
                start = np.random.randint(0, n_mels - width + 1)
                result[start : start + width, :] = result.min()

        return result

    def _time_mask(self, spectrogram, max_mask_width=20, num_masks=1):
        result = spectrogram.copy()
        n_mels, n_steps = spectrogram.shape

        # Dostosowanie szerokości maski, jeśli jest zbyt duża
        max_mask_width = min(max_mask_width, n_steps - 1)

        if max_mask_width >= 1:
            for _i in range(num_masks):
                width = np.random.randint(1, max_mask_width + 1)
                start = np.random.randint(0, n_steps - width + 1)
                result[:, start : start + width] = result.min()

        return result


class AudioAugmentationFactory:
    @staticmethod
    def get_strategy(feature_type):
        if feature_type in ["melspectrogram", "spectrogram", "cqt"]:
            return SpectrogramAugmentation()
        elif feature_type in ["mfcc", "delta_mfcc"]:
            return MFCCAugmentation()
        elif feature_type in ["zcr", "rms"]:
            return OneDimensionalAugmentation()
        elif feature_type in ["chroma", "tonnetz"]:
            return ChromaAugmentation()
        elif feature_type in ["tempogram", "delta_tempogram"]:
            return TempogramAugmentation()
        elif feature_type == "spectral_contrast":
            return SpectralContrastAugmentation()
        elif feature_type == "hpss":
            return HPSSAugmentation()
        else:
            # Domyślna strategia dla innych typów
            return SpectrogramAugmentation()


# Klasa główna augmentacji audio
class AudioAugmentation:
    def __init__(self, strategy=None, feature_type=None):
        if strategy:
            self.strategy = strategy
        elif feature_type:
            self.strategy = AudioAugmentationFactory.get_strategy(feature_type)
        else:
            self.strategy = SpectrogramAugmentation()  # Domyślna strategia

    def apply_augmentation(self, feature, augmentation_prob=0.5):
        """
        Wykonanie augmentacji dla podanej cechy audio.

        Args:
            feature: Cecha audio do augmentacji
            augmentation_prob: Prawdopodobieństwo zastosowania każdej augmentacji

        Returns:
            Zaugmentowana cecha audio lub oryginał, jeśli augmentacja nie jest możliwa
        """
        if self.strategy.is_applicable(feature):
            return self.strategy.apply_augmentation(feature, augmentation_prob)
        else:
            # W przypadku braku możliwości augmentacji, zwracany jest oryginał
            return feature

    # Metody statyczne dla kompatybilności wstecznej
    @staticmethod
    def add_noise(spectrogram, noise_level=0.005):
        noise = np.random.randn(*spectrogram.shape) * noise_level
        return spectrogram + noise

    @staticmethod
    def time_shift(spectrogram, shift_range=5):
        shift = np.random.randint(-shift_range, shift_range)
        if shift > 0:
            return np.pad(spectrogram, ((0, 0), (0, shift)), mode="constant")[:, shift:]
        else:
            return np.pad(spectrogram, ((0, 0), (-shift, 0)), mode="constant")[
                :, :shift
            ]

    @staticmethod
    def frequency_mask(spectrogram, max_mask_width=10, num_masks=1):
        result = spectrogram.copy()
        n_mels, n_steps = spectrogram.shape

        for _i in range(num_masks):
            width = np.random.randint(1, max_mask_width)
            start = np.random.randint(0, n_mels - width)
            result[start : start + width, :] = result.min()

        return result

    @staticmethod
    def time_mask(spectrogram, max_mask_width=20, num_masks=1):
        result = spectrogram.copy()
        n_mels, n_steps = spectrogram.shape

        for _i in range(num_masks):
            width = np.random.randint(1, max_mask_width)
            start = np.random.randint(0, n_steps - width)
            result[:, start : start + width] = result.min()

        return result


# Obsługa różnych strategii augmentacji
class AugmentedAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self, features, labels, feature_type=None, transform=None, augment=False
    ):
        self.features = features
        self.labels = labels
        self.transform = transform
        self.augment = augment

        # Utworzenie odpowiedniej strategii augmentacji
        if feature_type:
            self.augmenter = AudioAugmentation(feature_type=feature_type)
        else:
            self.augmenter = AudioAugmentation()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Pobranie cechy z odpowiednim formatem
        feature = self.features[idx]
        label = self.labels[idx]

        # Weryfikacja formatu 4D [batch, channel, height, width]
        if feature.ndim == 3:  # Format [batch, height, width]
            feature = feature.reshape(
                1, feature.shape[0], feature.shape[1], feature.shape[2]
            )

        # Zastosowanie augmentacji, zachowując format 4D
        if self.augment and np.random.random() < 0.5:
            # Wyodrębnienie części 3D do augmentacji
            feature_3d = feature.squeeze(0)
            feature_3d = self.augmenter.apply_augmentation(feature_3d)
            # Przywrócenie do formatu 4D
            feature = feature_3d[np.newaxis, :, :, :]

        # Konwersja na tensor
        feature = torch.FloatTensor(feature)

        return feature, label
