# dataset.py - klasy do przetwarzania danych
import torch
import torchaudio
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, data, max_length=48000 * 3, sample_rate=16000, augment=False):
        self.data = data
        self.max_length = max_length  # 3 sekundy przy 16kHz
        self.target_sample_rate = sample_rate
        self.augment = augment

        # Mapowanie etykiet emocji na indeksy
        self.emotion_to_idx = {
            "neutral": 0,
            "sadness": 1,
            "happiness": 2,
            "anger": 3,
            "fear": 4,
            "surprised": 5,
        }

        # Transformacje dla augmentacji danych
        if self.augment:
            self.time_mask = torchaudio.transforms.TimeMasking(
                time_mask_param=int(max_length * 0.1)
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Wczytanie pliku audio
        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]

        # Konwersja do tensora PyTorch
        audio_tensor = torch.tensor(audio).float()

        # Zmiana częstotliwości próbkowania jeśli potrzebna
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        # Konwersja na mono jeśli nagranie jest stereo
        if len(audio_tensor.shape) > 1 and audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0)

        # Standaryzacja długości nagrania
        if audio_tensor.shape[0] > self.max_length:
            # Losowe wybranie fragmentu zamiast przycinania
            if self.augment:
                start = torch.randint(
                    0, audio_tensor.shape[0] - self.max_length, (1,)
                ).item()
                audio_tensor = audio_tensor[start : start + self.max_length]
            else:
                # Przycinanie do max_length
                audio_tensor = audio_tensor[: self.max_length]
        else:
            # Uzupełnianie zerami
            padding = torch.zeros(self.max_length - audio_tensor.shape[0])
            audio_tensor = torch.cat([audio_tensor, padding])

        # Normalizacja amplitudy do zakresu [-1, 1]
        if torch.max(torch.abs(audio_tensor)) > 0:
            audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))

        # Znormalizowanie sygnału (standaryzacja)
        if torch.std(audio_tensor) > 0:
            audio_tensor = (audio_tensor - torch.mean(audio_tensor)) / torch.std(
                audio_tensor
            )

        # Augmentacja danych (tylko podczas treningu)
        if self.augment and torch.rand(1).item() < 0.5:
            # Dodanie losowego szumu
            noise_level = 0.005 * torch.rand(1).item()
            audio_tensor = audio_tensor + noise_level * torch.randn_like(audio_tensor)

            # Zastosowanie time masking zamiast SoX dla augmentacji
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(
                    0
                )  # Dodaj wymiar kanału dla TimeMasking
                audio_tensor = self.time_mask(audio_tensor)
                audio_tensor = audio_tensor.squeeze(0)  # Usuń ponownie wymiar kanału

        # Zmiana kształtu na [channels, sequence_length]
        audio_tensor = audio_tensor.unsqueeze(
            0
        )  # Dodanie wymiaru kanału (mono = 1 kanał)

        # Pobranie etykiety emocji
        emotion = item["emotion"]
        label = torch.tensor(self.emotion_to_idx[emotion])

        return audio_tensor, label
