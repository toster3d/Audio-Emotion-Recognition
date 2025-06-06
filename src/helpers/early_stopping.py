import torch


class EarlyStopping:
    """
    Klasa odpowiedzialna za wczesne zatrzymywanie treningu modelu na podstawie strat walidacyjnych.

    Atrybuty:
        patience (int): Liczba epok do monitorowania przed zatrzymaniem treningu.
        min_delta (float): Minimalna poprawa strat, aby zresetować licznik.
        counter (int): Licznik epok bez poprawy.
        best_loss (float): Najlepsza dotychczasowa strata walidacyjna.
        early_stop (bool): Flaga wskazująca, czy trening powinien zostać zatrzymany.
        path (str): Ścieżka do pliku, w którym zapisywany jest model.
    """

    def __init__(self, patience=7, min_delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        """
        Metoda wywoływana w każdej epoce, aby ocenić, czy należy zatrzymać trening.

        Args:
            val_loss (float): Strata walidacyjna uzyskana w bieżącej epoce.
            model: Model, którego stan ma być zapisany, jeśli strata ulegnie poprawie.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"Counter wczesnego zatrzymywania: {self.counter} z {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """
        Zapisuje stan modelu, gdy strata walidacyjna ulega poprawie.

        Args:
            model: Model, którego stan ma być zapisany.
        """
        print(f"Strata walidacyjna uległa poprawie. Zapis modelu do {self.path}")
        torch.save(model.state_dict(), self.path)
