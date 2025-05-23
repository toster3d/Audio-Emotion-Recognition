import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(AudioCNN, self).__init__()
        
        # Warstwy konwolucyjne 1D dla danych audio z lepszą progresją filtrów
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Warstwa globalna do redukcji rozmiaru
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Warstwy dropout dla lepszej regularyzacji
        self.dropout1 = nn.Dropout(0.3)  # Mniejszy dropout po warstwach konwolucyjnych
        self.dropout2 = nn.Dropout(0.5)  # Większy dropout po warstwach fully-connected
        
        # Warstwy w pełni połączone z lepszą architekturą
        self.fc1 = nn.Linear(256, 512)
        self.bn_fc = nn.BatchNorm1d(512)  # Dodany BatchNorm dla warstwy FC
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Bloki konwolucyjne z dropout
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = self.dropout1(x)  # Dropout po konwolucjach
        
        # Globalne pooling - redukuje wymiar przestrzenny do 1
        x = self.global_pool(x)
        
        # Spłaszczenie tensora
        x = x.view(x.size(0), -1)
        
        # Warstwy w pełni połączone z BatchNorm
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout2(x)  # Dropout między warstwami FC
        x = self.fc2(x)
        
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

