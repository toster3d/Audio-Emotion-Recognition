import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Definicja modelu ResNet
class AudioResNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(AudioResNet, self).__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        num_features = self.resnet.fc.in_features
        self.dropout = nn.Dropout(dropout_rate)
        self.resnet.fc = nn.Identity()  # Usuń ostatnią warstwę
        self.fc = nn.Linear(num_features, num_classes)
        
        # Inicjalizacja wag
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

