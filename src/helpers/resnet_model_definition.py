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


# class BasicBlock1D(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock1D, self).__init__()
#         self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out

# class AudioResNet1D(nn.Module):
#     def __init__(self, num_classes=6, dropout_rate=0.5):
#         super(AudioResNet1D, self).__init__()
        
#         # Inicjalizacja warstw początkowych
#         self.inplanes = 64
#         self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
#         # Warstwy rezydualne
#         self.layer1 = self._make_layer(BasicBlock1D, 64, 2)
#         self.layer2 = self._make_layer(BasicBlock1D, 128, 2, stride=2)
#         self.layer3 = self._make_layer(BasicBlock1D, 256, 2, stride=2)
#         self.layer4 = self._make_layer(BasicBlock1D, 512, 2, stride=2)
        
#         # Warstwa końcowa
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc = nn.Linear(512, num_classes)
        
#         # Inicjalizacja wag
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 nn.init.constant_(m.bias, 0)
    
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv1d(self.inplanes, planes * block.expansion, 
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)
        
#     def forward(self, x):
#         # x ma wymiar [batch, 1, time]
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.dropout(x)
#         x = self.fc(x)
        
#         return x
