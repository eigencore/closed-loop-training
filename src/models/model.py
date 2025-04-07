import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Bloque de convolución con normalización y activación"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    """Bloque residual simple para ResNet"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Conexión residual (shortcut)
        self.shortcut = nn.Sequential()
        # Si cambia el tamaño o el número de canales, ajustamos la conexión
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # Suma residual
        out = F.relu(out)
        return out

class SimpleCNN(nn.Module):
    """Una CNN simple para clasificación CIFAR-10"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(self.conv1(x))       # -> 32x16x16
        x = self.pool(self.conv2(x))       # -> 64x8x8
        x = self.pool(self.conv3(x))       # -> 128x4x4
        
        x = x.view(-1, 128 * 4 * 4)        # Aplanar
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class SmallResNet(nn.Module):
    """Una versión pequeña de ResNet para CIFAR-10"""
    def __init__(self, num_blocks=[2, 2, 2], num_classes=10):
        super(SmallResNet, self).__init__()
        self.in_channels = 64
        
        # Capa inicial
        self.conv1 = ConvBlock(3, 64)
        
        # Capas residuales
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        
        # Clasificador
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)                  # -> 64x32x32
        
        x = self.layer1(x)                 # -> 64x32x32
        x = self.layer2(x)                 # -> 128x16x16
        x = self.layer3(x)                 # -> 256x8x8
        
        x = self.avgpool(x)                # -> 256x1x1
        x = x.view(x.size(0), -1)          # Aplanar
        x = self.fc(x)
        
        return x