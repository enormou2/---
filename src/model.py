"""
CNN模型定义模块
包含：自定义CNN、MLP基线模型、逻辑回归基线
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GarbageCNN(nn.Module):
    """垃圾分类CNN模型 —— 4个卷积块 + 全局平均池化 + 全连接"""

    def __init__(self, num_classes=6, input_channels=3, dropout_rate=0.5):
        super(GarbageCNN, self).__init__()

        # 卷积块1
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # 卷积块2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # 卷积块3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # 卷积块4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接分类头
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(256, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        features = {}

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        features['conv1'] = x

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        features['conv2'] = x

        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        features['conv3'] = x

        x = F.relu(self.bn4(self.conv4(x)))
        features['conv4'] = x

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc_out(x)

        if return_features:
            return x, features
        return x


class MLPBaseline(nn.Module):
    """MLP基线模型 —— 用于对比CNN的优势"""

    def __init__(self, input_size, num_classes=6, hidden_sizes=(512, 256, 128)):
        super(MLPBaseline, self).__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = h
        layers.append(nn.Linear(prev_size, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.net(x)


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    print("=" * 60)
    print("模型定义测试")
    print("=" * 60)

    # 测试CNN
    cnn = GarbageCNN(num_classes=6, input_channels=3)
    total, trainable = count_parameters(cnn)
    print(f"\nCNN模型:")
    print(f"  总参数量: {total:,}")
    print(f"  可训练参数: {trainable:,}")

    dummy_input = torch.randn(1, 3, 128, 128)
    output = cnn(dummy_input)
    print(f"  输入形状: (1, 3, 128, 128)")
    print(f"  输出形状: {output.shape}")

    # 测试MLP
    mlp = MLPBaseline(input_size=3 * 128 * 128, num_classes=6)
    total_mlp, _ = count_parameters(mlp)
    print(f"\nMLP基线模型:")
    print(f"  总参数量: {total_mlp:,}")
    print(f"  CNN参数仅为MLP的 {total / total_mlp * 100:.1f}%")
