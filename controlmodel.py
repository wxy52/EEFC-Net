import torch
import torch.nn as nn
import torch.nn.functional as F


class bulid_CNN(nn.Module):
    def __init__(self):
        super(bulid_CNN, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)  # (1, 200) -> (32, 200)
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # (32, 200) -> (32, 100)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)  # (32, 100) -> (64, 100)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # (64, 100) -> (64, 50)

        self.fc1 = nn.Linear(64 * 50, 128)  # Flattened size (64 * 50) -> 128
        self.fc2 = nn.Linear(128, 1)  # Output layer for binary classification

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension, making it (batch_size, 1, 200)

        x = F.relu(self.conv1(x))  # (batch_size, 32, 200)
        x = self.pool1(x)  # (batch_size, 32, 100)

        x = F.relu(self.conv2(x))  # (batch_size, 64, 100)
        x = self.pool2(x)  # (batch_size, 64, 50)

        x = x.view(x.size(0), -1)  # Flatten (batch_size, 64 * 50)

        x = F.relu(self.fc1(x))  # (batch_size, 128)

        return x


class bulid_model(nn.Module):
    def __init__(self):
        super(bulid_model, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)  # (1, 160) -> (32, 160)
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # (32, 160) -> (32, 80)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)  # (32, 80) -> (64, 80)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # (64, 80) -> (64, 40)

        self.fc1 = nn.Linear(64 * 40, 128)  # Flattened size (64 * 40) -> 128
        self.fc2 = nn.Linear(128, 1)  # Output layer for binary classification

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension, making it (batch_size, 1, 160)

        x = F.relu(self.conv1(x))  # (batch_size, 32, 160)
        x = self.pool1(x)  # (batch_size, 32, 80)

        x = F.relu(self.conv2(x))  # (batch_size, 64, 80)
        x = self.pool2(x)  # (batch_size, 64, 40)

        x = x.view(x.size(0), -1)  # Flatten (batch_size, 64 * 40) = (batch_size, 2560)

        x = F.relu(self.fc1(x))  # (batch_size, 128)

        return x

class bulid_CN(nn.Module):
    def __init__(self):
        super(bulid_CN, self).__init__()

        # 第一卷积层：输入大小 (batch_size, 1, 116)，输出大小 (batch_size, 32, 116)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)  # (1, 116) -> (32, 116)
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # (32, 116) -> (32, 58)

        # 第二卷积层：输入大小 (batch_size, 32, 58)，输出大小 (batch_size, 64, 58)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)  # (32, 58) -> (64, 58)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # (64, 58) -> (64, 29)

        # 全连接层：输入大小从 (64 * 29) 展平到 128，输出大小为 128
        self.fc1 = nn.Linear(64 * 29, 128)  # Flattened size (64 * 29) -> 128

        # 二分类输出层：输出为 1，表示二分类的概率
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # 输入数据是 (batch_size, 116)，需要加一个通道维度变为 (batch_size, 1, 116)
        x = x.unsqueeze(1)  # Add channel dimension, making it (batch_size, 1, 116)

        # 第一卷积层和池化
        x = F.relu(self.conv1(x))  # (batch_size, 32, 116)
        x = self.pool1(x)  # (batch_size, 32, 58)

        # 第二卷积层和池化
        x = F.relu(self.conv2(x))  # (batch_size, 64, 58)
        x = self.pool2(x)  # (batch_size, 64, 29)

        # 展平层，将 (batch_size, 64, 29) 展平为 (batch_size, 64 * 29)
        x = x.view(x.size(0), -1)  # Flatten (batch_size, 64 * 29) = (batch_size, 1856)

        # 全连接层
        x = F.relu(self.fc1(x))  # (batch_size, 128)

        return x


if __name__ == "__main__":
    # Example of using this function to create a model
    model = bulid_CN()

    # Example of input (batch_size=32, sequence_length=200)
    input_data = torch.randn(8, 116)
    # Forward pass through the model
    output = model(input_data)
    print(output)  # Expected output shape: (batch_size, 1)
