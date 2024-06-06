# model.py

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, input_shape):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (input_shape[0] // 4) * (input_shape[1] // 4), 128)
        self.fc2 = nn.Linear(128, 5)  # Output shape should match the number of classes/boxes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * (x.shape[2] * x.shape[3]))  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_model(input_shape):
    model = SimpleCNN(input_shape)
    return model
