import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.base_model import BaseModel

class SimpleCNN(BaseModel):
    def __init__(self):
        # super(SimpleCNN, self).__init__()
        super(SimpleCNN, self).__init__(
            name="CNNNoNumeric",
            description="Basic CNN",
            input_requires_mask=False,
            input_requires_numerics=False,
            supports_variable_sequence_length=False
        )

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5, 21), padding=(2, 10))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(5, 21), padding=(2, 10))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 11), padding=(1, 5))

        # Max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        # Fully connected layers
        # Calculate the size after convolution and pooling
        # Input size: (batch_size, 1, 32, 600)
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Fully connected layer
        self.fc = nn.Linear(128, 1)


        # After conv and pooling layers: (batch_size, 64, 4, 75)
        # self.fc1 = nn.Linear(64 * 4 * 75, 128)
        # self.fc2 = nn.Linear(128, 1)  # Output a single value for regression

    def analyse_number_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Total number of parameters: {params}")
        # Check number of params in per layer:

        # Params in conv1
        params = sum([np.prod(p.size()) for p in self.conv1.parameters()])
        print(f"Total number of parameters in conv1: {params}")

        # Params in conv2
        params = sum([np.prod(p.size()) for p in self.conv2.parameters()])
        print(f"Total number of parameters in conv2: {params}")

        # Params in conv3
        params = sum([np.prod(p.size()) for p in self.conv3.parameters()])
        print(f"Total number of parameters in conv3: {params}")


        # Params in fc1
        params = sum([np.prod(p.size()) for p in self.fc.parameters()])
        print(f"Total number of parameters in fc1: {params}")

        # # Params in fc2
        # params = sum([np.prod(p.size()) for p in self.fc2.parameters()])
        # print(f"Total number of parameters in fc2: {params}")

    def forward(self, x):

        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.gap(x)
        # Flatten the tensor
        x = x.view( -1, 128 )
        x = self.fc(x)  # Output layer without activation (regression)
        return x