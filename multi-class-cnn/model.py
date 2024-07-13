import torch
from torch import nn


class FashionMNISTModel0(nn.Module):  # non CNN, just sequential linear
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Flatten(),  # transform image from torch.Size([1, 28, 28]) to torch.Size([1, 784])
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


class FashionMNISTModel1(nn.Module):  # non CNN, non-linear by adding ReLU activations
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Flatten(),  # transform image from torch.Size([1, 28, 28]) to torch.Size([1, 784])
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_shape),
            nn.ReLU()
        )

    def forward(self, x: torch.tensor):
        return self.linear_layer_stack(x)
