import torch
from torch import nn


# Build a model
# Subclass nn.Module
# Create 2 `nn.Linear` layers. Visualize neural network https://playground.tensorflow.org
# implement forward method

class CircleModel0(nn.Module):  # Subclass nn.Module
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)  # out features equals to in features of next level

        # the same using nn.Sequential
        # self.two_linear_layers = nn.Sequential(
        #     nn.Linear(in_features=2, out_features=5),
        #     nn.Linear(in_features=5, out_features=1)
        # )

    def forward(self, x):
        # the same using nn.Sequential
        # return self.two_linear_layers(x)
        return self.layer_2(self.layer_1(x))  # x -> layer_1 -> layer_2 -> output
