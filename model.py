# What the model does:
# Starts with random values for weight and bias
# Looks at training data and adjusts the random values to better represent (get closer to) the ideal values
# (weight and bias) we used to create the data.
#
# It will use two main algorithms:
# 1. Gradient descent
# Back propagation
import torch
from torch import nn


class LinearRegressionModel(nn.Module):  # nn.Module  - base class for any neural network modules.
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                               requires_grad=True,  # will be updated through gradient descend
                                               dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                            requires_grad=True,  # will be updated in through descend
                                            dtype=torch.float))

    # Forward method defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x is input, returns torch.Tensor
        return self.weights * x + self.bias  # we know that relationship is linear, we do not know weight and bias
