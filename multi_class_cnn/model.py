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


class CNNModel0(nn.Module):  # CNN. Replicates TinyVGG from https://poloclub.github.io/cnn-explainer/
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, last_linear_in_features_multiplier: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.output_block = nn.Sequential(
            nn.Flatten(),  # x.shape = torch.Size([10, 7, 7]), flatten shape = torch.Size([10, 49])
            nn.Linear(in_features=hidden_units*last_linear_in_features_multiplier,  # 49!
                      out_features=output_shape)
        )

    def forward(self, x):
        # print(x.shape)  # torch.Size([32, 1, 28, 28]) with batch size = 32
        x = self.conv_block_1(x)  # torch.Size([32, 10, 14, 14])
        # print(f"conv_block1 shape:{x.shape}")
        x = self.conv_block_2(x)  # torch.Size([32, 10, 7, 7])
        # print(f"conv_block2 shape:{x.shape}")
        x = self.output_block(x)  # torch.Size([32, 10])
        # print(f"output_block shape:{x.shape}")
        # return self.output_block(self.conv_block_2(self.conv_block_1(x)))  # operator fusion to speed up gpu computation
        return x

