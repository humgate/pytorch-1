from torch import nn


class FashionMNISTCNNModel0(nn.Module):  # CNN. Replicates TinyVGG from https://poloclub.github.io/cnn-explainer/
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
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
            nn.Linear(in_features=hidden_units * 49,  # 49!
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
        return x
