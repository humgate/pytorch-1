from torch import nn


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


# Try to improve the model0 by adding more neurons and more linear layers
class CircleModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))


# Try to improve the model1 by adding non-linear ReLU activation functions between linear layers
#
class CircleModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return (
            self.layer_3(
                self.relu(
                    self.layer_2(
                        self.relu(
                            self.layer_1(x)
                        )
                    )
                )
            )
        )






