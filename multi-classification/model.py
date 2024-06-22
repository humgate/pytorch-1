from torch import nn


class BlobMultiClass0(nn.Module):
    def __init__(self, in_features, out_features, hidden_units=8):
        """Inits multi-class classification model

        Args:
            in_features (int): Number of input features
            out_features (iny): Number of output features
            hidden_units (int): Number of hidden units between layers, default 8
        Returns:

        Example:
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_units),
            # If we remove this relu and second relu below, the model will become pure linear. But the data is almost
            # separable linearly, so purely linear model will learn and will do preds with just a bit worse accuracy
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=out_features),
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
