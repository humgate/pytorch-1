from torch import nn


class Patchify(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.patch_dim = patch_size * patch_size  # will name it as embed_dim later

    def forward(self, x):
        x = self.unfold(x)
        x = x.view(-1, x.shape[0] // (self.patch_size * self.patch_size), self.patch_size * self.patch_size)
        return x
