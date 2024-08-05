import torch
from torch import nn


class Embedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector and
    adds learnable class token and learnable positional embedding
    Args:
        image_size (int): Image size. Defaults to 224
        num_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): d_model hyperparameter. Size of embedding to turn image into. Defaults to 768.
    """

    def __init__(self,
                 image_size: int = 224,
                 num_channels: int = 3,
                 patch_size: int = 16,
                 embedding_dim: int = 768,
                 batch_size: int = 32):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2

        # Create learnable class token
        self.class_token = nn.Parameter(data=torch.randn(batch_size, 1, embedding_dim),
                                        requires_grad=True)  # torch.Size([batch_size, 1, 768])
        # Create learnable positional embedding
        self.positional_embedding = (torch.randn(batch_size, self.num_patches + 1, embedding_dim)
                                     .to(device=torch.device("cuda")))

    def forward(self, x):
        # Assert to check inputs have the correct shape
        assert x.shape[-1] == x.shape[-2] == self.image_size
        assert x.shape[-3] == self.num_channels

        # Reshape into a batch of patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()

        # Flatten the patches
        x = x.view(x.shape[0], -1, self.num_channels * self.patch_size * self.patch_size)

        # Check shape
        assert x.shape[1] == self.num_patches
        assert x.shape[2] == self.num_channels * self.patch_size * self.patch_size

        # Concat class_token with patch embedding
        x = torch.cat((self.class_token, x), dim=1)

        # Add positional embedding
        x = self.positional_embedding + x
        return x
