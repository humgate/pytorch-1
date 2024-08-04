import torch
from torch import nn


class Embedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector and
    adds learnable class token and learnable positional embedding

    Args:
        image_size (int): Image size. Defaults to 224
        num_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): d_model hyperparameter. Size of embedding to turn image into. Defaults to 768 (3*16*16).
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

        # 3 channel 224x224 image will be represented by 196 patches of 768 flatten length each in patch embedding.
        # First convolution layer will turn image into patches
        # According to "Hybrid Architecture" mentioned in the paper https://arxiv.org/abs/2010.11929
        # passing image through the convolutional layer turns it into a series of 768 feature/activation maps,
        # so the output shape of convolution layer will be:
        # torch.Size([1, 768, 14, 14]) - [batch_size, embedding_dim, feature_map_height, feature_map_width].
        # The feature maps all kind of represent the original image. The important thing is these features may change
        # over time as the neural network learns, so these feature maps are learnable embedding of original image.
        self.conv2d_patcher = nn.Conv2d(in_channels=num_channels,
                                        out_channels=embedding_dim,
                                        kernel_size=patch_size,
                                        stride=patch_size,
                                        padding=0)

        # Then we will flatten the square 14x14 feature maps to single dimension vector, so result of flatten
        # will be: torch.Size([1, 768, 196])
        self.flatten = nn.Flatten(start_dim=2,  # only flatten the feature map dimensions into a single vector
                                  end_dim=3)

        # Create learnable class token
        # One 768 length learnable class token will be added to each image represented by 196 patches
        self.class_token = nn.Parameter(data=torch.randn(batch_size, 1, embedding_dim),
                                        requires_grad=True)  # torch.Size([batch_size, 1, 768])

        # Create learnable positional embedding
        # 196 for each patch plus 1 for class token positional embeddings will be added for each image
        num_patches = (image_size * image_size) // (patch_size * patch_size)  # 196
        self.positional_embedding = nn.Parameter(data=torch.randn(batch_size, num_patches + 1, embedding_dim),
                                                 requires_grad=True)  # torch.Size([batch_size, 197, 768])

    def forward(self, x):
        # Assert to check inputs have the correct shape
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, (f"Input image size must be divisible by patch size,"
                                                         f" image shape: {image_resolution}, "
                                                         f"patch size: {self.patch_size}")
        #  Create patch embedding
        x = self.conv2d_patcher(x)
        x = self.flatten(x)
        # Change order so the batch comes in first dimension and embedding in last -> [batch_size, N, P^2•C]
        x = x.permute(0, 2, 1)

        # Concat class_token with patch embedding
        x = torch.cat((self.class_token, x), dim=1)

        # Add positional embedding
        x = self.positional_embedding + x

        return x
