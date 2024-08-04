import torch
from torch import nn

from multi_class_vit.embedding import Embedding


class MultiHeadAttentionBlock(nn.Module):
    """Creates a multi-head attention block ("MHA").
    """

    def __init__(self,
                 embedding_dim: int = 768,  # ViT d_model hyperparameter
                 num_heads: int = 12,  # Amount of attention heads
                 attn_dropout: float = 0):  # doesn't look like the paper uses any dropout in MSABlocks
        super().__init__()

        # Create the Normalization layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Create the Multi-Head Attention (MHA) layer
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                          num_heads=num_heads,
                                                          dropout=attn_dropout,
                                                          batch_first=True)  # our batch dimension comes first

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multi_head_attention(query=x,  # query embeddings
                                                   key=x,  # key embeddings
                                                   value=x,  # value embeddings
                                                   need_weights=False)  # we do need the weights, just the outputs
        return attn_output


class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ("MLP")."""

    def __init__(self,
                 embedding_dim: int = 768,  # d_model
                 linear_hidden_units: int = 3072,  # 4 * d_model
                 dropout: float = 0.1):  # Dropout 10%
        super().__init__()

        # Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # Create the Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=linear_hidden_units),
            nn.GELU(),  # "The MLP contains two layers with a GELU non-linearity (section 3.1)."
            nn.Dropout(p=dropout),
            nn.Linear(in_features=linear_hidden_units,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block."""

    def __init__(self,
                 embedding_dim: int = 768,  # d_model
                 num_heads: int = 12,  # Amount of attention heads
                 attn_dropout: float = 0,  # % of dropout for attention layers
                 linear_hidden_units: int = 3072,  # 4 * d_model
                 linear_dropout: float = 0.1):  # % of dropout for MLP
        super().__init__()

        # MHA block
        self.mha_block = MultiHeadAttentionBlock(embedding_dim=embedding_dim,
                                                 num_heads=num_heads,
                                                 attn_dropout=attn_dropout)

        # MLP block
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                  linear_hidden_units=linear_hidden_units,
                                  dropout=linear_dropout)

    def forward(self, x):
        #  Forward through MHA and apply skip connection (residual connection) for MHA block
        x = self.mha_block(x) + x

        #  Forward through MLP and apply skip connection (residual connection) for MLP block
        x = self.mlp_block(x) + x

        return x


class ViT(nn.Module):
    """Creates a Vision Transformer model."""

    def __init__(self,
                 image_size: int = 224,  # Training resolution of image
                 num_channels: int = 3,  # Number of channels in input image
                 patch_size: int = 16,  # Patch size
                 num_transformer_layers: int = 12,  # Layers from paper for ViT-Base
                 embedding_dim: int = 768,  # d_model for ViT-Base
                 linear_hidden_units: int = 3072,  # MLP size for ViT-Base
                 num_heads: int = 12,  # MHA for ViT-Base
                 attn_dropout: float = 0,  # Dropout for attention projection
                 linear_dropout: float = 0.1,  # % of dropout for MLP layers
                 embedding_dropout: float = 0.1,  # Dropout for patch and position embeddings
                 num_classes: int = 3,
                 batch_size: int = 32):
        super().__init__()

        # Check the image size is divisible by the patch size
        assert image_size % patch_size == 0, (f"Input image size must be divisible by patch size,"
                                              f" image shape: {image_size}, "
                                              f"patch size: {patch_size}")

        # Create learnable embedding
        self.embedding = Embedding(image_size=image_size,
                                   num_channels=num_channels,
                                   patch_size=patch_size,
                                   embedding_dim=embedding_dim,
                                   batch_size=batch_size)

        # Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential())
        self.transformer_encoder = nn.Sequential(  # "*" means "all"
            *[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                      num_heads=num_heads,
                                      linear_hidden_units=linear_hidden_units,
                                      linear_dropout=linear_dropout) for _ in
              range(num_transformer_layers)])

        # Create linear classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    def forward(self, x):
        # Create embedding
        x = self.embedding(x)

        # Run embedding dropout (Appendix B.1)
        x = self.embedding_dropout(x)

        # Pass embedding through transformer encoder layers
        x = self.transformer_encoder(x)

        # Put 0 index logit through classifier
        x = self.classifier(x[:, 0])  # run on each sample in a batch at 0 index

        return x
