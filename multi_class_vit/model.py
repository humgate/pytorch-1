import torch
from torch import nn


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
            nn.Linear(in_features=linear_hidden_units,  # needs to take same in_features as out_features of layer above
                      out_features=embedding_dim),  # take back to embedding_dim
            nn.Dropout(p=dropout)  # "Dropout, when used, is applied after every dense layer."
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

