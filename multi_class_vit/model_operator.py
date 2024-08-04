import torch
from torch.utils.data import DataLoader

from multi_class_vit.dataset import load_dataset_vit
from multi_class_vit.embedding import Embedding
from multi_class_vit.model import MultiHeadAttentionBlock, MLPBlock


def multi_class_vit_model_operator():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data, test_data = load_dataset_vit(
        images_path="data/pizza_steak_sushi/",
        download_url="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
        resized_image_size=224,
        patch_size=16
    )

    batch_size = 32
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False)

    patchify = Embedding(image_size=224,
                         num_channels=3,
                         patch_size=16,
                         embedding_dim=3 * 16 * 16,
                         batch_size=1)

    image_batch, label_batch = next(iter(train_dataloader))
    image = image_batch[0]  # torch.Size([1, 3, 224, 224])
    embedding = patchify(image.unsqueeze(0))
    multi_head_attn_block = MultiHeadAttentionBlock(embedding_dim=768,
                                                    num_heads=12,
                                                    attn_dropout=0)
    attn = multi_head_attn_block(embedding)
    mlp_block = MLPBlock(embedding_dim=768, linear_hidden_units=3072, dropout=0.1)
    mlp_output = mlp_block(attn)

    print("done")
