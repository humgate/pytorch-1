import torch
from torch.utils.data import DataLoader

from multi_class_vit.dataset import load_dataset_vit
from multi_class_vit.patch_embedding import PatchEmbedding
from util.plotter import Plotter


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

    patchify = PatchEmbedding(in_channels=3, patch_size=16, embedding_dim=3*16*16)

    image_batch, label_batch = next(iter(train_dataloader))
    image = image_batch[0]  # torch.Size([1, 3, 224, 224])
    patch_embedded_image = patchify(image.unsqueeze(0))  # torch.Size([1, 196, 768])

    print("done")
