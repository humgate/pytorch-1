import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader

from multi_class_vit.dataset import load_dataset_vit

from multi_class_vit.model import ViT
from util.model_functions import train
from torchinfo import summary

from util.plotter import Plotter


def multi_class_vit_model_operator():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data, test_data = load_dataset_vit(
        images_path="data/pizza_steak_sushi/",
        download_url="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
        resized_image_size=224,
        patch_size=16
    )

    batch_size = 16
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=True)

    vit_model = ViT(image_size=224,
                    num_channels=3,
                    patch_size=16,
                    embedding_dim=3 * 16 * 16,
                    linear_hidden_units=4 * 768,
                    num_heads=12,
                    linear_dropout=0.1,
                    embedding_dropout=0.1,
                    num_classes=3,
                    batch_size=batch_size).to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    optimizer = torch.optim.Adam(params=vit_model.parameters(),
                                 lr=3e-3,  # Base LR from for ViT from paper
                                 betas=(0.9, 0.999),  # default values and also mentioned in ViT paper section 4.1
                                 weight_decay=0.3)  # from the ViT paper section 4.1 (Training & Fine-tuning)

    summary(vit_model, input_size=(batch_size, 3, 224, 224))

    vit_model_results = train(model=vit_model,
                              train_data_loader=train_dataloader,
                              test_data_loader=test_dataloader,
                              loss_fn=nn.CrossEntropyLoss().to(device),
                              optimizer=optimizer,
                              accuracy_fn=torchmetrics.Accuracy("multiclass", num_classes=3).to(device),
                              epochs=5,
                              device=torch.device(device))

    Plotter.plot_loss_accuracy_curves(vit_model_results)


