import torch
import torchmetrics
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from multi_class_vit.model import ViT
from util.model_functions import train
from torchinfo import summary

from util.plotter import Plotter


def multi_class_vit_model_operator():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size = 256
    image_size = 28
    num_channels = 1
    num_classes = 10
    patch_size = 14

    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),  # how to transform the data
        target_transform=None  # how to transform the labels/targets
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),  # how to transform the data
        target_transform=None  # how to transform the labels/targets
    )

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=True)

    vit_model = ViT(image_size=image_size,
                    num_channels=num_channels,
                    patch_size=patch_size,
                    num_transformer_layers=8,
                    embedding_dim=num_channels * patch_size * patch_size,  # d_model
                    linear_hidden_units=4 * num_channels * patch_size * patch_size,  # 4 * d_model
                    num_heads=7,
                    linear_dropout=0.1,
                    embedding_dropout=0.1,
                    num_classes=num_classes,
                    batch_size=batch_size).to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    optimizer = torch.optim.Adam(params=vit_model.parameters(),
                                 lr=0.001,
                                 betas=(0.9, 0.999),  # default values and also mentioned in ViT paper section 4.1
                                 weight_decay=0.0001)

    summary(vit_model, input_size=(batch_size, num_channels, image_size, image_size))

    vit_model_results = train(model=vit_model,
                              train_data_loader=train_dataloader,
                              test_data_loader=test_dataloader,
                              loss_fn=nn.CrossEntropyLoss().to(device),
                              optimizer=optimizer,
                              accuracy_fn=torchmetrics.Accuracy("multiclass", num_classes=num_classes).to(device),
                              epochs=40,
                              device=torch.device(device))

    Plotter.plot_loss_accuracy_curves(vit_model_results)
