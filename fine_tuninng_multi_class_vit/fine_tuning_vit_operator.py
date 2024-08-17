from pathlib import Path

import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.models import ViT_B_16_Weights
from torchvision.models import vit_b_16

from multi_class_vit.dataset import Food101Subset
from util.model_functions import train
from util.plotter import Plotter


def fine_tuning_vit_operator():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = Path("data")
    labels = ["pizza", "steak", "sushi"]
    num_classes = 3
    batch_size = 50

    weights = ViT_B_16_Weights.IMAGENET1K_V1
    transforms = weights.transforms()  # transforms are stored inside torchvision.models weights
    train_data = Food101Subset(root=data_path,
                               split="train",
                               download=True,
                               labels=labels,
                               transform=transforms)
    test_data = Food101Subset(root=data_path,
                              split="test",
                              download=True,
                              labels=labels,
                              transform=transforms)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=True)

    model = vit_b_16(weights=weights).to(device)
    num_heads = model.encoder.layers[0]
    print(f"Number of heads in the encoder: {num_heads}")
    # Replace the last linear layer output features to 3 (num_classes)
    # out_features = model.heads[-1].in_features
    # model.heads[-1] = nn.Linear(in_features=out_features, out_features=num_classes)
    # summary(model, input_size=(32, 3, 224, 224))
    #
    # torch.manual_seed(42)
    # torch.cuda.manual_seed(42)
    # optimizer = torch.optim.Adam(params=model.parameters(),
    #                              lr=0.0001,
    #                              betas=(0.9, 0.999),
    #                              weight_decay=0.0001)
    #
    # model_results = train(model=model,
    #                       train_data_loader=train_dataloader,
    #                       test_data_loader=test_dataloader,
    #                       loss_fn=nn.CrossEntropyLoss().to(device),
    #                       optimizer=optimizer,
    #                       accuracy_fn=torchmetrics.Accuracy("multiclass", num_classes=num_classes).to(device),
    #                       epochs=4,
    #                       device=torch.device(device))
    #
    # Plotter.plot_loss_accuracy_curves(model_results)
