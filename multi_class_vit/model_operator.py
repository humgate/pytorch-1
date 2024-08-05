import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader

from multi_class_vit.dataset import load_small_subset_dataset, load_full_dataset_for_selected_labels

from multi_class_vit.model import ViT
from util.model_functions import train
from torchinfo import summary

from util.plotter import Plotter


def multi_class_vit_model_operator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 150
    image_size = 64

    # train_data, test_data = load_small_subset_dataset(
    #     images_path="data/large_pizza_sushi_steak_reviewed/",
    #     download_url="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    #     resized_image_size=image_size)

    # We will use full set (750 train and 250 test images per class) of pizza, steak and sushi from Food101 dataset
    train_data, test_data = load_full_dataset_for_selected_labels(
        images_path="data",
        resized_image_size=image_size,
        labels=["pizza", "sushi", "steak"])

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 drop_last=True)

    vit_model = ViT(image_size=image_size,
                    num_channels=3,
                    patch_size=16,
                    num_transformer_layers=12,
                    embedding_dim=3 * 16 * 16,  # d_model
                    linear_hidden_units=4 * 768,  # 4 * d_model
                    num_heads=12,
                    linear_dropout=0.1,
                    embedding_dropout=0.1,
                    num_classes=3,
                    batch_size=batch_size).to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    optimizer = torch.optim.Adam(params=vit_model.parameters(),
                                 lr=0.001,  # Base LR from for ViT from paper
                                 betas=(0.9, 0.999),  # default values and also mentioned in ViT paper section 4.1
                                 weight_decay=0.3)  # from the ViT paper section 4.1 (Training & Fine-tuning)

    summary(vit_model, input_size=(batch_size, 3, image_size, image_size))

    vit_model_results = train(model=vit_model,
                              train_data_loader=train_dataloader,
                              test_data_loader=test_dataloader,
                              loss_fn=nn.CrossEntropyLoss().to(device),
                              optimizer=optimizer,
                              accuracy_fn=torchmetrics.Accuracy("multiclass", num_classes=3).to(device),
                              epochs=10,
                              device=torch.device(device))

    Plotter.plot_loss_accuracy_curves(vit_model_results)
