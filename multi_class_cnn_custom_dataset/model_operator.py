from pathlib import Path

import torch
import torchmetrics
from torch import nn
from torch.utils.data import DataLoader

from multi_class_cnn.model import CNNModel0
from multi_class_cnn_custom_dataset.dataset_operator import load_dataset, read_image_to_device
from util.model_functions import train
from util.plotter import Plotter


def multi_class_cnn_model_operator():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data, test_data = load_dataset(
        "data/pizza_steak_sushi/",
        "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    )

    batch_size = 32
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)  # remove images order
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 shuffle=False)
    model_4 = CNNModel0(input_shape=3,
                        hidden_units=10,
                        output_shape=len(train_data.classes),
                        last_linear_in_features_multiplier=256).to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    epochs = 3
    model_4_results = train(model=model_4,
                            train_data_loader=train_dataloader,
                            test_data_loader=test_dataloader,
                            loss_fn=nn.CrossEntropyLoss().to(device),
                            optimizer=torch.optim.Adam(params=model_4.parameters(), lr=0.01),
                            accuracy_fn=torchmetrics.Accuracy("multiclass", num_classes=3).to(device),
                            epochs=epochs,
                            device=torch.device(device))

    Plotter.plot_loss_accuracy_curves(model_4_results)

    #  Test model on custom image not existing in any train and test datasets
    image = read_image_to_device(Path("data/pizza-test-image.jpeg"), device)
    model_4.eval()
    with torch.inference_mode():
        pred = model_4.forward(image).argmax()
    print(f"Image classified as {train_data.classes[pred].title()}")  # Image classified as Pizza
