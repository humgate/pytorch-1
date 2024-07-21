import random

import torchvision
from torchvision import datasets
from torchvision.datasets import FashionMNIST

from util.model_functions import *
from util.plotter import Plotter
from .model import *


def multi_class_pred_cnn():
    model_name = "cnn_fashion_mnist.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    #  random.seed(42)
    test_images = []
    test_labels = []
    for image, label in random.sample(list(test_data), k=32):  # Get random 10 images from MNIST dataset
        test_images.append(image)
        test_labels.append(label)

    model_2 = CNNModel0(input_shape=1, hidden_units=10, output_shape=len(train_data.classes)).to(device)
    load_model(model_2, model_name)

    pred_labels = make_predictions(model=model_2,
                                   data=test_images,
                                   device=torch.device(device))

    Plotter.show_batch_images(test_data, test_images, test_labels, pred_labels)

    # Make predictions on whole test_data
    test_images = []
    test_labels = []
    for image, label in list(test_data):
        test_images.append(image)
        test_labels.append(label)

    pred_labels = make_predictions(model=model_2,
                                   data=test_images,
                                   device=torch.device(device))
    # Confusion matrix
    Plotter.show_confusion_matrix(test_data, pred_labels)
