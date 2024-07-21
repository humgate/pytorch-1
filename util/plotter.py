import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from matplotlib import pyplot as plt
from util.helper_functions import plot_decision_boundary
import torch
import torchmetrics
from torchmetrics import ConfusionMatrix
import mlxtend
from mlxtend.plotting import plot_confusion_matrix


class Plotter:
    @staticmethod
    def plot_predictions(train_data,
                         train_labels,
                         test_data,
                         test_labels,
                         predictions=None
                         ):
        """
        Plots training data, test data and compares predictions.
        """
        plt.figure(figsize=(10, 7))

        # Plot training data in blue
        plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

        # Plot test data in green
        plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

        # Plot predictions if they exists
        if predictions is not None:
            plt.scatter(test_data, predictions, c="r", s=4, label="Prediction")

        # Legend
        plt.legend(prop={"size": 14})
        plt.show()

    @staticmethod
    def plot_loss_curves(epoch_count,
                         loss_values,
                         test_loss_values
                         ):
        plt.plot(epoch_count, loss_values, label="Train loss")
        plt.plot(epoch_count, test_loss_values, label="Test loss")
        plt.title("Training and test loss curves ")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    @staticmethod
    def plot_decision_boundary(model,
                               train_features,
                               train_labels,
                               test_features,
                               test_labels):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Train")
        plot_decision_boundary(model, train_features, train_labels)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 2)
        plt.title("Test")
        plot_decision_boundary(model, test_features, test_labels)

        plt.show()

    @staticmethod
    def show_image(image, label, title):
        plt.imshow(image, cmap="gray")
        plt.title(label=title + " - " + str(label), fontsize=10)
        plt.show()

    @staticmethod
    def show_image_with_dict(image_dict):
        plt.imshow(image_dict['image'], cmap="gray")
        plt.title(label=
                  image_dict['image_path'] + "\n"
                  + "width:" + str(image_dict['image'].width) + " height:" + str(image_dict['image'].height) + "\n"
                  + " Image shape:" + str(np.array(image_dict['image']).shape))
        plt.show()

    @staticmethod
    def show_random_images(data):
        torch.manual_seed(42)
        fig = plt.figure(figsize=(16, 8))
        rows, cols = 4, 4
        for i in range(1, rows * cols + 1):
            random_idx = torch.randint(0, len(data), size=[1]).item()
            img, lab = data[random_idx]
            fig.add_subplot(rows, cols, i)
            plt.imshow(img.squeeze(), cmap="gray")
            plt.title(data.classes[lab] + " - " + str(lab))
            plt.axis(False)
        plt.show()

    @staticmethod
    def show_batch_images(data, features_batch, labels_batch, preds_batch=None):
        torch.manual_seed(42)
        rows = len(features_batch) // 8
        cols = len(features_batch) // rows
        fig = plt.figure(num="Class name - truth number - pred number", figsize=(16, 8))
        fig.text(10, 10, "Class name - truth class - pred class")
        for i in range(1, len(features_batch) + 1):
            img, lab = features_batch[i - 1], labels_batch[i - 1]
            fig.add_subplot(rows, cols, i)
            plt.imshow(img.squeeze(), cmap="gray")
            color = "black"
            if preds_batch is not None:
                color = "green" if preds_batch[i - 1] == lab else "red"
            plt.title(f"{data.classes[lab].title()} - "
                      f"{labels_batch[i - 1]}"
                      f"{' - ' + str(preds_batch[i - 1].item()) if preds_batch is not None else ''}",
                      fontsize=10,
                      c=color)
            plt.axis(False)
        plt.show()

    @staticmethod
    def show_confusion_matrix(data, preds_batch):
        conf_matrix = ConfusionMatrix(task="multiclass", num_classes=len(data.classes))
        conf_matrix_tensor = conf_matrix(preds=preds_batch, target=data.targets)

        plot_confusion_matrix(conf_mat=conf_matrix_tensor.numpy(),
                              class_names=data.classes,
                              figsize=(10, 7))
        plt.show()

    @staticmethod
    def show_image_from_dict(image_dict):
        image = image_dict['image']
        img_as_array = np.array(image)
        Plotter.show_image(image,
                           image_dict['image_path'].parent.stem,
                           str(image_dict['image_path']) + "\n"
                           + "width:" + str(image.width) + " height:" + str(image.height) + "\n"
                           + " Image shape:" + str(img_as_array.shape))

    @staticmethod
    def plot_loss_accuracy_curves(results: Dict[str, List[float]]):
        train_loss = results["train_loss"]
        test_loss = results["test_loss"]
        train_acc = results["train_acc"]
        test_acc = results["test_acc"]
        epochs = range(len(results["train_loss"]))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss, label="train_loss")
        plt.plot(epochs, test_loss, label="test_loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc, label="train_accuracy")
        plt.plot(epochs, test_acc, label="test_accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()
        plt.show()
