from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import requests
from pathlib import Path
from helper_functions import plot_predictions, plot_decision_boundary
from model import CircleModel0

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(F"torch device set to {device}")

    # if Path("helper_functions.py").is_file():
    #     print("helper_functions.py already exists, skipping download")
    # else:
    #     print("Downloading helper_functions.py")
    #     request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main"
    #                            "/helper_functions.py")
    #     with open("helper_functions.py", "wb") as f:
    #         f.write(request.content)

    # 1. Create the data
    n_samples = 1000  # Make 1000 samples
    #  X - features, y - labels
    X, y = make_circles(n_samples, noise=0.03, random_state=42)

    print(f"First 5 samples of X:\n {X[:5]}")
    print(f"First 5 samples of y:\n {y[:5]}")

    # plt.scatter(x=X[:, 0],  # X1 is horizontal coordinate of X feature
    #             y=X[:, 1],  # X2 is vertical coordinate of X feature
    #             c=y,  # y is the circle (label) the X dot belongs to
    #             cmap=plt.cm.RdYlBu)
    # plt.show()

    circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})
    print(circles.head(10))

    #  Turn data into tensors
    print(type(X))  # numpy.ndarray
    X = torch.from_numpy(X).type(torch.float32)  # explicit setting tensor type to float32, ndarray type is int64
    y = torch.from_numpy(y).type(torch.float32)

    #  Split the data into train and test subsets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Model instance
    model_0 = CircleModel0().to(device)
    print(next(model_0.parameters()).device)

    # Loss function - binary cross entropy
    #  loss_fn = nn.BCELoss  # requires inputs to have gone through the sigmoid activation function
    loss_fn = nn.BCEWithLogitsLoss()  # BCE with built-in sigmoid activation function, expects logits as input

    # Optimizer
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

    # Accuracy calc - true positive / true positive + true negative
    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct / len(y_pred)) * 100
        return acc

    # Train model.
    # Model outputs are raw logits.
    # We convert these logits into prediction probabilities using activation function (sigmoid)
    # Then we convert prediction probabilities to prediction labels by rounding them or argmax() them

    # View the firs 5 outputs of the forward pass on the test data
    # model_0.eval()
    # with torch.inference_mode():
    #     y_logits = model_0(X_test.to(device)[:5])
    #     print(y_logits)
    #
    #     # Convert logits with sigmoid activation function to prediction probabilities
    #     y_pred_probs = torch.sigmoid(y_logits)
    #     print(y_pred_probs)
    #
    #     # Convert probabilities to labels
    #     y_pred = torch.round(y_pred_probs)  # y_pred_probs >= 0.5 ( y=1 class1) else (y=0 class2)
    #     print(y_pred)
    #
    #     # In full, all at once
    #     y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))
    #     print(y_pred_labels)
    #     print(torch.eq(y_pred.squeeze(), y_pred_labels.squeeze()).squeeze())


    # with torch.inference_mode():
    #     untrained_preds = model_0(X_test.to(device))
    #     print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
    #     print(f"\nFirst 10 labels:\n{y_test[:10]}")

    # Training and evaluation loop
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    epochs = 100  # epochs - amount of complete loops through all the data
    # Put the data tensors to target device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    for epoch in range(epochs):
        model_0.train()  # set training mode
        # Forward pass
        y_logits = model_0(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        # Calculate loss and accuracy
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward propagation
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Testing
        model_0.eval()
        with torch.inference_mode():
            test_logits = model_0(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

            if epoch % 10 == 0:
                print(f"Epoch: {epoch} |"
                      f"Loss: {loss:.5f}, Acc: {acc:.2f}% |"
                      f"Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("Train")
    plot_decision_boundary(model_0, X_train, y_train)
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,2)
    plt.title("Test")
    plot_decision_boundary(model_0, X_test, y_test)
    plt.show()

