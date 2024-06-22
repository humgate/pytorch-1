from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from model import CircleModel0, CircleModel1, CircleModel2
from util.plotter import Plotter

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch device set to {device}")

    # 1. Create the data
    n_samples = 1000  # Make 1000 samples
    X, y = make_circles(n_samples, noise=0.03, random_state=42)  # X - features, y - labels

    print(f"First 5 samples of X:\n {X[:5]}")
    print(f"First 5 samples of y:\n {y[:5]}")

    plt.scatter(x=X[:, 0],  # X1 is horizontal coordinate of X feature
                y=X[:, 1],  # X2 is vertical coordinate of X feature
                c=y,  # y is the circle (label) the X dot belongs to
                cmap=plt.cm.RdYlBu)
    plt.show()

    circles = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "label": y})
    print(circles.head(10))

    # Turn data into tensors
    print(type(X))  # numpy.ndarray
    X = torch.from_numpy(X).type(torch.float32)  # explicit setting tensor type to float32, ndarray type is int64
    y = torch.from_numpy(y).type(torch.float32)

    # Split the data into train and test subsets, 80% - train, 20% - test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Model0 instance
    model_0 = CircleModel0().to(device)
    print(next(model_0.parameters()).device)

    # Loss function - binary cross entropy
    # loss_fn = nn.BCELoss  # requires inputs to have gone through the sigmoid activation function
    loss_fn = nn.BCEWithLogitsLoss()  # BCE with built-in sigmoid activation function, expects logits as input

    # Optimizer
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

    # Accuracy = true positive / true positive + true negative
    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        accuracy = (correct / len(y_pred)) * 100
        return accuracy

    # 3. Train model0
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    X_train, y_train = X_train.to(device), y_train.to(device)  # Put the data tensors to target device
    X_test, y_test = X_test.to(device), y_test.to(device)

    epochs = 100  # epochs - amount of complete loops through all the data
    for epoch in range(epochs):
        model_0.train()  # set training mode
        # Forward pass
        # Model outputs are raw logits.
        # We convert these logits into prediction probabilities using sigmoid activation function.
        # Then we convert prediction probabilities to prediction labels by rounding them or argmax() them.
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
        model_0.eval()  # turn off training mode
        with torch.inference_mode():
            test_logits = model_0(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

            if epoch % 10 == 0:
                print(f"Epoch: {epoch} |"
                      f"Loss: {loss:.5f}, Acc: {acc:.2f}% |"
                      f"Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

    # Visualize decision boundary - model_0 is unable to learn
    Plotter.plot_decision_boundary(model_0, X_train, y_train, X_test, y_test)

    # Model1 instance
    model_1 = CircleModel1().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model_1.parameters(), lr=0.01)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Model1 Training
    epochs = 1000
    for epoch in range(epochs):
        model_1.train()
        y_logits = model_1(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Model1 testing
        model_1.eval()
        with torch.inference_mode():
            test_logits = model_1(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

            if epoch % 100 == 0:
                print(f"Epoch: {epoch} |"
                      f"Loss: {loss:.5f}, Acc: {acc:.2f}% |"
                      f"Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

    # Visualize decision boundary - model_1 is unable to learn regardless of additional neurons and layers
    Plotter.plot_decision_boundary(model_1, X_train, y_train, X_test, y_test)

    # Model2 instance
    model_2 = CircleModel2().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    epochs = 10000
    for epoch in range(epochs):
        #  Model2 training
        model_2.train()
        y_logits = model_2(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #  Model2 testing
        model_2.eval()
        with torch.inference_mode():
            test_logits = model_2(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            test_loss = loss_fn(test_logits, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

            if epoch % 1000 == 0:
                print(f"Epoch: {epoch} |"
                      f"Loss: {loss:.5f}, Acc: {acc:.2f}% |"
                      f"Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
    # Visualize decision boundary for model_2 - woo-hoo!
    Plotter.plot_decision_boundary(model_2, X_train, y_train, X_test, y_test)
