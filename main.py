import torch
import matplotlib.pyplot as plt
from torch import nn

from model import LinearRegressionModel


def print_pytorch_ver():
    print('torch version: ' + torch.__version__)
    print(torch.cuda.current_device())


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
        plt.scatter(test_data, predictions, c="r")

    # Legend
    plt.legend(prop={"size": 14})
    plt.show()


if __name__ == '__main__':
    # Known parameters
    weight = 0.7
    bias = 0.3

    start = 0
    end = 1
    step = 0.02

    # Inputs data
    X = torch.arange(start, end, step).unsqueeze(dim=1)  # add dimension
    # Outputs data
    y = weight * X + bias

    # Splitting data into sets. Training and Testing sets
    train_split = int(0.8 * len(X))
    X_train, y_train = X[:train_split], y[:train_split]
    X_test, y_test = X[train_split:], y[train_split:]
    # print(len(X_train), len(y_train), len(X_test), len(y_test))  # 40 40 10 10
    # plot_predictions(X_train, y_train, X_test, y_test, None)

    # Manual seed
    torch.manual_seed(42)

    # Instance of the model
    model_0 = LinearRegressionModel()
    print(model_0.state_dict())  # model parameters

    # Loss function - mean absolute error
    loss_fn = nn.L1Loss()

    # Optimizer - adjusts parameters based on loss got on each step to minimize loss - stochastic gradient descent
    optimizer = torch.optim.SGD(params=model_0.parameters(),
                                lr=0.01)  # learning rate == step size - how big are optimizer parameters changes

    # Training loop.
    # 0. Loop thought the data.
    # 1. Forward pass or forward propagation (data moving through the model's `forward` functions)
    # 2. Calculate the loss (compare forward pass predictions to truth labels)
    # 3. Optimizer zero grad
    # 4. Loss backwards propagation - move backwards through the network to calculate the gradients of each parameter
    # with respect to the loss
    # 5. Optimizer step - adjust parameters to reduce loss - gradient descent

    # Training
    epochs = 300  # An epoch is one single loop through the data
    for epoch in range(epochs):  # 0. Loop through the data
        model_0.train()  # gradient tracking on

        # 1. Forward pass
        y_pred = model_0(X_train)
        #  0. Loop through the data
        # 2. Calculate loss
        loss = loss_fn(y_pred, y_train)

        # 3. Zero optimizer gradients (they accumulate by default )
        optimizer.zero_grad()

        # 4. Perform back propagation on the loss with respect to parameters
        loss.backward()

        # 5. Step the optimizer
        optimizer.step()

        #  Testing
        model_0.eval()  # gradient tracking off

        print(model_0.state_dict())

    # Make predictions with model
    with torch.inference_mode():
       y_preds = model_0(X_test)

    print(y_preds, y_test)
    plot_predictions(X_train, y_train, X_test, y_test, predictions=y_preds)
