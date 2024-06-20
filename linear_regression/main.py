import numpy as np
import torch
from torch import nn

from model import LinearRegressionModel
from plotter import Plotter

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
    print(len(X_train), len(y_train), len(X_test), len(y_test))  # 40 40 10 10

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

    # Interesting data to collect and review about the model training process
    epoch_count = []
    loss_values = []
    test_loss_values = []

    # Training
    epochs = 200  # An epoch is one single loop through the data
    # 0. Loop through the data
    for epoch in range(epochs):
        model_0.train()  # gradient tracking on

        # 1. Forward pass or forward propagation (data moving through the model's `forward` functions)
        y_pred = model_0(X_train)

        # 2. Calculate the loss (compare forward pass predictions to ground truth labels)
        loss = loss_fn(y_pred, y_train)

        # 3. Zero optimizer gradients (they accumulate by default)
        optimizer.zero_grad()

        # 4. Backwards propagation on the loss with respect to parameters (calculate the gradients of each parameter
        # with respect to the loss
        loss.backward()

        # 5. Step the optimizer in gradient descent
        optimizer.step()

        #  Testing
        model_0.eval()  # off the model's settings not needed for testing
        with torch.inference_mode():  # off gradient tracking
            test_preds = model_0(X_test)
            test_loss = loss_fn(test_preds, y_test)
        if epoch % 10 == 0:
            epoch_count.append(epoch)
            loss_values.append(loss)
            test_loss_values.append(test_loss)
            print(f"Epoch: {epoch} | Test: {loss} | Test loss: {test_loss}")

    Plotter.plot_predictions(X_train, y_train, X_test, y_test, predictions=test_preds)
    Plotter.plot_loss_curves(epoch_count,
                             np.array(torch.tensor(loss_values).numpy()),
                             np.array(torch.tensor(test_loss_values).numpy()))
