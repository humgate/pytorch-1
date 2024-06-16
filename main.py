import torch
import matplotlib.pyplot as plt


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
        plt.scatter(test_data, predictions)

    # Legend
    plt.legend(prop={"size":14})
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
    plot_predictions(X_train, y_train, X_test, y_test, None)
