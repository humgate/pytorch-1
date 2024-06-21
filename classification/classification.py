from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

from model import CircleModel0

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(F"torch device set to {device}")

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
    loss_fn = nn.BCEWithLogitsLoss  # BCE with built-in sigmoid activation function

    # Optimizer
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

    # Accuracy calc - true positive / true positive + true negative
    def accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        acc = (correct/ len(y_pred)) * 100
        return acc

    # Train model.
    # Model outputs are raw logits.
    # We convert these logits into prediction probabilities using activation function (sigmoid)
    # Then we convert prediction probabilities to prediction labels by rounding them or argmax() them
    torch.manual_seed(42)
    model_0.eval()
    with torch.inference_mode():
        y_logits = model_0(X_test.to(device)[:5])
        print(y_logits)

        # Convert logits with sigmoid
        y_pred_probs = torch.sigmoid(y_logits)
        print(y_pred_probs)

        # Convert probabilities to lables
        y_pred = torch.round(y_pred_probs)  # y_pred_probs >= 0.5 (class1) else (class2)
        print(y_pred)

        y_pred_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))
        print(y_pred_labels)
        print(torch.eq(y_pred.squeeze(), y_pred_labels.squeeze()).squeeze())


    # with torch.inference_mode():
    #     untrained_preds = model_0(X_test.to(device))
    #     print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
    #     print(f"\nFirst 10 labels:\n{y_test[:10]}")








