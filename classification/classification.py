from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    # Create the data
    n_samples = 1000  # Make 1000 samples
    #  X - features, y - labels
    X, y = make_circles(n_samples, noise=0.03, random_state=42)
    print(len(X), len(y))
    print(f"First 5 samples of X:\n {X[:5]}")
    print(f"First 5 samples of y:\n {y[:5]}")

    plt.scatter(x=X[:, 0],  # X1 is horizontal coordinate
                y=X[:, 1],  # X2 is vertical coordinate
                c=y,  # is what circle is the dot from
                cmap=plt.cm.RdYlBu)
    plt.show()

    circles = pd.DataFrame({"X1": X[:, 0],  # X1 is horizontal coordinate
                            "X2": X[:, 1],  # X2 is vertical coordinate
                            "label": y})  # is what circle is the dot from
    print(circles.head(10))

    #  Turn data into tensors
    print(type(X))  # numpy.ndarray
    X = torch.from_numpy(X).type(torch.float32)
    y = torch.from_numpy(X).type(torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


