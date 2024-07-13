from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch import nn
from model import BlobMultiClass0
from util.helper_functions import accuracy_fn
from util.plotter import Plotter
from torchmetrics import Accuracy

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch device set to {device}")

    # Create data
    NUM_CLASSES = 4
    NUM_FEATURES = 2
    RANDOM_SEED = 42

    X_blob, y_blob = make_blobs(n_samples=1000,
                                n_features=NUM_FEATURES,
                                centers=NUM_CLASSES,
                                cluster_std=1.5,
                                random_state=RANDOM_SEED)

    # Turn data into tensors
    X_blob = torch.from_numpy(X_blob).type(torch.float)
    y_blob = torch.from_numpy(y_blob).type(torch.long)

    # Split into train and test
    X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob,
                                                                            y_blob,
                                                                            test_size=0.2,
                                                                            random_state=RANDOM_SEED)
    # Put the data tensors to target device
    X_blob_train, y_blob_train = X_blob_train.to(device), y_blob_train.to(device)
    X_blob_test, y_blob_test = X_blob_test.to(device), y_blob_test.to(device)

    # Visualize the data we just created
    plt.figure(figsize=(10, 7))
    plt.scatter(X_blob[:, 0],
                X_blob[:, 1],
                c=y_blob,  #
                cmap=plt.cm.RdYlBu)
    plt.show()
    print(X_blob_train.shape, y_blob_train[:5])

    # Instantiate blob multi-class model and send it to device
    model_4 = BlobMultiClass0(in_features=2, out_features=4, hidden_units=8).to(device)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.SGD(params=model_4.parameters(), lr=0.1)

    # Raw model_4 output (logits) ->
    # Pred probabilities (using torch.softmax on logits) ->
    # Pred labels (using argmax on pred probabilities) ->
    # Predictions
    model_4.eval()
    with torch.inference_mode():
        y_logits = model_4(X_blob_test)
        y_preds_probs = torch.softmax(y_logits, dim=1)
        # sum of 4 probabilities for each y_preds_probs = 1; y_preds_probs[0] == 1
        print(torch.sum(y_preds_probs[0]))  # tensor(1., device='cuda:0')
        print(y_preds_probs.shape)  # torch.Size([200, 4])
        # max from 4 probabilities is the pred result, we get it using argmax()
        y_preds = torch.argmax(y_preds_probs, dim=1)
        print(y_preds.shape)  # torch.Size([200])
        print(y_preds[:10])  # tensor([1, 1, 3, 1, 1, 1, 3, 1, 3, 1], device='cuda:0')

    # Model_4 training
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    epochs = 100
    for epoch in range(epochs):
        model_4.train()
        y_logits = model_4(X_blob_train)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # description in comments above

        loss = loss_fn(y_logits, y_blob_train)
        acc = accuracy_fn(y_true=y_blob_train, y_pred=y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Testing
        model_4.eval()
        with torch.inference_mode():
            test_logits = model_4(X_blob_test)
            test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)
            test_loss = loss_fn(test_logits, y_blob_test)
            test_acc = accuracy_fn(y_true=y_blob_test, y_pred=test_preds)
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} |"
                      f"Loss: {loss:.5f}, Acc: {acc:.2f}% |"
                      f"Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

    # Visualize decision boundary for model_4 - woo-hoo!
    Plotter.plot_decision_boundary(model_4, X_blob_train, y_blob_train, X_blob_test, y_blob_test)

    # Additional metrics:
    # Accuracy - out of 100 samples, how many dos our model get right? Good for balanced classes amounts
    # Precision
    # Recall
    # F1 - score
    # Classification report

    torchmetric_accuracy = Accuracy("multiclass", num_classes=4).to(device)
    print(torchmetric_accuracy(test_preds, y_blob_test))  # tensor(0.9950, device='cuda:0')


