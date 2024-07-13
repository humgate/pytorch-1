import torch
import torchmetrics
from torch import nn
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
from util.plotter import Plotter
from model import FashionMNISTModel0
from util.timer import Timer
from tqdm import tqdm

if __name__ == '__main__':
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"torch device set to {device}, "
          f"torch version {torch.__version__}, "
          f"torchvision version {torchvision.__version__}")

    # 1. Getting a dataset (FashionMNIST) from torchvision.datasets
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

    print(f"train data length: {len(train_data)}, test data length: {len(test_data)}")  # train: 60000, test: 10000
    print(f"class_names_idx = {train_data.class_to_idx}")  # 'T-shirt/top': 0, 'Trouser': 1, 'Pullover': 2, ...

    # Check the images
    image, label = train_data[0]
    print(f"image.shape = {image.shape}")  # torch.Size([1, 28, 28]) - color channels, width, height
    print(f"image class = {label}")  # T-Shirt/Top
    print(f"image label = {train_data.classes[0].title()}")  # 9
    # Plotter.show_image(image.squeeze(), label, train_data.classes[0].title())
    # Plotter.show_random_images(train_data, 8)

    # 2. Prepare DataLoader
    # train_data & test_data are in the form of PyTorch Datasets (60000 and 10000)
    # Dataloader turns dataset to a Python iterable, specifically into batches. Will use 32 as batch size
    # Because
    # 1) 32 is more computationally efficient than 60000,
    # 2) it gives more chances to update gradients per epoch on training

    BATCH_SIZE = 32

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)  # remove images order

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)
    print(f"train DataLoader: {len(train_dataloader)} batches of {BATCH_SIZE} each")  # 1875 batches of 32 each
    print(f"train DataLoader: {len(test_dataloader)} batches of {BATCH_SIZE} each")  # 313 batches of 32 each

    train_features_batch, train_labels_batch = next(iter(train_dataloader))
    print(f"train_features_batch.shape= {train_features_batch.shape}")  # torch.Size([32, 1, 28, 28])
    print(f"train_labels_batch.shape= {train_labels_batch.shape}")  # torch.Size([32])
    print(len(train_features_batch))  # 32
    # Plotter.show_batch_images(train_data,
    #                           train_features_batch,
    #                           train_labels_batch,
    #                           )

    # 3. Instantiate model0, loss function, optimizer, evaluation metric
    model_0 = FashionMNISTModel0(input_shape=784, hidden_units=10, output_shape=len(train_data.classes)).to(device)
    print(model_0)
    # FashionMNISTModel0(
    #   (linear_layer_stack): Sequential(
    #     (0): Flatten(start_dim=1, end_dim=-1)
    #     (1): Linear(in_features=784, out_features=10, bias=True)
    #     (2): Linear(in_features=10, out_features=10, bias=True)
    #   )
    # )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)
    accuracy_fn = torchmetrics.Accuracy("multiclass", num_classes=10).to(device)

    # 4. Train model_0 on batches
    torch.manual_seed(42)
    timer_1 = Timer()
    timer_1.start_timer()

    epochs = 3
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}")
        train_loss = 0

        # train
        for batch, batch_data in enumerate(train_dataloader):
            (X, y) = batch_data  # the same as X = batch_data[0] y = batch_data[1]
            model_0.train()
            y_pred = model_0(X)  # Forward pass
            loss = loss_fn(y_pred, y)  # Calc loss per batch
            train_loss += loss  # accumulate loss with loss from previous batches
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # updating model parameters once per batch
            if batch % 100 == 0:
                print(f"Batch {batch} - Looked at  {batch * BATCH_SIZE}/{len(train_data)} samples")
        train_loss /= len(train_dataloader)  # average loss value across all batches in dataloader

        # test
        test_loss, test_acc = 0, 0
        model_0.eval()
        with torch.inference_mode():
            for X_test, y_test in test_dataloader:
                test_pred = model_0(X_test)
                test_loss += loss_fn(test_pred, y_test)
                test_acc += accuracy_fn(test_pred.argmax(dim=1), y_test)
            test_loss /= len(test_dataloader)
            test_acc /= len(test_dataloader)
        print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f} |  Test acc: {test_acc:.4f}")

    timer_1.stop_timer()
    timer_1.print_elapsed_time()
