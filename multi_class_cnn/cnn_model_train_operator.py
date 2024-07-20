import random

import torchvision
from torchvision import datasets

from util.model_functions import *
from util.timer import Timer
from .model import *


def multi_class_train_cnn():
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    # 2. Prepare DataLoader
    BATCH_SIZE = 32
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)  # remove images order

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False)

    # 3. Instantiate CNN model2
    model_2 = FashionMNISTCNNModel0(input_shape=1, hidden_units=10, output_shape=len(train_data.classes)).to(device)
    save_model(model_2, "cnn_fashion_mnist.pth")
    print(model_2)
    '''
    FashionMNISTCNNModel0(
      (conf_block_1): Sequential(
        (0): Conv2d(1, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conf_block_2): Sequential(
        (0): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): Conv2d(10, 10, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU()
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (output_block): Sequential(
        (0): Flatten(start_dim=1, end_dim=-1)
        (1): Linear(in_features=490, out_features=10, bias=True)
      )
    )
    '''

    # Convolution layer creates out_channels = 10 images of the same or smaller (depending on stride & padding)
    # size where each image focuses on its own feature ("channel") from original image
    color_image = torch.randn(size=(32, 3, 64, 64))[0]
    print(color_image.shape)  # torch.Size([3, 64, 64])
    conv_layer = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3), stride=1, padding=0)
    image_after_conv_layer = conv_layer.forward(color_image)
    print(image_after_conv_layer.shape)  # torch.Size([10, 62, 62])

    # MaxPool creates compressed image kernel_size=4 times smaller than input image
    # where each pixel is max from each 4 X 4 (non overlapping) region of input image
    max_pool_layer = nn.MaxPool2d(kernel_size=4)
    image_after_max_pool_layer = max_pool_layer(image_after_conv_layer)
    print(image_after_max_pool_layer.shape)  # torch.Size([10, 15, 15])

    # 4. Validate CNN model layers hyperparameters
    # image, label = train_data[0]  # get 1 input instance of x for batched model training
    # print(image.shape)  # torch.Size([1, 28, 28])
    # image = image.unsqueeze(dim=0)  # x in batched training will have +1 dim for batch, so add it
    # print(image.shape)  # torch.Size([1, 1, 28, 28])
    # model_2.eval()
    # with torch.inference_mode():
    #     y = model_2(image.to(device))
    # print(y.shape)  # model output shape = torch.Size([1, 10]) with in_features = 10*49 on last linear layer

    # 5. loss, optimizer, evaluation metric, timer
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.1)
    accuracy_fn = torchmetrics.Accuracy("multiclass", num_classes=10).to(device)
    timer = Timer()

    # 6. Train CNN model_2 on batches
    timer.start_timer()
    epochs = 3
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}")
        train_on_batches(model_2, train_dataloader, loss_fn, optimizer, accuracy_fn, torch.device(device))
        test_on_batches(model_2, test_dataloader, loss_fn, accuracy_fn, torch.device(device))
    timer.stop_timer()
    timer.print_elapsed_time()
    save_model(model_2, "cnn_fashion_mnist.pth")

    # 7. Get cnn model test results
    print(eval_model(model_2, test_dataloader, loss_fn, accuracy_fn, torch.device(device)))
