import os
import random
import zipfile
from pathlib import Path
from typing import Tuple

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from util.plotter import Plotter
from .image_folder_dataset import ImageFolderDataset
from util.wget import Wget
from util.file_utils import get_class_names


def get_random_image(image_path):
    image_path_list = list(image_path.glob("*/*/*.jpg"))
    random_image_path = random.choice(image_path_list)
    image = Image.open(random_image_path)
    image_dict = {'image': image, 'image_path': random_image_path}
    return image_dict


def load_dataset(images_path: str, download_url: str, batch_size=32) -> Tuple[DataLoader, DataLoader]:
    """
    The data we're going to be using is a subset of the Food101 dataset.
    Subset contains 3 classes of food and 1000 images pres class (750 training 250 testing).
    Food101 is popular computer vision benchmark as it contains 1000 images of 101 different kinds of foods,
    totaling 101,000 images (75,750 train and 25,250 test).
    """
    # Download and unzip images
    data_path = Path(images_path).parent
    image_path = Path(images_path)
    file_path = data_path / Path(download_url).name
    if not image_path.is_dir():
        print(f"Creating {image_path}...")
        image_path.mkdir(parents=True, exist_ok=True)
        Wget.get_file(
            file_path,
            "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            print("Unzipping pizza, steak, sushi images...")
            zip_ref.extractall(image_path)

    # Show dir content Data. Dir names will be classnames
    for dirpath, dirnames, filenames in os.walk(image_path):
        print(f"There are {len(dirnames)} dirs and {len(filenames)} images in {dirpath}")
    # Show random image with attributes
    # Plotter.show_image_from_dict(get_random_image(image_path))

    # Will transform our images to tensors
    data_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.TrivialAugmentWide(num_magnitude_bins=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    # Check transformed image
    image = get_random_image(image_path)
    image_tensor = data_transform(image['image'])
    Plotter.show_image(image_tensor.permute(1, 2, 0), image['image_path'], str(image['image_path'].parent.stem))

    # Create datasets with ImageFolder
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    train_data = datasets.ImageFolder(root=str(train_dir),
                                      transform=data_transform,  # transform for the data
                                      target_transform=None)  # transform for the labels/targets
    test_data = datasets.ImageFolder(root=str(test_dir),
                                     transform=data_transform,
                                     target_transform=None)
    # print(train_data.classes)  # ['pizza', 'steak', 'sushi']
    # print(train_data.class_to_idx)  # {'pizza': 0, 'steak': 1, 'sushi': 2}
    # print(train_data.samples[0])  # ('data/pizza_steak_sushi/train/pizza/1008844.jpg', 0)
    # print(train_data[0][0].shape)  # torch.Size([3, 64, 64])
    # Plotter.show_image(train_data[0][0].permute(1, 2, 0), train_data.classes[0], str(train_data[0][1]))

    # The same using custom dataset
    train_data_custom = ImageFolderDataset(target_dir=train_dir,
                                           transform=data_transform)
    test_data_custom = ImageFolderDataset(target_dir=test_dir,
                                          transform=data_transform)
    # Plotter.show_image(train_data_custom[0][0].permute(1, 2, 0), train_data.classes[0], str(train_data[0][1]))

    # Turn datasets in Dataloaders
    train_dataloader = DataLoader(dataset=train_data_custom,
                                  batch_size=batch_size,
                                  num_workers=os.cpu_count(),
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 num_workers=os.cpu_count(),
                                 shuffle=True)

    return train_dataloader, test_dataloader



