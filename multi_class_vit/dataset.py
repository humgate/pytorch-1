import random
import zipfile
from pathlib import Path
from typing import Tuple

import torch
import torchvision.datasets
from PIL import Image
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from util.wget import Wget


def get_random_image(image_path):
    image_path_list = list(image_path.glob("*/*/*.jpg"))
    random_image_path = random.choice(image_path_list)
    image = Image.open(random_image_path)
    image_dict = {'image': image, 'image_path': random_image_path}
    return image_dict


def read_image_to_device(image_path, resized_image_size: int, device):
    image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    image = image / 255.
    image_tensor = get_data_transform(resized_image_size)(image)
    image_tensor = torch.unsqueeze(image_tensor, dim=0)
    return image_tensor.to(device)


def get_data_transform(resized_image_size: int):
    data_transform = transforms.Compose([
        transforms.Resize(resized_image_size),
        transforms.CenterCrop(resized_image_size),
        transforms.ToTensor(),
    ])
    return data_transform


def load_dataset_vit(images_path: str,
                     download_url: str,
                     resized_image_size: int,
                     patch_size: int) -> Tuple[ImageFolder, ImageFolder]:
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

    # Create datasets with ImageFolder
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    train_data = datasets.ImageFolder(root=str(train_dir),
                                      transform=get_data_transform(resized_image_size=resized_image_size),
                                      target_transform=None)  # transform for the labels/targets
    test_data = datasets.ImageFolder(root=str(test_dir),
                                     transform=get_data_transform(resized_image_size=resized_image_size),
                                     target_transform=None)

    return train_data, test_data
