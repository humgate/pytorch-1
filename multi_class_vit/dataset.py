import random
import zipfile
from pathlib import Path
from typing import Tuple

import torch
import torchvision.datasets
from PIL import Image
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder, Food101

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
        transforms.Resize((resized_image_size, resized_image_size)),
        # transforms.CenterCrop(resized_image_size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return data_transform


def load_small_subset_dataset(images_path: str,
                              download_url: str,
                              resized_image_size: int) -> Tuple[ImageFolder, ImageFolder]:
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


def load_full_dataset_for_selected_labels(images_path: str, resized_image_size: int, labels: []):
    data_path = Path(images_path)
    train_data = Food101Subset(root=data_path,
                               split="train",
                               download=True,
                               labels=labels,
                               transform=get_data_transform(resized_image_size=resized_image_size))
    test_data = Food101Subset(root=data_path,
                              split="test",
                              download=True,
                              labels=labels,
                              transform=get_data_transform(resized_image_size=resized_image_size))

    return train_data, test_data


# Subclass Food101 to make it load all its images but only for specified labels
class Food101Subset(Food101):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False, labels=None):
        super(Food101Subset, self).__init__(root, split, transform, target_transform, download)

        if labels is not None:
            indices = [i for i in range(len(self._labels)) if self.classes[self._labels[i]] in labels]

            filtered_labels = []
            filtered_image_files = []
            filtered_classes = []

            for i in indices:
                label_i = self.get_new_label(self.classes[self._labels[i]], labels)
                filtered_labels.append(label_i)
                filtered_image_files.append(self._image_files[i])
                if label_i < len(self.classes):
                    filtered_classes.append(self.classes[self._labels[i]])

            self._labels = filtered_labels
            self._image_files = filtered_image_files
            self.classes = filtered_classes
            self.class_to_idx = {j: i for i, j in enumerate(self.classes)}

    @staticmethod
    def get_new_label(class_name, labels):
        for i in range(len(labels)):
            if labels[i] == class_name:
                return i
