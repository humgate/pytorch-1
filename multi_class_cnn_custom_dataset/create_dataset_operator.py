import os
import zipfile
from pathlib import Path

import torch

from util.plotter import Plotter
from util.wget import Wget


def create_custom_dataset():
    """
    The data we're going to be using is a subset of the Food101 dataset.
    Subset contains 3 classes of food and 1000 images pres class (750 training 250 testing).
    Food101 is popular computer vision benchmark as it contains 1000 images of 101 different kinds of foods,
    totaling 101,000 images (75,750 train and 25,250 test).
    """
    # Download and unzip images
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"
    file_path = data_path / "pizza_steak_sushi.zip"
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
    Plotter.show_random_image(image_path)
