import os
import random
import zipfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from typing import Tuple, Dict, List

from util.plotter import Plotter
from util.wget import Wget


def get_class_names(images_dir) -> Tuple[List[str], Dict[str, int]]:
    class_names = sorted(os.listdir(images_dir))
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
    class_names = [class_name for class_name in class_names if os.path.isdir(os.path.join(images_dir, class_name))]
    return class_names, class_to_idx
