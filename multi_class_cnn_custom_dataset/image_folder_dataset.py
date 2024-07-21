import pathlib
from typing import Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import find_classes


class ImageFolderDataset(Dataset):
    def __init__(self, target_dir: (str, pathlib.Path), transform=None):
        if isinstance(target_dir, pathlib.Path):
            paths = target_dir
        elif isinstance(target_dir, str):
            paths = pathlib.Path(target_dir)
        else:
            raise TypeError("target_dir type is invalid")

        self.paths = list(paths.glob("*/*.*"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(target_dir)

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        if self.transform is not None:
            image = self.transform(image)
        return image, class_idx

