# coding: utf-8
from PIL import Image
import cv2
from torch.utils.data import Dataset
import pathlib
from typing import Tuple, Dict, List
import os


# make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """ Find the class folder names in a target directory.
    Assumes target directory is in standard image classification format.
    Returns:
    Tuple(list_of_class_names, dict(class_name: index))
    """
    # 1, get the class names by scanning the directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    # 2, raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
    # 3, create a directory of index labels
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def Identity(img):
    result = cv2.merge([img, img, img])
    return result

class ImageFolderCustom(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.paths = list(pathlib.Path(root).glob("*/*"))  # get all image paths
        self.transform = transform
        self.target_transform = target_transform
        self.classes, self.class_to_idx = find_classes(root)

    def load_image(self, index):
        image_path = self.paths[index]
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # GRAY
        return img

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = self.load_image(index)
        img = Identity(img)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            class_idx = self.target_transform(class_idx)

        return img, class_idx

