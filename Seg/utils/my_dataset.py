import os
from PIL import Image
import cv2
from torch.utils.data import Dataset
from .utils import Input_strategy


class NEULoad(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(NEULoad, self).__init__()
        self.flag = "train" if train else "test"
        data_root = os.path.join(root, self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms

        img_names = os.listdir(os.path.join(data_root, "images"))
        mask_names = os.listdir(os.path.join(data_root, "masks"))

        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.roi_mask = [os.path.join(data_root, "masks", i) for i in mask_names]

    def __getitem__(self, idx):
        # img = Image.open(self.img_list[idx]).convert('RGB')
        img = cv2.imread(self.img_list[idx], cv2.IMREAD_GRAYSCALE)
        img = Input_strategy(img, 'GICA')
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        mask = Image.open(self.roi_mask[idx])

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)


class DCLoad(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DCLoad, self).__init__()
        self.flag = "train" if train else "test"
        data_root = os.path.join(root, self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms

        img_names = os.listdir(os.path.join(data_root, "images"))
        mask_names = os.listdir(os.path.join(data_root, "masks"))

        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        self.roi_mask = [os.path.join(data_root, "masks", i) for i in mask_names]

    def __getitem__(self, idx):

        img = cv2.imread(self.img_list[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.roi_mask[idx], cv2.IMREAD_GRAYSCALE)

        img = Input_strategy(img, 'GICA')
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        mask = Image.fromarray(mask//255)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)
