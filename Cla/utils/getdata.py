from torchvision import datasets, transforms
from torchvision.transforms import autoaugment
from .dataset import ImageFolderCustom


def GetData(args):

    if args.dataset.lower() == 'mini-imagenet100':
        transform_trains = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            autoaugment.RandAugment(magnitude=15),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_vals = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        train_sets = datasets.ImageFolder(root=args.data_dir+"train", transform=transform_trains)
        val_sets = datasets.ImageFolder(root=args.data_dir+"val", transform=transform_vals)

    elif args.dataset.lower() in ['ssd-10', 'neu-cls']:
        transform_trains = transforms.Compose([
            # transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            autoaugment.RandAugment(magnitude=15),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.18, 0.18, 0.18)),
        ])
        transform_vals = transforms.Compose([
            # transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.18, 0.18, 0.18)),
        ])
        train_sets = ImageFolderCustom(root=args.data_dir+"train", transform=transform_trains)
        val_sets = ImageFolderCustom(root=args.data_dir+"val", transform=transform_vals)

    else:
        raise NotImplementedError()

    return train_sets, val_sets
