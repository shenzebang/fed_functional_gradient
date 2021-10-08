import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset


import torch
import numpy as np


class MyDataset(Dataset):
    def __init__(self, data, target, train, transform=None):
        self.data = data
        self.target = target
        self.train = train
        self.transform = transform
        assert data.shape[0] == target.shape[0]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        sample = self.data[item]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, self.target[item]

def make_dataset(data, target, train, transform=None):
    return MyDataset(data, target, train, transform)


def load_dataset(dataset):
    if dataset == "cifar10":
        dataset_train = datasets.CIFAR10(root='datasets/' + dataset, download=True)
        # dataset_train.data = torch.as_tensor(dataset_train.data).permute(0, 3, 1, 2)
        dataset_train.targets = torch.as_tensor(np.array(dataset_train.targets))
        dataset_test = datasets.CIFAR10(root='datasets/' + dataset, train=False, download=True)
        # dataset_test.data = torch.as_tensor(dataset_test.data).permute(0, 3, 1, 2)
        dataset_test.targets = torch.as_tensor(np.array(dataset_test.targets))
        n_classes = 10
        n_channels = 3
    elif dataset == "cifar100":
        dataset_train = datasets.CIFAR100(root='datasets/' + dataset, download=True)
        # dataset_train.data = torch.as_tensor(dataset_train.data).permute(0, 3, 1, 2)
        dataset_train.targets = torch.as_tensor(np.array(dataset_train.targets))
        dataset_test = datasets.CIFAR100(root='datasets/' + dataset, train=False, download=True)
        # dataset_test.data = torch.as_tensor(dataset_test.data).permute(0, 3, 1, 2)
        dataset_test.targets = torch.as_tensor(np.array(dataset_test.targets))
        n_classes = 100
        n_channels = 3
    else:
        raise NotImplementedError

    return dataset_train, dataset_test, n_classes, n_channels

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def make_transforms(dataset, train=True, no_data_augmentation=False, is_distill=False):
    if dataset == "cifar10" or dataset == "cifar100":
        if train:
            if not no_data_augmentation:
                if not is_distill:
                    transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
                        normalize,
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
                        normalize,
                    ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    else:
        raise NotImplementedError

    return transform


def make_dataloader(args, type, dataset: MyDataset):
    if type == "train":
        dataloader = DataLoader(dataset, batch_size=args.local_dataloader_batch_size, shuffle=True, num_workers=0)
    elif type == "test":
        dataloader = DataLoader(dataset, batch_size=args.test_dataloader_batch_size, shuffle=False, num_workers=4)
    elif type == "distill":
        dataloader = DataLoader(dataset, batch_size=args.distill_dataloader_batch_size, shuffle=True, num_workers=4)

    return dataloader


def split_dataset(n_workers, homo_ratio, dataset: VisionDataset, transform=None):
    data = dataset.data
    label = dataset.targets

    # centralized case, no need to split
    if n_workers == 1:
        return [make_dataset(data, label, dataset.train, transform)]

    homo_ratio = homo_ratio
    n_workers = n_workers

    n_data = data.shape[0]

    n_homo_data = int(n_data * homo_ratio)

    n_homo_data = n_homo_data - n_homo_data % n_workers
    n_data = n_data - n_data % n_workers

    if n_homo_data > 0:
        data_homo, label_homo = data[0:n_homo_data], label[0:n_homo_data]
        data_homo_list, label_homo_list = np.split(data_homo, n_workers), label_homo.chunk(n_workers)

    if n_homo_data < n_data:
        data_hetero, label_hetero = data[n_homo_data:n_data], label[n_homo_data:n_data]
        label_hetero_sorted, index = torch.sort(label_hetero)
        data_hetero_sorted = data_hetero[index]

        data_hetero_list, label_hetero_list = np.split(data_hetero_sorted, n_workers), label_hetero_sorted.chunk(
            n_workers)

    if 0 < n_homo_data < n_data:
        data_list = [np.concatenate([data_homo, data_hetero], axis=0) for data_homo, data_hetero in
                     zip(data_homo_list, data_hetero_list)]
        label_list = [torch.cat([label_homo, label_hetero], dim=0) for label_homo, label_hetero in
                      zip(label_homo_list, label_hetero_list)]
    elif n_homo_data < n_data:
        data_list = data_hetero_list
        label_list = label_hetero_list
    else:
        data_list = data_homo_list
        label_list = label_homo_list

    return [make_dataset(_data, _label, dataset.train, transform) for _data, _label in zip(data_list, label_list)]
