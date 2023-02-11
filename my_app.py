import torch
import torchvision
import torchvision.transforms as transforms


def get_data(path, train: bool = True):
    transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels = 1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]
        )
    dataset = torchvision.datasets.CIFAR100(
        root=path,
        train=train,
        download=True,
        transform=transform
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
        )
    return dataloader

train_loader = get_data("./datasets/")
test_loader = get_data("./datasets/", train=False)