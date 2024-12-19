import sys
from pathlib import Path

from torchvision import datasets, transforms

# 将项目根目录添加到Python路径
root_dir = str(Path(__file__).parent.parent.parent)
sys.path.append(root_dir)

from base_dataset.config import RAW_DATA_DIR


def load_mnist():
    """
    Load MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root=RAW_DATA_DIR,
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = datasets.MNIST(
        root=RAW_DATA_DIR,
        train=False,
        transform=transform,
        download=True
    )

    train_data = train_dataset.data.numpy().reshape(len(train_dataset), -1)
    test_data = test_dataset.data.numpy().reshape(len(test_dataset), -1)

    train_labels = train_dataset.targets.numpy()
    test_labels = test_dataset.targets.numpy()

    return train_data, train_labels, test_data, test_labels
