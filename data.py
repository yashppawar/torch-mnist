from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets 

def load_train_data(batch_size=32, shuffle=True):
    """
    Download the dataset and load it in DataLoader and return them
    """
    training_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
    )

    training_data = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=shuffle
    )     

    return training_data


def load_test_data(batch_size=32):
    """
    Download the dataset and load it in DataLoader and return them
    """
    test_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = DataLoader(
        test_data,
        batch_size=batch_size,
    )     

    return test_data


if __name__ == '__main__':
    load_train_data()
    load_test_data()
