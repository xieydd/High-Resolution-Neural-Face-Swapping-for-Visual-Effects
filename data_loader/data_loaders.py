from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
        
        
class ImageNetDataLoader(BaseDataLoader):
    """
    ImageNet data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),##
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.data_dir = data_dir
        self.dataset = datasets.ImageFolder(self.data_dir, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
        
class Cifar10DataLoader(BaseDataLoader):
    """
    ImageNet data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=2, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
        
class Cifar10DataLoader_Eff(BaseDataLoader):
    """
    ImageNet data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=2, training=True):
        trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR10(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
        
        
class Cifar100DataLoader(BaseDataLoader):
    """
    ImageNet data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=2, training=True):
        trsfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[x / 255.0 for x in[0.507, 0.487, 0.441]],
                                     std=[x / 255.0 for x in [0.267, 0.256, 0.276]])
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR100(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
        
        
        
class Cifar100DataLoader_Eff(BaseDataLoader):
    """
    ImageNet data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.1, num_workers=2, training=True):
        trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.data_dir = data_dir
        self.dataset = datasets.CIFAR100(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
