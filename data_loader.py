import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def load_images(filepath):
    with open(filepath, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')  # Magic number
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_labels(filepath):
    with open(filepath, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')  # Magic number
        num_labels = int.from_bytes(f.read(4), 'big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_mnist_data_loaders(train_images_path, train_labels_path, test_images_path, test_labels_path, batch_size=64, transform=None): 
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0,5,), (0.5,))
    ])
    train_images = load_images(train_images_path)
    train_labels = load_labels(train_labels_path)
    test_images = load_images(test_images_path)
    test_labels = load_labels(test_labels_path)

    train_dataset = MNISTDataset(train_images, train_labels)
    test_dataset = MNISTDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader, test_loader

if __name__ == "__main__":
    train_images_path = './Data/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte'
    train_labels_path = './Data/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    test_images_path = './Data/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    test_labels_path = './Data/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

    get_mnist_data_loaders(train_images_path, train_labels_path, test_images_path, test_labels_path)
    
    print("Dataset loaded")
