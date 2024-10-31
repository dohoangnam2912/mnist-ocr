import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import matplotlib.pyplot as plt

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
        # if self.transform:
        #     image = self.transform(image)
        return image, label

def get_data_loaders(train_images_path, train_labels_path, test_images_path, test_labels_path, batch_size=64, transform=None): 
    train_images = load_images(train_images_path)
    train_labels = load_labels(train_labels_path)
    test_images = load_images(test_images_path)
    test_labels = load_labels(test_labels_path)

    train_dataset = MNISTDataset(train_images, train_labels, transform=None)
    test_dataset = MNISTDataset(test_images, test_labels, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    return train_loader, test_loader

if __name__ == "__main__":
    # train_images_path = './Data/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte'
    # train_labels_path = './Data/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    # test_images_path = './Data/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    # test_labels_path = './Data/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    train_images_path = './Data/EMNIST/emnist-balanced-train-images-idx3-ubyte'
    train_labels_path = './Data/EMNIST/emnist-balanced-train-labels-idx1-ubyte'
    test_images_path = './Data/EMNIST/emnist-balanced-test-images-idx3-ubyte'
    test_labels_path = './Data/EMNIST/emnist-balanced-test-labels-idx1-ubyte'
    train_loader, data_loader = get_data_loaders(train_images_path, train_labels_path, test_images_path, test_labels_path)
    images, labels = next(iter(data_loader))

    # Visualize a batch
    class_labels = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Digits
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',  # Uppercase Letters
        'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'  # Selected lowercase letters
        ]
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        img = images[i].squeeze().cpu().numpy()
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label: {class_labels[labels[i].item()]}")
        axes[i].axis('off')
    plt.show()
    print("Dataset loaded")
