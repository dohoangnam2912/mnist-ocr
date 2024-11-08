import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
import matplotlib.pyplot as plt
from PIL import Image

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

from PIL import Image

class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None, rotate=False):
        self.images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
        self.rotate = rotate

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        
        # Apply rotation if the rotate flag is set
        if self.rotate:
            image = image.transpose(-2, -1)  # Rotate by swapping height and width dimensions
            
        # Convert to numpy and then to PIL Image for compatibility with torchvision transforms
        image = image.squeeze(0).numpy()  # Convert to numpy
        image = Image.fromarray((image * 255).astype(np.uint8))  # Convert to PIL Image
        
        # Apply the augmentation transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Convert back to tensor if needed
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)  # Ensure the final format is a tensor

        return image, label


def get_data_loaders(train_images_path, train_labels_path, test_images_path, test_labels_path, batch_size=64): 
    # Define data augmentation transformations for the training dataset
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),                  # Rotate images by +/- 10 degrees
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random translation
        # transforms.RandomHorizontalFlip(),              # Horizontal flip
        transforms.ToTensor(),                           # Convert to tensor
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3)
    ])
    
    # Define simple transformations for the test dataset (no augmentation)
    test_transform = transforms.ToTensor()

    # Load images and labels
    train_images = load_images(train_images_path)
    train_labels = load_labels(train_labels_path)
    test_images = load_images(test_images_path)
    test_labels = load_labels(test_labels_path)

    # Create datasets
    train_dataset = MNISTDataset(train_images, train_labels, transform=train_transform, rotate=True)
    test_dataset = MNISTDataset(test_images, test_labels, transform=test_transform, rotate=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == "__main__":
    # File paths
    train_images_path = './Data/EMNIST/emnist-balanced-train-images-idx3-ubyte'
    train_labels_path = './Data/EMNIST/emnist-balanced-train-labels-idx1-ubyte'
    test_images_path = './Data/EMNIST/emnist-balanced-test-images-idx3-ubyte'
    test_labels_path = './Data/EMNIST/emnist-balanced-test-labels-idx1-ubyte'

    # Load data with augmentation for training set
    train_loader, test_loader = get_data_loaders(train_images_path, train_labels_path, test_images_path, test_labels_path)

    # Visualize a batch
    images, labels = next(iter(test_loader))
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
    print("Dataset loaded with data augmentation for training set")
