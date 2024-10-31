import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from data_loader import get_data_loaders

# Define the Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Keep dimensions 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downsize to 14x14
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Downsize to 7x7
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # Upsize to 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1), # Upsize to 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  # Keep 28x28
            nn.Sigmoid()  # Output in range [0, 1]
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Check if multiple GPUs are available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = Autoencoder()

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    autoencoder = nn.DataParallel(autoencoder)

autoencoder.to(device)

# Hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Data preparation
transform = transforms.Compose([transforms.ToTensor()])

train_images_path = './Data/EMNIST/emnist-balanced-train-images-idx3-ubyte'
train_labels_path = './Data/EMNIST/emnist-balanced-train-labels-idx1-ubyte'
test_images_path = './Data/EMNIST/emnist-balanced-test-images-idx3-ubyte'
test_labels_path = './Data/EMNIST/emnist-balanced-test-labels-idx1-ubyte'
train_loader, _ = get_data_loaders(train_images_path, train_labels_path, test_images_path, test_labels_path)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for images, _ in train_loader:
        images = images.to(device)

        # Forward pass
        outputs = autoencoder(images)
        loss = criterion(outputs, images)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
