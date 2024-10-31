import torch.nn as nn
import torch.optim as optim
import torch

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size= 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size= 3, padding= 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # Embedding layer
        self.fc2 = nn.Linear(128, 47) # Output

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = torch.relu(self.fc1(x)) # Embedding
        x = self.fc2(x) # Output
        return x
    
class Autoencoder(nn.Module):
    def __init__(self, num_classes=47):  # Adjust number of classes for EMNIST
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in range [0, 1]
        )
        
        # Classifier for classification, using the correct flattened size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),  # Updated to match the encoder output size (1024)
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Output layer for classification
        )
        
    def forward(self, x, classify=False):
        x = self.encoder(x)
        if classify:
            return self.classifier(x)  # Use this for classification
        else:
            return self.decoder(x)  # Use this for reconstruction


    
if __name__ == "__main__":
    # Check encoder output size for a sample input
    model = Autoencoder()
    sample_input = torch.randn(1, 1, 28, 28)  # Example input for EMNIST
    encoded_output = model.encoder(sample_input)
    print("Encoder output shape:", encoded_output.shape)

