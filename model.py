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
    
if __name__ == "__main__":
    model = CNNModel()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model) # Enable multi-GPU
        print("Activate dual VGA")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
