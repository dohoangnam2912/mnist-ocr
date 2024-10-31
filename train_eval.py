from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import string

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    running_loss /= len(train_loader)
    train_accuracy = correct / total
    return running_loss, train_accuracy

def evaluate_baseline(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            all_preds.extend(preds.cpu().numpy()) # Converted to CPU
            all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        test_loss /= len(test_loader)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

        cm = confusion_matrix(all_labels, all_preds)
        
        return accuracy, precision, recall, f1, cm, test_loss

def plot_metrics(metrics):
    epochs = range(1, len(metrics["train_loss"]) + 1)
    
    # Plot Train and Test Loss
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics["test_loss"], label="Test Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Test Loss Over Epochs")
    plt.legend()

    # Plot Train and Test Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, metrics["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, metrics["test_accuracy"], label="Test Accuracy",  color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Test Accuracy Over Epochs")
    plt.legend()

    # Plot Test Precision, Recall, F1 Score
    plt.subplot(2, 2, 3)
    plt.plot(epochs, metrics["test_precision"], label="Precision", color="blue")
    plt.plot(epochs, metrics["test_recall"], label="Recall", color="purple")
    plt.plot(epochs, metrics["test_f1_score"], label="F1 Score", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Precision, Recall, F1 Score Over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()



def plot_misclassified_samples(model, test_loader, device, num_samples=10):
    save_path = './evaluation_metrics/misclassified/miss.png'
    class_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Digits
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',  # Uppercase Letters
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'  # Selected lowercase letters
    ]
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(len(labels)):
                if preds[i] != labels[i] and len(misclassified_images) < num_samples:
                    misclassified_images.append(images[i].cpu())
                    misclassified_labels.append(labels[i].cpu().item())
                    misclassified_preds.append(preds[i].cpu().item())
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 15))
    for i in range(num_samples):
        img = misclassified_images[i].squeeze()
        true_label = class_labels[misclassified_labels[i]]
        pred_label = class_labels[misclassified_preds[i]]
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"True: {true_label}\nPred: {pred_label}")
        axes[i].axis('off')
    plt.savefig(save_path)
    print(f"Miss classified saved to {save_path}")
    plt.show()


def plot_confusion_matrix(cm, file_name):
    # Plot the confusion matrix
    save_dir = './evaluation_metrics/confusion_matrix/mnist/'
    save_path = save_dir + file_name
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    
    # Save the plot
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.show()

def plot_confusion_matrix_emnist(cm, file_name="emnist_confusion_matrix.png"):
    save_dir = './evaluation_metrics/confusion_matrix/emnist/'
    save_path = save_dir + file_name
    class_labels = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Digits
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',  
        'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'  
    ]
    # Plot the confusion matrix with labels
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix for EMNIST (Balanced)")
    
    # Save the plot
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.show()
