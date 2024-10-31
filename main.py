import torch
from data_loader import get_data_loaders
from model import CNNModel
from train_eval import train, evaluate_baseline, plot_metrics, plot_misclassified_samples, plot_confusion_matrix, plot_confusion_matrix_emnist
import torch.optim as optim
import torch.nn as nn

if __name__ == "__main__":
    # MNIST dataset
    # train_images_path = './Data/MNIST/train-images-idx3-ubyte/train-images-idx3-ubyte'
    # train_labels_path = './Data/MNIST/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    # test_images_path = './Data/MNIST/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    # test_labels_path = './Data/MNIST/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
    #train_loader, test_loader = get_mnist_data_loaders(train_images_path, train_labels_path, test_images_path, test_labels_path)

    # EMNIST dataset
    train_images_path = './Data/EMNIST/emnist-balanced-train-images-idx3-ubyte'
    train_labels_path = './Data/EMNIST/emnist-balanced-train-labels-idx1-ubyte'
    test_images_path = './Data/EMNIST/emnist-balanced-test-images-idx3-ubyte'
    test_labels_path = './Data/EMNIST/emnist-balanced-test-labels-idx1-ubyte'
    train_loader, test_loader = get_data_loaders(train_images_path, train_labels_path, test_images_path, test_labels_path)

    model = CNNModel()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model) # Enable multi-GPU
        print("Activate dual VGA")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    best_accuracy = 0.0
    metrics = {
    "train_loss": [],
    "train_accuracy": [],
    "train_accuracy": [],
    "test_loss": [],
    "test_accuracy": [],
    "test_precision": [],
    "test_recall": [],
    "test_f1_score": [],
    }
    for epoch in range(num_epochs):
        # Train the model on the training data
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        metrics['train_loss'].append(train_loss)
        # Calculate test accuracy every epoch
        accuracy, precision, recall, f1, cm, test_loss = evaluate_baseline(model, test_loader, criterion, device)
        metrics["train_accuracy"].append(train_accuracy)
        metrics["test_accuracy"].append(accuracy)
        metrics["test_precision"].append(precision)
        metrics["test_recall"].append(recall)
        metrics["test_f1_score"].append(f1)
        metrics["test_loss"].append(test_loss)
        # Print basic metrics every epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f"best_model_epoch_{epoch + 1}.pth")
            print(f"Best model saved at epoch {epoch + 1} with accuracy: {accuracy:.4f}")
        # Every 5 epochs, calculate and print detailed metrics
        if (epoch + 1) % 5 == 0 or epoch + 1 == num_epochs:
            print(f"Detailed Metrics - Every 5 Epochs or Final Epoch")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
            plot_confusion_matrix_emnist(cm, "emnist_confusion_matrix.png")
    plot_misclassified_samples(model, test_loader, device)
    plot_metrics(metrics)
    # Save as TorchScript
    # scripted_model = torch.jit.script(model)
    # scripted_model.save("mnist_model_scripted.pt")

    # # Export to ONNX
    # dummy_input = torch.randn(1, 1, 28, 28, device=device)
    # torch.onnx.export(model, dummy_input, "mnist_model.onnx")
