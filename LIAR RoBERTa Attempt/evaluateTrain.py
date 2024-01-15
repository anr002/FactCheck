import torch
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from tqdm import tqdm


def calculateAccuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y_true).float()
    accuracy = correct.sum() / len(correct)
    return accuracy

# Function to evaluate the model
def evaluate(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.logits, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_samples
    return accuracy

# Function to plot learning curves
import matplotlib.pyplot as plt

def plot_learning_curves(training_losses, validation_losses, training_accuracies, validation_accuracies, lr, batch_size, epochs, save_path):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.title(f'Loss over epochs (lr={lr}, batch_size={batch_size}, epochs={epochs})')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(training_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.title(f'Accuracy over epochs (lr={lr}, batch_size={batch_size}, epochs={epochs})')
    plt.legend()


    plt.savefig(save_path)
    plt.close()

# Modified training function with early stopping
def trainModel(model, train_data_loader, val_data_loader, optimizer, device, n_epochs, patience=4, grad_clip=None, scheduler=None):
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []
    criterion = CrossEntropyLoss()
    best_val_accuracy = 0
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_data_loader, desc=f"Epoch {epoch + 1}/{n_epochs}"):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()

            # Apply gradient clipping if grad_clip is set
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # Step the learning rate scheduler if provided
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

        avg_training_loss = total_loss / len(train_data_loader)
        training_losses.append(avg_training_loss)

        train_accuracy = evaluate(model, train_data_loader, device)
        val_accuracy = evaluate(model, val_data_loader, device)
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(val_accuracy)

        print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {avg_training_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('No improvement, early stopping.')
                break

    return model, training_losses, validation_losses, training_accuracies, validation_accuracies
