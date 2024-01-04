import torch
from torch.utils.data import DataLoader
from evaluateTrain import trainModel, evaluate
import optuna

# Optuna for hyperparameter tuning
def objective(trial, train_dataset, val_dataset, device):
    lr = trial.suggest_float('lr', 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 3, 5)

    from defineModel import create_model
    model = create_model(num_labels=2)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)

    
    model, training_losses, validation_losses, training_accuracies, validation_accuracies = trainModel(
        model, train_data_loader, val_data_loader, optimizer, device, epochs
    )

    
    val_accuracy = evaluate(model, val_data_loader, device)
    return val_accuracy

