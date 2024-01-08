import torch
from torch.utils.data import DataLoader
from evaluateTrain import trainModel, evaluate
import optuna
from transformers import get_linear_schedule_with_warmup

# Optuna for hyperparameter tuning
def objective(trial, train_dataset, val_dataset, device):
    # Existing hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = trial.suggest_int('epochs', 3, 5)

    # Additional hyperparameters
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    optimizer_type = trial.suggest_categorical('optimizer_type', ['Adam', 'AdamW', 'SGD'])
    grad_clip = trial.suggest_float('grad_clip', 0.5, 5.0)



    from defineModel import create_model
    model = create_model(num_labels=2)
    model.to(device)

    # Define the optimizer with weight decay
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Create data loaders
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Scheduler (optional, if you want to include it)
    num_training_steps = epochs * len(train_data_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=trial.suggest_int('warmup_steps', 0, num_training_steps // 3),
        num_training_steps=num_training_steps
    )

    # Train the model and get the training and validation metrics
    model, training_losses, validation_losses, training_accuracies, validation_accuracies = trainModel(
        model, train_data_loader, val_data_loader, optimizer, scheduler, device, epochs, grad_clip
    )

    # Evaluate the model on the validation set
    val_accuracy = evaluate(model, val_data_loader, device)
    return val_accuracy