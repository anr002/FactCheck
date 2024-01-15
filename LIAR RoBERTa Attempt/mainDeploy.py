import torch
from preprocessData import loadPreprocessData
from defineModel import create_model, factFiction, get_tokenizer
from evaluateTrain import trainModel, plot_learning_curves
from torch.utils.data import DataLoader
from torch.optim import AdamW
import os

# Using 4090 if not available uses CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Obtain data
train_filepath = "C:/Users/andre/Documents/fake_and_real_news_dataset.csv"
max_seq_len = 150  # Define the maximum sequence length
train_df_augmented, val_df_augmented = loadPreprocessData(train_filepath, max_seq_len)

# Build tokenizer
tokenizer = get_tokenizer()
train_dataset = factFiction(train_df_augmented['text'], train_df_augmented['label'], tokenizer, max_seq_len)
val_dataset = factFiction(val_df_augmented['text'], val_df_augmented['label'], tokenizer, max_seq_len)

# Best hyperparameters from Optuna study
best_hyperparams = {
    'lr': 3.787133473003539e-05,
    'batch_size': 64,
    'epochs': 5,
    'weight_decay': 0.0016860454811001672,
    'optimizer_type': 'AdamW',
    'grad_clip': 4.770796982957951,
    'warmup_steps': 383
}

# Create the model with given parameters 
model = create_model(num_labels=2)
model.to(device)

# Create the optimizer with the best hyperparameters
optimizer = AdamW(model.parameters(), lr=best_hyperparams['lr'], weight_decay=best_hyperparams['weight_decay'])

# Create data loaders with the best batch size
train_data_loader = DataLoader(train_dataset, batch_size=best_hyperparams['batch_size'], shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=best_hyperparams['batch_size'])



# Train the model with the best hyperparameters
model, training_losses, validation_losses, training_accuracies, validation_accuracies = trainModel(
    model=model,
    train_data_loader=train_data_loader,
    val_data_loader=val_data_loader,
    optimizer=optimizer,
    device=device,
    n_epochs=best_hyperparams['epochs'],
    grad_clip=best_hyperparams['grad_clip'],
)

# Plot accuracy data
plot_directory = 'C:/Users/andre/Documents/PlotCurves'
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)
plot_file_path = os.path.join(plot_directory, 'learning_curve.png')

plot_learning_curves(
    training_losses, validation_losses, training_accuracies, validation_accuracies,
    best_hyperparams['lr'], best_hyperparams['batch_size'], best_hyperparams['epochs'],
    save_path=plot_file_path
)

# Save model
output_dir = 'C:/Users/andre/Documents/ModelSave'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model_save_path = f'{output_dir}/BestModelRoBERTa.pth'
torch.save(model.state_dict(), model_save_path)