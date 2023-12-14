import torch
from preprocessData import loadPreprocessData
from defineModel import create_model, factFiction, get_tokenizer
from evaluateTrain import trainModel, plot_learning_curves
from torch.utils.data import DataLoader
import os

# Using 4090 if not available uses CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OBtain data
train_filepath = 'C:/Users/andre/OneDrive/Documents/Data Science Projects/Python/FactCheck/train.csv'
max_seq_len = 512  # Define the maximum sequence length
train_df_augmented, val_df_augmented = loadPreprocessData(train_filepath, max_seq_len)

# Build tokenizer
tokenizer = get_tokenizer()
train_dataset = factFiction(train_df_augmented['text'], train_df_augmented['label'], tokenizer, max_seq_len)
val_dataset = factFiction(val_df_augmented['text'], val_df_augmented['label'], tokenizer, max_seq_len)

# These parameters were obtained from studies of about 22 trials. The studies took over 24 hours to reach 22 trials and couldn't afford to keep it running
best_hyperparams = {'lr': 1.3078181107772179e-05, 'batch_size': 64, 'epochs': 4}

# Create the model with given parameters 
model = create_model(num_labels=2)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_hyperparams['lr'])
train_data_loader = DataLoader(train_dataset, batch_size=best_hyperparams['batch_size'], shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=best_hyperparams['batch_size'])

#Training model with my best paramaters (obtained from previous previous training)
model, training_losses, validation_losses, training_accuracies, validation_accuracies = trainModel(
    model, train_data_loader, val_data_loader, optimizer, device, best_hyperparams['epochs']
)

#Plot accuracy data
plot_directory = 'C:/Users/andre/OneDrive/Documents/Data Science Projects/Python/FactCheck/PlotCurves'
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)
plot_file_path = os.path.join(plot_directory, 'learning_curve.png')

plot_learning_curves(
    training_losses, validation_losses, training_accuracies, validation_accuracies,
    best_hyperparams['lr'], best_hyperparams['batch_size'], best_hyperparams['epochs'],
    save_path=plot_file_path
)

#Save model
output_dir = 'C:/Users/andre/OneDrive/Documents/Data Science Projects/Python/FactCheck/ModelSave'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model_save_path = f'{output_dir}/BestModel.pth'
torch.save(model.state_dict(), model_save_path)
