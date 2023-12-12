import torch
from preprocessData import loadPreprocessData
from defineModel import create_model, factFiction, get_tokenizer
from evaluateTrain import trainModel, plot_learning_curves
from optunaOptimization import objective
import optuna
from torch.utils.data import DataLoader
import os

# Using 4090 if not available uses CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#Obtain data and set max token length, BERT takes in a max of 512 tokens, when I obtained my "best hyperparamaters" I used 150 for efficiency
max_seq_len = 150 
train_filepath = 'C:/Users/andre/OneDrive/Documents/Data Science Projects/Python/FactCheck/train.csv'
train_df_augmented, val_df_augmented = loadPreprocessData(train_filepath, max_seq_len)


# Build the tokenizer 
tokenizer = get_tokenizer()
train_dataset = factFiction(train_df_augmented['text'], train_df_augmented['label'], tokenizer, max_seq_len)
val_dataset = factFiction(val_df_augmented['text'], val_df_augmented['label'], tokenizer, max_seq_len)

# For hyperparameter tuning, using Optuna
n_trials = 1 #setting to 1 for quick test
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, device), n_trials)  

# Print best hyperparameters obtained from the Optuna studies
best_hyperparams = study.best_trial.params
print("Best hyperparameters:", best_hyperparams)

# Given best hyperparameters, finally create the model using those paramaters
model = create_model(num_labels=2)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_hyperparams['lr'])
train_data_loader = DataLoader(train_dataset, batch_size=best_hyperparams['batch_size'], shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=best_hyperparams['batch_size'])

# Train the mdodel
model, training_losses, validation_losses, training_accuracies, validation_accuracies = trainModel(
    model, train_data_loader, val_data_loader, optimizer, device, best_hyperparams['epochs']
)

# Plotting Accuracy and Save Plot Data
plot_directory = 'C:/Users/andre/OneDrive/Documents/Data Science Projects/Python/FactCheck/PlotCurves'
if not os.path.exists(plot_directory):
    os.makedirs(plot_directory)
plot_file_path = os.path.join(plot_directory, 'learning_curve.png')

plot_learning_curves(
    training_losses, validation_losses, training_accuracies, validation_accuracies,
    best_hyperparams['lr'], best_hyperparams['batch_size'], best_hyperparams['epochs'],
    save_path=plot_file_path
)

output_dir = 'C:/Users/andre/OneDrive/Documents/Data Science Projects/Python/FactCheck/ModelSave'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model_save_path = f'{output_dir}/BestModel.pth'
torch.save(model.state_dict(), model_save_path)
