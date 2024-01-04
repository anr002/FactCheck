import torch
from preprocessData import loadPreprocessData
from defineModel import create_model, factFiction, get_tokenizer
from evaluateTrain import evaluate
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_filepath = 'C:/Users/andre/OneDrive/Documents/Data Science Projects/Python/FactCheck/train.csv'
max_seq_len = 150  # Define the maximum sequence length
train_df_augmented, val_df_augmented = loadPreprocessData(train_filepath, max_seq_len)

tokenizer = get_tokenizer()
train_dataset = factFiction(train_df_augmented['text'], train_df_augmented['label'], tokenizer, max_seq_len)
val_dataset = factFiction(val_df_augmented['text'], val_df_augmented['label'], tokenizer, max_seq_len)

#Loading the trained model, note model was trained using a seq len of 150
model_save_path = 'C:/Users/andre/OneDrive/Desktop/FactCheck/BestModelOutput/BestModel.pth'
model = create_model(num_labels=2)
model.load_state_dict(torch.load(model_save_path))
model.to(device)
model.eval() 


best_hyperparams = {'lr': 1.3078181107772179e-05, 'batch_size': 64, 'epochs': 4}
val_data_loader = DataLoader(val_dataset, batch_size=best_hyperparams['batch_size'])

val_accuracy = evaluate(model, val_data_loader, device)
print(f"Validation Accuracy: {val_accuracy}")


