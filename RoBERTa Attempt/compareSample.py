import pandas as pd
import torch
from defineModel import create_model, get_tokenizer
from preprocessData import clean_txt, get_split
from sklearn.metrics import accuracy_score

def predictNewsArticle(model, tokenizer, article_content, max_seq_len):
    cleaned_content = clean_txt(article_content)
    split_content = get_split(cleaned_content, max_seq_len)
    inputs = tokenizer(
        split_content, 
        padding=True, 
        truncation=True, 
        max_length=max_seq_len, 
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.sigmoid(outputs.logits).squeeze()
    is_fake_news = predictions.mean().item() > 0.5
    return is_fake_news

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(num_labels=2)
model.load_state_dict(torch.load('C:/Users/andre/OneDrive/Documents/Data Science Projects/Python/FactCheck/ModelSave/BestModel.pth'))
model.to(device)
model.eval()

# Load the tokenizer
tokenizer = get_tokenizer()
max_seq_len = 150

# Load the test data
test_data = pd.read_csv('submit.csv')

# Initialize a list to store the predictions
predictions = []

# Iterate over the test articles
for i, row in test_data.iterrows():
    # Get the article content
    article_content = row['text']  # Replace 'text' with the correct column name

    # Predict the label for the article
    is_fake_news = predictNewsArticle(model, tokenizer, article_content, max_seq_len)

    # Append the prediction to the list
    predictions.append(int(is_fake_news))

# Create a DataFrame with the IDs and predicted labels
predicted_data = pd.DataFrame({
    'id': test_data['id'],
    'label': predictions
})

# Save the DataFrame to a CSV file
predicted_data.to_csv('predictions.csv', index=False)

# Load the answer key
answer_key = pd.read_csv('sample.csv')

# Calculate accuracy
accuracy = accuracy_score(answer_key['label'], predicted_data['label'])

print(f'Accuracy: {accuracy * 100:.2f}%')