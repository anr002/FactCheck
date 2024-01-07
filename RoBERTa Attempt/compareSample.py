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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(num_labels=2)
model.load_state_dict(torch.load('C:/Users/andre/Documents/ModelSave/BestModel.pth'))
model.to(device)
model.eval()

tokenizer = get_tokenizer()
max_seq_len = 150

test_data = pd.read_csv('C:/Users/andre/Documents/FactCheck/RoBERTa Attempt/test.csv')

predictions = []

for i, row in test_data.iterrows():
    article_content = row['text']  

    if isinstance(article_content, str):
        is_fake_news = predictNewsArticle(model, tokenizer, article_content, max_seq_len)

        predictions.append(int(is_fake_news))
    else:
        predictions.append(0) 

predicted_data = pd.DataFrame({
    'id': test_data['id'],
    'label': predictions
})

predicted_data.to_csv('predictions.csv', index=False)



answer_key = pd.read_csv('C:/Users/andre/Documents/FactCheck/RoBERTa Attempt/submit.csv')

accuracy = accuracy_score(answer_key['label'], predicted_data['label'])

print(f'Accuracy: {accuracy * 100:.2f}%')