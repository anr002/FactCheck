import torch
from defineModel import create_model, get_tokenizer
from preprocessData import clean_txt, get_split
import requests
from bs4 import BeautifulSoup

def readArticle(url):
    # Fetch the content of the article from the URL
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    

    article_container = soup.find('div', {'class': 'articlebody'})

    # Extracting text within <p> container
    paragraphs = article_container.find_all('p') if article_container else []
    article_content = ' '.join([p.get_text() for p in paragraphs])
    return article_content


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

    # 0 = real, 1 = fake
    is_fake_news = predictions.mean().item() > 0.5
    return is_fake_news

def predictSingleText(model, tokenizer, text, max_seq_len):
    cleaned_text = clean_txt(text)
    split_text = get_split(cleaned_text, max_seq_len)

    
    inputs = tokenizer.encode_plus(
        split_text,
        add_special_tokens=True,
        max_length=max_seq_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits)

    avg_prob = probabilities.mean().item()
    return avg_prob > 0.5  



# Loading trained model from file, this model was trained previously but due to time. This is better. This received 98% accuracy for test and validation sets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(num_labels=2)
model.load_state_dict(torch.load('C:/Users/andre/OneDrive/Documents/Data Science Projects/Python/FactCheck/ModelSave/BestModel.pth'))
model.to(device)
model.eval()

# Max Seq Length < 512. For resources sake use 150.
tokenizer = get_tokenizer()
max_seq_len = 150

# URL of the news article to be tested. MSN works best
article_url = 'https://www.rollingstone.com/politics/politics-features/trump-election-plans-corrupt-voter-registration-eric-eagleai-1234920083/'
article_content = readArticle(article_url)


fakeNewsProb = predictNewsArticle(model, tokenizer, article_content, max_seq_len)
print(f"The article is {'fake' if fakeNewsProb else 'real'} news.")

