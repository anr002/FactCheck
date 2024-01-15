import torch
from defineModel import create_model, get_tokenizer
from preprocessData import clean_txt, get_split
import requests
from bs4 import BeautifulSoup
import csv

def save_article_to_csv(url, content, prediction, csv_filename='articles.csv'):
    # Replace newlines in content to avoid breaking CSV format
    content = content.replace('\n', ' ').replace('\r', ' ')
    with open(csv_filename, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([url, content, 'fake' if prediction else 'real'])
    print(f"Article saved to CSV with prediction: {'fake' if prediction else 'real'}")

def readArticle(url):
    try:
        response = requests.get(url)
        # Check if the request was successful
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all paragraph tags with the specified class
            article_paragraphs = soup.find_all('p', class_='paragraph larva')
            
            # Check if any paragraphs were found
            if not article_paragraphs:
                print("No paragraphs with the specified class were found.")
                return ""
            
            # Extract the text from each paragraph and join them into a single string
            article_content = ' '.join(paragraph.get_text() for paragraph in article_paragraphs)
            
            # Check if any text was extracted
            if not article_content.strip():
                print("No content extracted from the article.")
                return ""
            
            print("Content extracted successfully.")
            return article_content
        else:
            print(f"Failed to retrieve the article. Status code: {response.status_code}")
            return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""

def predictNewsArticle(model, tokenizer, article_content, max_seq_len):
    cleaned_content = clean_txt(article_content)
    print(f"Cleaned content length: {len(cleaned_content)}")  # Debugging print statement
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

# Load the trained model from file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(num_labels=2)
model_path = 'C:/Users/andre/Documents/ModelSave/BestModelRoBERTa.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

tokenizer = get_tokenizer()
max_seq_len = 150

# URL of the news article to be tested
article_url = 'https://www.rollingstone.com/politics/politics-features/trump-election-plans-corrupt-voter-registration-eric-eagleai-1234920083/'
article_content = readArticle(article_url)

if article_content:
    fakeNewsProb = predictNewsArticle(model, tokenizer, article_content, max_seq_len)
    save_article_to_csv(article_url, article_content, fakeNewsProb)
    print(f"The article is {'fake' if fakeNewsProb else 'real'} news.")
else:
    print("No content to analyze.")