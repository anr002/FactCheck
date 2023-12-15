# Fake News?

## Overview
This project is centered on developing a machine learning model to distinguish between real and fake news. Utilizing BERT (Bidirectional Encoder Representations from Transformers), the goal is to build a robust classifier capable of accurately identifying the legitimacy of news articles and plan to incorporate this functionality to long form conversations like podcast utilizing OpenAI's Whisper library.

## Key Features
- **Dataset was obtained from Kaggle's Fake News Competition**: https://www.kaggle.com/competitions/fake-news/overview 
- **BERT Model Integration**: Employed BERT model from Hugging Face's Transformers library, for better accessibiltity and ease of use.
- **Hyperparameter Optimization**: Used Optuna for fine-tuning model parameters.
- **Advanced Data Preprocessing**: Implemented preprocessing techniques to prepare text data for efficient processing.
- **Model Training and Evaluation**: The model undergoes rigorous training and evaluation, cross checking accuracy with a test and validaiton set.
- **Real-Time News Analysis**: Includes functionality to fetch and analyze news articles in real-time, delivering practical applicability in identifying fake news (needs work).

## Libraries Used
- `torch`: For constructing and training the neural network.
- `transformers` (Hugging Face): For pre-trained BERT model and tokenizer
- `pandas`: For data manipulation and analysis.
- `numpy`: For handling numerical operations.
- `optuna`: For hyperparameter tuning.
- `matplotlib`: To visualize training and validation results.
- `tqdm`: For displaying training progress.
- `BeautifulSoup` and `requests`: To retrieve and parse online news articles.

## How It Works
1. **Data Preparation**: Preprocessing routines are applied to transform news articles into a suitable format for the model.
2. **Model Configuration and Tuning**: The BERT model is configured for sequence classification, with its hyperparameters optimized for our specific dataset.
3. **Training and Evaluation**: The model is trained and evaluated, ensuring high accuracy and reliability.
4. **Deployment for Prediction**: The model is deployed to classify news articles sourced from URLs, providing real-time predictions. (Later want to incorproate live fact checking of long form audio sources such as podcasts)

## Technical Approach
This project not only demonstrates my interest in machine learning but also reflects a commitment to using technology for addressing contemporary challenges. By leveraging the power of machine learning, I feel that a lot of societal issues remedied.

## Future Enhancements
Future plans include improving the model's accuracy, enhancing its efficiency, and expanding its capabilities to a wider range of news sources and formats. The project is a step towards utilizing cutting-edge technology in the fight against misinformation.

To obtain my bestModel.pth please download from https://drive.google.com/file/d/1Qahrag8060XhNoUbYIJbnyH_vULKSc4m/view?usp=sharing
