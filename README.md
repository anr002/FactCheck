# Fact News or Real News?

## Overview
This project is centered on developing a machine learning model to distinguish between real and fake news. Utilizing the advanced capabilities of BERT (Bidirectional Encoder Representations from Transformers), the goal is to build a robust classifier capable of accurately identifying the legitimacy of news articles.

## Key Features
- **Dataset was obtained from Kaggle's Fake News Competition**: https://www.kaggle.com/competitions/fake-news/overview 
- **BERT Model Integration**: Employs the BERT model from Hugging Face's Transformers library, renowned for its state-of-the-art performance in natural language processing tasks.
- **Hyperparameter Optimization**: Uses Optuna for fine-tuning model parameters to achieve optimal performance.
- **Advanced Data Preprocessing**: Implements comprehensive preprocessing techniques to prepare text data for efficient processing by the BERT model.
- **Model Training and Evaluation**: The model undergoes rigorous training and evaluation, ensuring reliability and accuracy in its predictions.
- **Real-Time News Analysis**: Includes functionality to fetch and analyze news articles in real-time, delivering practical applicability in identifying fake news.

## Libraries Used
- `torch`: For constructing and training the neural network.
- `transformers` (Hugging Face): Provides easy access to the pre-trained BERT model and tokenizer, simplifying the implementation process.
- `pandas`: For data manipulation and analysis.
- `numpy`: For handling numerical operations.
- `optuna`: For systematic hyperparameter tuning.
- `matplotlib`: To visualize training and validation results.
- `tqdm`: For displaying training progress.
- `BeautifulSoup` and `requests`: To retrieve and parse online news articles.

## How It Works
1. **Data Preparation**: Preprocessing routines are applied to transform news articles into a suitable format for BERT.
2. **Model Configuration and Tuning**: The BERT model is configured for sequence classification, with its hyperparameters optimized for our specific dataset.
3. **Training and Evaluation**: The model is rigorously trained and evaluated, ensuring high accuracy and reliability.
4. **Deployment for Prediction**: The model is deployed to classify news articles sourced from URLs, providing real-time insights into their authenticity.

## Technical Approach
This project not only demonstrates my skills in machine learning but also reflects a commitment to using technology for addressing contemporary challenges. By leveraging the power of BERT through the Hugging Face Transformers library, this project simplifies the complexities of natural language understanding, making it more accessible and efficient.

## Future Enhancements
Future plans include improving the model's accuracy, enhancing its efficiency, and expanding its capabilities to a wider range of news sources and formats. The project is a step towards utilizing cutting-edge technology in the fight against misinformation.
