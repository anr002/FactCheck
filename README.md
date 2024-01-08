# Fake News?

## Changes
With RoBERTa I was able to obtain the following:
Loss: 0.0358, Training Accuracy: 0.9951, Validation Accuracy: 0.9565

I am currently working to integrate the LIAR dataset. This dataset requires thourough preprocessing as the dataset provides claims with a verdict. The verdict identifier used are varied. For binary classification (fake vs. reliable), I will need to map these to two categories. For instance, 'true', 'mostly-true', and 'half-true' could be considered 'real', while 'barely-true', 'false', and 'pants-fire' could be considered 'fake'



## References
LIAR: A BENCHMARK DATASET FOR FAKE NEWS DETECTION

William Yang Wang, "Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection, to appear in Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL 2017), short paper, Vancouver, BC, Canada, July 30-August 4, ACL.

## Overview

There's been some success in learning to work with BERT but the application is very resouce intensive. I previously rented a server with an NVIDIA GPU but it was quite expensive and building the model took about 20 hours.

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
