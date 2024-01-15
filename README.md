# Fake News?

## Overview

This project is evolving to develop a more accurate machine learning model to distinguish between real and fake news. After some success with BERT, I am now exploring RoBERTa (A Robustly Optimized BERT Pretraining Approach) due to its improved performance characteristics. The goal remains to build a robust classifier capable of accurately identifying the legitimacy of news articles, with plans to extend this functionality to long-form conversations like podcasts using OpenAI's Whisper library.

## Recent Changes

With RoBERTa I was able to obtain the following:
Loss: 0.0358, Training Accuracy: 0.9951, Validation Accuracy: 0.9565

This was using the same Kaggle dataset but I am currently running a more rigerous training session using the new enhanced dataset.

## Enhanced Hyperparameter Tuning and Extensive Training

I have expanded the scope of hyperparameter tuning to include additional parameters such as learning rate schedules, weight decay, and warm-up ratios. This comprehensive approach allows us to explore a wider configuration space and identify the most effective settings for our model.

To ensure the robustness of the model, I am currently training the model to run through 100 studies which will take much longer but will ensure that nothing is missed. It is important to note that the transition to roBERTa allows for me to expand the tuning and study length since roBERTa is more efficient than BERT at a minimal price.

## Key Features

- **Enhanced Dataset**: In addition to the dataset obtained from Kaggle's Fake News Competition, we are now incorporating the LIAR dataset to improve the model's understanding of nuanced truth classifications.
- **RoBERTa Model Integration**: Transitioned from BERT to RoBERTa model from Hugging Face's Transformers library for enhanced model performance.
- **Expanded Hyperparameter Optimization**: Conducting 100 studies with Optuna to fine-tune a broader range of hyperparameters, aiming for the most optimized model configuration.
- **Advanced Data Preprocessing**: Continuation of our sophisticated preprocessing techniques to prepare text data for the most efficient processing by our models.
- **Rigorous Model Training and Evaluation**: Maintaining our commitment to rigorous training and evaluation to ensure high accuracy and reliability.
- **Real-Time News Analysis**: Developing functionality to fetch and analyze news articles in real-time, with the goal of delivering practical applicability in identifying fake news.

## Libraries Used

- `torch`: For constructing and training the neural network.
- `transformers` (Hugging Face): For the pre-trained RoBERTa model and tokenizer.
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `optuna`: For hyperparameter tuning.
- `matplotlib`: To visualize training and validation results.
- `tqdm`: For displaying training progress.
- `BeautifulSoup` and `requests`: To retrieve and parse online news articles.

## How It Works

1. **Data Preparation**: Applying preprocessing routines to transform news articles into a format suitable for the model.
2. **Model Configuration and Tuning**: Configuring the RoBERTa model for sequence classification and optimizing its hyperparameters for our specific datasets.
3. **Training and Evaluation**: Training the model and evaluating its performance to ensure accuracy and reliability.
4. **Deployment for Prediction**: Preparing to deploy the model to classify news articles sourced from URLs, with future plans to extend to live fact-checking of long-form audio sources.

## Technical Approach

This project reflects a commitment to using technology to address contemporary challenges such as misinformation. By leveraging advanced machine learning techniques, we aim to contribute to societal well-being.

## Future Enhancements

Future updates will focus on improving the model's accuracy, enhancing its efficiency, and expanding its capabilities to a wider range of news sources and formats. The project is a step towards utilizing cutting-edge technology in the fight against misinformation.

To download the `bestModel.pth`, please visit this [Google Drive link](https://drive.google.com/file/d/1Qahrag8060XhNoUbYIJbnyH_vULKSc4m/view?usp=sharing).
