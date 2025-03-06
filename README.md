Sentiment Analysis Using DistilBERT & Machine Learning Models
🚀 A deep learning-based sentiment analysis project using DistilBERT embeddings and multiple classifiers, including Logistic Regression, Random Forest, and Gradient Boosting.

📌 Project Overview
This project performs sentiment analysis on the IMDB dataset, classifying reviews as either positive or negative. We use DistilBERT, a lighter version of BERT, to extract text embeddings, which are then used for training deep learning and machine learning models.

📂 Dataset
Dataset Name: IMDB Dataset
Number of Classes: 2 (Positive, Negative)
Number of Records: 50,000 (Balanced)

🛠️ Technologies & Libraries Used
Python
TensorFlow/Keras
Hugging Face Transformers (DistilBERT)
Scikit-learn
Pandas
NumPy
NLTK
Joblib

🔨 Project Workflow
1️⃣ Load Dataset
2️⃣ Preprocess Text (lowercase, remove special characters, stopwords, stemming)
3️⃣ Extract DistilBERT embeddings
4️⃣ Train Multiple Models (Deep Learning, Logistic Regression, Random Forest, Gradient Boosting)
5️⃣ Evaluate Performance
6️⃣ Save Trained Models
7️⃣ Make Predictions on New Reviews

🧩 Model Architectures
Deep Learning Model

Input Layer: 768 (DistilBERT embedding size)
Hidden Layer: 128 neurons, ReLU activation
Dropout: 0.3
Output Layer: 1 neuron, Sigmoid activation
Machine Learning Models

Logistic Regression
Random Forest (200 estimators)
Gradient Boosting (100 estimators, learning rate 0.1)
