📰 Fake News Detection System
📌 Overview

This project aims to detect fake vs. real news articles using both traditional ML models (TF-IDF + Logistic Regression) and state-of-the-art Transformer models (DistilBERT, BERT, RoBERTa).

It was developed as part of the NIT Hackathon, showcasing how Natural Language Processing (NLP) can help combat misinformation.

⚙️ Features

✅ Data cleaning & preprocessing (stopwords removal, lemmatization, regex filters)

✅ Train/Validation/Test dataset splits

✅ Baseline ML model (Logistic Regression with TF-IDF)

✅ Transformer-based classifiers (DistilBERT, BERT)

✅ Model evaluation (Accuracy, Precision, Recall, F1-score, Classification Report)

✅ Visualization (Wordclouds, Label Distribution plots)

✅ Claim extraction using spaCy

✅ Final article-level decision (aggregated claim classification)

📂 Project Structure
.
├── data/                         # Original datasets (Fake.csv, True.csv, fake_or_real_news.csv)
├── cleaned_data/                 # Train, Validation, Test splits
├── models/                       # Saved fine-tuned models
├── preprocessing.py              # Data cleaning & preprocessing functions
├── train_distilbert.py           # DistilBERT training script
├── baseline_tfidf.py             # TF-IDF + Logistic Regression baseline
├── inference.py                  # Claim extraction + final article analysis
└── README.md                     # Project documentation

📊 Datasets

## Dataset

The dataset WELFake_Dataset.csv is too large for GitHub.
You can download it from Google Drive here:  
[WELFake_Dataset.csv (233 MB)](https://drive.google.com/file/d/1nZQrvCcy_umMI7nhaTf9wqhY9n3ksNG3/view?usp=drive_link)

**Instructions:**  
After downloading, place the CSV in the same directory as your code before running scripts that need it.

Deployed at:
https://huggingface.co/spaces/Bhargav77/Fake-News
