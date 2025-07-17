# 🌍 Climate Change Modelling— Sentiment Analysis Project
Sentiment Analysis on Climate Conversations— Modelling Public Perception of Climate Change

It focuses on analysing climate-related comments using **Natural Language Processing (NLP)**, sentiment classification, and entity recognition techniques to uncover public perception trends around climate change.


## 📌 Project Overview

As climate change becomes one of the most discussed topics globally, analysing public sentiment and themes can provide valuable insights. In this project, we use Python to explore, clean, and model real-world climate conversation data from NASA-related sources.


## 🧠 Key Features

- ✅ **Data Cleaning & Preprocessing**  
  Handle missing values, normalise text, remove noise and stopwords.
- 💬 **Sentiment Analysis**  
  Assign polarity (positive, negative, neutral) using `TextBlob`.
- 🏷️ **Sentiment Classification Model**  
  Train a logistic regression model to classify sentiments using TF-IDF features.
- 🧾 **Named Entity Recognition (NER)**  
  Extract locations, organisations, and events using spaCy.
- 📊 **Trend Analysis**  
  Time-series visualisation of sentiment evolution month-over-month.
- 🔥 **Engagement Insights**
  Explore the correlation between sentiment and the number of likes and comments.


## 🗃️ Dataset

- Source: Internally provided (`climate_nasa.csv`)
- Format: Real social media comment data (text, date, likes, etc.)

## 🛠️ Tools & Libraries

- **Python** (Pandas, NumPy, Seaborn, Matplotlib)
- **NLP**: NLTK, TextBlob, spaCy
- **Modelling**: Scikit-learn (Logistic Regression)
- **Preprocessing**: TF-IDF Vectorizer
- **Datetime & Trends**: `pandas.to_datetime()`, time grouping


## 📈 Visualizations

- 📦 Sentiment Distribution Bar Charts
- 📉 Monthly Sentiment Trends
- 🧠 Top Named Entities Bar Plot (ORG, GPE, EVENT)
- ✅ Confusion Matrix for Model Evaluation

