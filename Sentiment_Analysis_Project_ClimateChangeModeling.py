import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Text & NLP
import re
import string
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import spacy


# ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# ================================
# 📥 Load Dataset
# ================================

df = pd.read_csv("Project2_ClimateChangeModeling\climateNasa.csv")

print("Shape:", df.shape)
print(df.head())

# ================================
# 🔍 Initial Exploration
# ================================
print(df.info())
print(df.describe(include='all'))
print("Missing values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# ================================
# 🧹 Text Preprocessing
# ================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    try:
        tokens = nltk.word_tokenize(text)
    except:
        tokens = text.split()  # fallback if punkt fails
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)


df['clean_text'] = df['text'].apply(clean_text)

# ================================
# 💬 Sentiment Analysis
# ================================
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

df['sentiment_score'] = df['clean_text'].apply(get_sentiment)
df['sentiment_label'] = df['sentiment_score'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

# ================================
# 📊 Visualizations
# ================================

# Sentiment distribution
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='sentiment_label', palette='Set2')
plt.title("Sentiment Distribution")
plt.show()

# Likes vs Sentiment
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x='sentiment_label', y='likesCount')
plt.title("Engagement (Likes) by Sentiment")
plt.show()


# ================================
# 🧠 Named Entity Recognition (NER)
# ================================
def extract_entities(text):
    if pd.isnull(text) or not isinstance(text, str):
        return []
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


df['entities'] = df['text'].apply(extract_entities)

# Flatten and count entities
from collections import Counter
all_entities = sum(df['entities'], [])
entity_counts = Counter([ent for ent, label in all_entities if label in ['ORG', 'GPE', 'EVENT']])

# Top 10
top_entities = dict(entity_counts.most_common(10))
plt.figure(figsize=(10,4))
sns.barplot(x=list(top_entities.values()), y=list(top_entities.keys()), palette='Blues_r')
plt.title("Top Named Entities (ORG, GPE, EVENT)")
plt.xlabel("Count")
plt.ylabel("Entity")
plt.show()

# ================================
# 🤖 Text Classification (ML Model)
# ================================
# Binary sentiment only
df_binary = df[df['sentiment_label'] != 'neutral']
X = df_binary['clean_text']
y = df_binary['sentiment_label'].map({'positive': 1, 'negative': 0})

# Vectorization
vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.25, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ================================
# 📈 Trend Analysis Over Time
# ================================
df['Date'] = pd.to_datetime(df['date'])
df['year_month'] = df['Date'].dt.to_period('M')

monthly_trend = df.groupby(['year_month', 'sentiment_label']).size().unstack().fillna(0)

monthly_trend.plot(kind='line', figsize=(12,6), marker='o')
plt.title("Monthly Sentiment Trend")
plt.ylabel("Number of Comments")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
