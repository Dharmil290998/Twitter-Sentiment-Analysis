import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from collections import Counter
import pickle
import re

# Load the test dataset
data = pd.read_csv("C:/Users/Dell/OneDrive/Desktop/sem 3/NLP/ass-4/Sentiment_Data.csv", encoding='ISO-8859-1', nrows=10000)

# Preprocess the tweets for prediction
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+', '', text)  # Remove mentions but keep hashtags
    text = re.sub(r'[^A-Za-z0-9\s#]', '', text)  # Remove punctuation but keep hashtags
    text = text.lower()  # Convert to lowercase
    return text

data['processed_text'] = data['Tweet'].apply(clean_text)

# Load the trained LSTM model and tokenizer
model = load_model('C:/Users/Dell/OneDrive/Desktop/sem 3/deploymnet/Final Presentation/lstm_model.h5')
with open('C:/Users/Dell/OneDrive/Desktop/sem 3/deploymnet/Final Presentation/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Function to filter tweets by hashtag
def filter_tweets_by_hashtag(data, hashtag):
    hashtag = hashtag.lower().strip("#")
    filtered_data = data[data['Tweet'].str.contains(f"#{hashtag}", case=False, na=False)]
    return filtered_data

# Function to analyze sentiment and classify risk
def analyze_sentiment_and_classify_risk(filtered_tweets, tokenizer, max_length, model):
    if filtered_tweets.empty:
        return {}, "No tweets found for this hashtag."

    # Preprocess tweets for prediction
    sequences = tokenizer.texts_to_sequences(filtered_tweets['processed_text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

    # Predict sentiments
    predictions = model.predict(padded_sequences)
    predicted_classes = np.argmax(predictions, axis=1)  # Get class with highest probability

    # Map predictions back to sentiment labels and classify risk
    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    risk_mapping = {"Negative": "High Risk", "Neutral": "Medium Risk", "Positive": "Low Risk"}
    predicted_sentiments = pd.Series(predicted_classes).map(sentiment_mapping)
    risk_classification = predicted_sentiments.map(risk_mapping).value_counts(normalize=True) * 100

    # Get the Market Trend (dominant risk)
    market_trend = risk_classification.idxmax()
    return risk_classification.to_dict(), market_trend

# Extract trending hashtags
def get_trending_hashtags(data):
    hashtags = data['Tweet'].apply(lambda x: re.findall(r"#\w+", x.lower())).sum()
    return [tag for tag, _ in Counter(hashtags).most_common(5)]

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide", page_icon="📊")

# Page Styling
st.markdown(
    """
    <style>
    .header-title {
        background: linear-gradient(90deg, #ff6f61, #ffc107);
        color: white;
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .content-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 5px;
    }
    .input-section {
        text-align: center;
        margin-bottom: 20px;
    }
    .pie-chart {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .trending-section {
        text-align: center;
        margin-top: 20px;
    }
    .recent-tweets {
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown("<div class='header-title'>Tweet Scraper and Sentiment Analysis System</div>", unsafe_allow_html=True)

# Input Section
st.markdown("<div class='content-container'>", unsafe_allow_html=True)
st.markdown("<div class='input-section'>", unsafe_allow_html=True)
hashtag_input = st.text_input("Enter your hashtag (e.g., #Tesla):", "")
st.markdown("</div>", unsafe_allow_html=True)

if hashtag_input:
    # Filter tweets by hashtag
    filtered_tweets = filter_tweets_by_hashtag(data, hashtag_input)
    if filtered_tweets.empty:
        st.error("No tweets found for this hashtag.")
    else:
        # Perform analysis
        risk_summary, market_trend = analyze_sentiment_and_classify_risk(
            filtered_tweets, tokenizer, max_length=100, model=model
        )

        # Display Market Trend
        st.markdown(f"<h3 style='text-align: center;'>Market Trend: <span style='color: #4CAF50;'>{market_trend}</span></h3>", unsafe_allow_html=True)
        # Pie Chart Section
        st.markdown("<div class='pie-chart'>", unsafe_allow_html=True)

        # Resize the figure for a smaller pie chart
        fig, ax = plt.subplots(figsize=(1, 1))  # Smaller dimensions for the figure
        labels = list(risk_summary.keys())
        sizes = list(risk_summary.values())

        # Customize the pie chart
        wedges, texts, autotexts = ax.pie(
               sizes,
               labels=labels,
               autopct="%1.1f%%",
               startangle=140,
                colors=["#4CAF50", "#FFC107", "#F44336"],
                textprops={'fontsize': 4 },  # Reduce font size for labels
        )

        # Customize the appearance of the percentage numbers
        for autotext in autotexts:
            autotext.set_fontsize(4)  # Set smaller font size for percentage text
            autotext.set_color("black")  # Optional: Set color for percentage text

        ax.axis("equal")  # Equal aspect ratio ensures the pie is drawn as a circle
        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)


        # Trending Hashtags
        trending_hashtags = [
            f"#{tag.strip('#')}" for tag, _ in Counter(data["Tweet"].str.findall(r"#\w+").sum()).most_common(5)
        ]
        st.markdown("<div class='trending-section'><h3>Trending Hashtags</h3></div>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{', '.join(trending_hashtags)}</p>", unsafe_allow_html=True)

        # Recent Tweets
        st.markdown("<div class='trending-section'><h3>Recent Tweets</h3></div>", unsafe_allow_html=True)
        sample_tweets = filtered_tweets['Tweet'].head(10).tolist()
        for tweet in sample_tweets:
            st.markdown(f"<p class='recent-tweets'>{tweet}</p>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
