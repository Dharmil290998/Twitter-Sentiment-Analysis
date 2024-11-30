import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

# Load the test dataset
try:
    test_data = pd.read_csv(
        "C:/Users/Dell/OneDrive/Desktop/sem 3/NLP/ass-4/Sentiment_Data.csv",
        encoding="ISO-8859-1",
        nrows=10000
    )
except FileNotFoundError:
    raise FileNotFoundError("Test dataset not found. Please check the file path.")

# Preprocess the tweets for prediction
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+', '', text)  # Remove mentions but keep hashtags
    text = re.sub(r'[^A-Za-z0-9\s#]', '', text)  # Remove punctuation but keep hashtags
    text = text.lower()  # Convert to lowercase
    return text

test_data['processed_text'] = test_data['Tweet'].apply(clean_text)

# Load the trained LSTM model and tokenizer
try:
    model = load_model('C:/Users/Dell/OneDrive/Desktop/sem 3/deploymnet/Final Presentation/lstm_model.h5')
    with open('C:/Users/Dell/OneDrive/Desktop/sem 3/deploymnet/Final Presentation/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
except FileNotFoundError as e:
    raise FileNotFoundError("Model or tokenizer file not found. Please check the file paths.")

# Initialize Flask app
app = Flask(__name__)

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
    market_trend = risk_classification.idxmax() if not risk_classification.empty else "No trend"
    return risk_classification.to_dict(), market_trend

@app.route('/')
def home():
    return "Welcome to the Tweet Sentiment Analysis and Risk Classification API!"

@app.route('/analyze-risk', methods=['POST'])
def analyze_risk():
    # Get the hashtag from the POST request
    data = request.json
    hashtag = data.get('hashtag')

    # Validate input
    if not hashtag:
        return jsonify({"error": "Hashtag is required."}), 400

    # Filter tweets
    filtered_tweets = filter_tweets_by_hashtag(test_data, hashtag)
    if len(filtered_tweets) == 0:
        return jsonify({"error": f"No tweets found for hashtag: {hashtag}"}), 404

    # Analyze sentiment and classify risk
    risk_summary, market_trend = analyze_sentiment_and_classify_risk(filtered_tweets, tokenizer, max_length=100, model=model)

    # Get up to 10 tweets for display
    sample_tweets = filtered_tweets['Tweet'].head(10).tolist()

    # Return results
    return jsonify({
        "hashtag": hashtag,
        "risk_summary": risk_summary,
        "market_trend": market_trend,  # Replacing "dominant_risk" with "market_trend"
        "filtered_tweets_count": len(filtered_tweets),
        "sample_tweets": sample_tweets
    })

if __name__ == '__main__':
    app.run(debug=True)
