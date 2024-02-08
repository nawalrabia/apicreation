#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, jsonify, request
import pandas as pd
import pickle
import requests

app = Flask(__name__)

# Loading the model and vectorizer
def load_model_and_vectorizer(filename):
    with open(filename, 'rb') as model_file:
        loaded_model, loaded_vectorizer = pickle.load(model_file)
    return loaded_model, loaded_vectorizer

# Loading the model and vectorizer
model_filename_pkl = 'random_forest_model8.pkl'
random_forest_model, tfidf_vectorizer = load_model_and_vectorizer(model_filename_pkl)

# Function to classify with additional checks for spaces and URL length
def classify_text_with_checks(text, remove_spaces=True, convert_short_url=True):
    # Remove spaces if the remove_spaces parameter is True
    if remove_spaces:
        text = text.replace(" ", "")

    # Converts short URL to long URL if the convert_short_url parameter is True
    if convert_short_url:
        expanded_url = expand_short_url(text)
        if expanded_url:
            text = expanded_url

    text_tfidf = tfidf_vectorizer.transform([text])
    prediction = random_forest_model.predict(text_tfidf)[0]
    return prediction

# Function to expand short URL
def expand_short_url(short_url):
    try:
        response = requests.head(short_url, allow_redirects=True)
        long_url = response.url
        return long_url
    except Exception as e:
        print(f"Error expanding short URL: {e}")
        return None

# API Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Getting input data in JSON format
        json_data = request.json

        # Extracting URL from JSON data
        user_input_link = json_data.get('url', '')

        # Performing classification for link(benign or csam)
        prediction_link = classify_text_with_checks(user_input_link, remove_spaces=True, convert_short_url=True)

        # Returns predictions as JSON
        return jsonify({'prediction': prediction_link})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)

