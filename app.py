import requests
import json
from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
from flask_cors import CORS
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pyfiglet import figlet_format


app = Flask(__name__)
CORS(app)

# Load the AI detection model
model = tf.keras.models.load_model('Models/ai_detection_model.h5')
tokenizer = Tokenizer(num_words=5000)  # Initialize the tokenizer

def detect_plagiarism(input_text):
    burp0_url = "https://papersowl.com:443/plagiarism-checker-send-data"
    burp0_cookies = {
        "PHPSESSID": "qjc72e3vvacbtn4jd1af1k5qn1",
        "first_interaction_user": "%7B%22referrer%22%3A%22https%3A%5C%2F%5C%2Fwww.google.com%5C%2F%22%2C%22internal_url%22%3A%22%5C%2Ffree-plagiarism-checker%22%2C%22utm_source%22%3Anull%2C%22utm_medium%22%3Anull%2C%22utm_campaign%22%3Anull%2C%22utm_content%22%3Anull%2C%22utm_term%22%3Anull%2C%22gclid%22%3Anull%2C%22msclkid%22%3Anull%2C%22adgroupid%22%3Anull%2C%22targetid%22%3Anull%2C%22appsflyer_id%22%3Anull%2C%22appsflyer_cuid%22%3Anull%2C%22cta_btn%22%3Anull%7D",
        "first_interaction_order": "%7B%22referrer%22%3A%22https%3A%5C%2F%5C%2Fwww.google.com%5C%2F%22%2C%22internal_url%22%3A%22%5C%2Ffree-plagiarism-checker%22%2C%22utm_source%22%3Anull%2C%22utm_medium%22%3Anull%2C%22utm_campaign%22%3Anull%2C%22utm_content%22%3Anull%2C%22utm_term%22%3Anull%2C%22gclid%22%3Anull%2C%22msclkid%22%3Anull%2C%22adgroupid%22%3Anull%2C%22targetid%22%3Anull%2C%22appsflyer_id%22%3Anull%2C%22appsflyer_cuid%22%3Anull%2C%22cta_btn%22%3Anull%7D",
        "affiliate_user": "a%3A3%3A%7Bs%3A9%3A%22affiliate%22%3Bs%3A9%3A%22papersowl%22%3Bs%3A6%3A%22medium%22%3Bs%3A9%3A%22papersowl%22%3Bs%3A8%3A%22campaign%22%3Bs%3A9%3A%22papersowl%22%3B%7D"
    }
    burp0_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0",
        "Accept": "*/*",
        "Accept-Language": "ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3",
        "Accept-Encoding": "gzip, deflate",
        "Referer": "https://papersowl.com/free-plagiarism-checker",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://papersowl.com",
        "Dnt": "1",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "same-origin",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
        "Te": "trailers",
        "Connection": "close"
    }
    burp0_data = {
        "is_free": "false",
        "plagchecker_locale": "en",
        "product_paper_type": "1",
        "title": '',
        "text": str(input_text)
    }

    r = requests.post(burp0_url, headers=burp0_headers, cookies=burp0_cookies, data=burp0_data)
    result = json.loads(r.text)

    response = {
        "word_count": result["words_count"],
        "piter_lamjin_index": 100 - float(result["percent"]),
        "matches": result["matches"]
    }

    return response

def load_data():
    # Modify this function to load data if needed
    pass

def tokenize_data(text_data):
    tokenizer.fit_on_texts(text_data)

def detect_ai(input_text):
    sequence = pad_sequences(
        tokenizer.texts_to_sequences([input_text]),
        maxlen=100
    )
    prediction = model.predict(sequence)[0][0]

    if prediction > 0.5:
        result = "Message classified as AI."
    else:
        result = "Message is classified as human."

    return result

@app.route('/anti_plagiarism', methods=['POST'])
def anti_plagiarism():
    input_text = request.json['text']
    result = detect_plagiarism(input_text)
    return jsonify(result)

@app.route('/ai_detection', methods=['POST'])
def ai_detection():
    input_text = request.json['text']
    result = detect_ai(input_text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
