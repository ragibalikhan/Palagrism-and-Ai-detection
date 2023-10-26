"""
Training module for AI text detection

Author: Piter
"""
# pylint: disable=import-error
import pandas as pd
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import requests
import json
import pyfiglet

def load_data():
    """Load dataset from CSV file and split into training and testing sets."""

    dataf = pd.read_csv(
        './datasets/labeled_data.csv'
    )
    trains_text, tests_text, trains_labels, tests_labels = train_test_split(
        dataf['message'],
        dataf['class'],
        test_size=0.2
    )

    return trains_text, tests_text, trains_labels, tests_labels


def tokenize_data(trained_text, tested_text):
    """Tokenize text data using Tokenizer from Keras and pad sequences to a fixed length."""

    tokenizer_data = Tokenizer(
        num_words=5000
    )
    tokenizer_data.fit_on_texts(
        trained_text
    )
    trained_sequences = pad_sequences(
        tokenizer_data.texts_to_sequences(trained_text),
        maxlen=100
    )
    tested_sequences = pad_sequences(
        tokenizer_data.texts_to_sequences(tested_text),
        maxlen=100
    )

    return tokenizer_data, trained_sequences, tested_sequences


def define_model():
    """Define a convolutional neural network model using Keras."""

    input_layer = Input(
        shape=(100,)
    )
    embedding_layer = Embedding(
        input_dim=5000,
        output_dim=50
    )(input_layer)
    conv_layer = Conv1D(
        filters=128,
        kernel_size=5,
        activation='relu'
    )(embedding_layer)
    pooling_layer = MaxPooling1D(pool_size=5)(conv_layer)
    flatten_layer = Flatten()(pooling_layer)
    output_layer = Dense(
        units=1,
        activation='sigmoid'
    )(flatten_layer)
    model_defined = Model(
        inputs=input_layer,
        outputs=output_layer
    )
    model_defined.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    return model_defined


def train_model(model_training,sequences_training,labels_training,sequences_testing,labels_testing):
    """Train the convolutional neural network model."""

    model_training.fit(
        sequences_training,
        labels_training,
        epochs=1,
        batch_size=1, # Works better, dont be a puss
        validation_data=(sequences_testing, labels_testing)
    )


def save_model(saved_model):
    """Save the trained convolutional neural network model to a file."""

    saved_model.save(
        'Models/ai_detection_model.h5'
    )


def classify_input(model_classify, tokenizer_classify):
    """Classify a text string input as AI or human using the trained model and tokenizer."""

    # while True:
    user_input = input(
        'Enter a text string to Detection AI generated.(Please enter the same text.) '
    )

    if user_input.lower() == 'exit':
        exit()

    # if user_input.lower() == '':
    #     continue

    sequence = pad_sequences(
        tokenizer_classify.texts_to_sequences([user_input]), 
        maxlen=100
    )
    prediction = model_classify.predict(sequence)[0][0]

    if prediction > 0.5:
        print("Message classified as AI.")
    else:
        print("Message is classified as human.")

if __name__ == '__main__':
    train_text, test_text, train_labels, test_labels = load_data()
    tokenizer, train_sequences, test_sequences = tokenize_data(
        train_text,
        test_text
    )
    model = define_model()
    # train_model(model,
    #             train_sequences,
    #             train_labels,
    #             test_sequences,
    #             test_labels
    # )
    # save_model(
    #     model
    # )
    

    banner = pyfiglet.figlet_format("Pieter Lamjin")

    print(banner)

    text_to_check = input("[?] Input text to check anti Plagiarism with Pieter Lamjin > ")

    # prompt = f"Check for plagiarism in the following text: \n{text_to_check}\n\nAI-generated text:\n"

    # generated_text = generate_text(prompt)

    # print(f"[AI-generated text] {generated_text}\n")


    burp0_url = "https://papersowl.com:443/plagiarism-checker-send-data"

    burp0_cookies = {"PHPSESSID": "qjc72e3vvacbtn4jd1af1k5qn1", "first_interaction_user": "%7B%22referrer%22%3A%22https%3A%5C%2F%5C%2Fwww.google.com%5C%2F%22%2C%22internal_url%22%3A%22%5C%2Ffree-plagiarism-checker%22%2C%22utm_source%22%3Anull%2C%22utm_medium%22%3Anull%2C%22utm_campaign%22%3Anull%2C%22utm_content%22%3Anull%2C%22utm_term%22%3Anull%2C%22gclid%22%3Anull%2C%22msclkid%22%3Anull%2C%22adgroupid%22%3Anull%2C%22targetid%22%3Anull%2C%22appsflyer_id%22%3Anull%2C%22appsflyer_cuid%22%3Anull%2C%22cta_btn%22%3Anull%7D", "first_interaction_order": "%7B%22referrer%22%3A%22https%3A%5C%2F%5C%2Fwww.google.com%5C%2F%22%2C%22internal_url%22%3A%22%5C%2Ffree-plagiarism-checker%22%2C%22utm_source%22%3Anull%2C%22utm_medium%22%3Anull%2C%22utm_campaign%22%3Anull%2C%22utm_content%22%3Anull%2C%22utm_term%22%3Anull%2C%22gclid%22%3Anull%2C%22msclkid%22%3Anull%2C%22adgroupid%22%3Anull%2C%22targetid%22%3Anull%2C%22appsflyer_id%22%3Anull%2C%22appsflyer_cuid%22%3Anull%2C%22cta_btn%22%3Anull%7D", "affiliate_user": "a%3A3%3A%7Bs%3A9%3A%22affiliate%22%3Bs%3A9%3A%22papersowl%22%3Bs%3A6%3A%22medium%22%3Bs%3A9%3A%22papersowl%22%3Bs%3A8%3A%22campaign%22%3Bs%3A9%3A%22papersowl%22%3B%7D", "sbjs_migrations": "1418474375998%3D1", "sbjs_current_add": "fd%3D2022-05-24%2019%3A01%3A22%7C%7C%7Cep%3Dhttps%3A%2F%2Fpapersowl.com%2Ffree-plagiarism-checker%7C%7C%7Crf%3Dhttps%3A%2F%2Fwww.google.com%2F", "sbjs_first_add": "fd%3D2022-05-24%2019%3A01%3A22%7C%7C%7Cep%3Dhttps%3A%2F%2Fpapersowl.com%2Ffree-plagiarism-checker%7C%7C%7Crf%3Dhttps%3A%2F%2Fwww.google.com%2F", "sbjs_current": "typ%3Dorganic%7C%7C%7Csrc%3Dgoogle%7C%7C%7Cmdm%3Dorganic%7C%7C%7Ccmp%3D%28none%29%7C%7C%7Ccnt%3D%28none%29%7C%7C%7Ctrm%3D%28none%29", "sbjs_first": "typ%3Dorganic%7C%7C%7Csrc%3Dgoogle%7C%7C%7Cmdm%3Dorganic%7C%7C%7Ccmp%3D%28none%29%7C%7C%7Ccnt%3D%28none%29%7C%7C%7Ctrm%3D%28none%29", "sbjs_udata": "vst%3D1%7C%7C%7Cuip%3D%28none%29%7C%7C%7Cuag%3DMozilla%2F5.0%20%28Windows%20NT%206.3%3B%20Win64%3B%20x64%3B%20rv%3A100.0%29%20Gecko%2F20100101%20Firefox%2F100.0", "sbjs_session": "pgs%3D1%7C%7C%7Ccpg%3Dhttps%3A%2F%2Fpapersowl.com%2Ffree-plagiarism-checker", "_ga_788D7MTZB4": "GS1.1.1653411683.1.0.1653411683.0", "_ga": "GA1.1.1828699233.1653411683", "trustedsite_visit": "1", "trustedsite_tm_float_seen": "1", "AppleBannercookie_hide_header_banner": "1", "COOKIE_PLAGIARISM_CHECKER_TERMS": "1", "plagiarism_checker_progress_state": "1"}

    burp0_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0", "Accept": "*/*", "Accept-Language": "ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3", "Accept-Encoding": "gzip, deflate", "Referer": "https://papersowl.com/free-plagiarism-checker", "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8", "X-Requested-With": "XMLHttpRequest", "Origin": "https://papersowl.com", "Dnt": "1", "Sec-Fetch-Dest": "empty", "Sec-Fetch-Mode": "no-cors", "Sec-Fetch-Site": "same-origin", "Pragma": "no-cache", "Cache-Control": "no-cache", "Te": "trailers", "Connection": "close"}

    burp0_data = {"is_free": "false", "plagchecker_locale": "en", "product_paper_type": "1", "title": '', "text": str(text_to_check)}

    r = requests.post(burp0_url, headers=burp0_headers, cookies=burp0_cookies, data=burp0_data)

    result = json.loads(r.text)

    print("\n[!] Word count : " + str(result["words_count"]))
    print("\n[!] Piter Lamjin index : " + str(100 - float(result["percent"])))
    print("\n[!] Matches : " + str(result["matches"]))


    classify_input(
        model,
        tokenizer
    )
