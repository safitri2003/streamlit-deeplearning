import streamlit as st
import pickle
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Sentiment Analysis", layout="centered")

@st.cache_resource
def load_all():
    model = tf.keras.models.load_model("model_gru.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_map.pkl", "rb") as f:
        label_map = pickle.load(f)

    with open("config.pkl", "rb") as f:
        config = pickle.load(f)

    return model, tokenizer, label_map, config

model, tokenizer, label_map, config = load_all()
MAX_LEN = config["MAX_LEN"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

def predict_sentiment(text):
    text_clean = clean_text(text)
    seq = tokenizer.texts_to_sequences([text_clean])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    probs = model.predict(pad)[0]
    idx = np.argmax(probs)
    label = list(label_map.keys())[idx]
    confidence = probs[idx]

    return label, confidence, probs, text_clean


st.title("📊 Sentiment Analysis Komentar YouTube")
st.caption(
    "Model: GRU | Dataset: Komentar YouTube | "
    "Kelas: Negatif, Netral, Positif"
)

text = st.text_area("Masukkan komentar:")

if st.button("Prediksi"):
    if text.strip() == "":
        st.warning("Teks tidak boleh kosong")
    else:
        label, conf, probs, clean = predict_sentiment(text)

        st.success(f"Sentimen: **{label}**")
        st.write(f"Tingkat keyakinan model: **{conf*100:.2f}%**")

        st.subheader("Probabilitas Setiap Kelas")
        st.bar_chart({
            "Negatif": probs[0],
            "Netral": probs[1],
            "Positif": probs[2]
        })

        st.caption(f"Teks setelah preprocessing: {clean}")



