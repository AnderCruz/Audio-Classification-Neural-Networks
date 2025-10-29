import numpy as np
import tensorflow as tf
from scipy.signal import resample
import streamlit as st
import tensorflow_hub as hub
import librosa
import gdown
import os
import tempfile

# Configuração da página com ícone
st.set_page_config(
    page_title="Classificador de Áudio",
    page_icon="🎵",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = "audio_model.keras"
    url = "https://drive.google.com/uc?id=1-LjLpaLOfivA145KYU5sX-JssfJEwNiH"

    if not os.path.exists(model_path):
        st.info("📥 Baixando modelo...")
        gdown.download(url, model_path, quiet=False)
        st.success("✅ Download concluído!")

    model = tf.keras.models.load_model(model_path)
    st.success("✅ Modelo carregado com sucesso!")
    return model

@st.cache_resource
def load_yamnet_model():
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    return yamnet_model

def load_audio(file_path):
    waveform, sample_rate = librosa.load(file_path, sr=None)
    return waveform, sample_rate

def process_and_extract_embeddings(waveform, sample_rate, max_length=16000):
    def scipy_resample(wav, sample_rate):
        if sample_rate != 16000:
            wav = resample(wav, int(16000 / sample_rate * len(wav)))
        return wav

    wav = tf.py_function(scipy_resample, [waveform, sample_rate], tf.float32)
    
    audio_length = tf.shape(wav)[0]
    if audio_length > max_length:
        wav = wav[:max_length]
    else:
        pad_length = max_length - audio_length
        wav = tf.pad(wav, [[0, pad_length]], "CONSTANT")
    
    wav = tf.reshape(wav, [max_length])
    
    yamnet_model = load_yamnet_model()
    _, embeddings, _ = yamnet_model(wav)
    return embeddings

def predict_audio_class(file_path, model, class_map):
    if hasattr(file_path, "read"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(file_path.read())
            file_path = tmp.name

    waveform, sample_rate = load_audio(file_path)
    embeddings = process_and_extract_embeddings(waveform, sample_rate)
    predictions = model.predict(embeddings)
    final_prediction = np.mean(predictions, axis=0)
    predicted_class_index = np.argmax(final_prediction)
    predicted_class_name = class_map[predicted_class_index]
    return predicted_class_name

def main():
    st.title("Classificador de Áudio 🎶")
    st.write("Carregue um arquivo de áudio para classificar")

    uploaded_file = st.file_uploader("Carregar arquivo de áudio", type=["mp3"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/mp3")

        mapeamento = {'dog': 0, 'door_wood_creaks': 1, 'glass_breaking': 2}
        mapeamento_inverso = {v: k for k, v in mapeamento.items()}

        model = load_model()
        predicted_class = predict_audio_class(uploaded_file, model, mapeamento_inverso)

        # Classe prevista destacada com caixa colorida e centralizada
        st.markdown(
            f"""
            <div style="
                background-color:#4CAF50;
                padding:20px;
                border-radius:10px;
                text-align:center;
                color:white;
                font-size:24px;
                font-weight:bold;
            ">
                Classe Prevista: {predicted_class.upper()}
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
