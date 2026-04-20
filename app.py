import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import joblib
import os
import requests
import tempfile

from model import TransformerEncoder

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Deepfake Voice Detection",
    page_icon="🎙️",
    layout="centered"
)

# --- CONSTANTS ---
SR = 16000
DURATION = 4.0
N_MFCC = 40
SAMPLES_PER_TRACK = int(SR * DURATION)
MODEL_PATH = "results/best_transformer_model.keras"  # Using .keras as per existing structure
SCALER_PATH = "processed_data/scaler.pkl"

# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_assets():
    import urllib.request
    # Bypass Streamlit Git LFS Pointer Bug and corrupted downloads
    url = "https://github.com/Naveen-star-1/Deepfake_detection/raw/main/results/best_transformer_model.keras"
    
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 50_000_000:
        print("Model missing or corrupted. Starting safe download...", flush=True)
        st.warning("Downloading massive 95MB deepfake model... This will take ~30 seconds.")
        
        if os.path.exists(MODEL_PATH):
            try:
                os.remove(MODEL_PATH)
            except Exception as e:
                print(f"Warning: Could not delete old file: {e}", flush=True)
                
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        try:
            urllib.request.urlretrieve(url, MODEL_PATH)
        except Exception as e:
            raise FileNotFoundError(f"Safe download failed: {e}")
            
        final_size = os.path.getsize(MODEL_PATH)
        print(f"Download finished. File size: {final_size} bytes", flush=True)
        
        if final_size < 50_000_000:
            os.remove(MODEL_PATH)
            raise ValueError(f"Download integrity verification failed. Expected >50MB, got {final_size} bytes. GitHub might be blocking the download. Please reboot the app from Streamlit settings.")
            
        st.success("Download completed & verified!")
            
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}.")
        
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"TransformerEncoder": TransformerEncoder})
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# --- PREPROCESSING ---
def preprocess_audio(file_path, scaler):
    y, _ = librosa.load(file_path, sr=SR)
    
    # 1. Normalize amplitude
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
        
    # 2. Trim/Pad to 4 seconds
    if len(y) > SAMPLES_PER_TRACK:
        y = y[:SAMPLES_PER_TRACK]
    else:
        y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)), mode='constant')
        
    # 3. Extract MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    mfcc = mfcc.T
    
    # 4. Standardize features
    mfcc_flat = mfcc.reshape(-1, mfcc.shape[1])
    mfcc_scaled = scaler.transform(mfcc_flat)
    mfcc_scaled = mfcc_scaled.reshape(1, mfcc.shape[0], mfcc.shape[1])
    
    # 5. Reshape to match model input shape -> (batch, sequence, features, channels)
    X_input = np.expand_dims(mfcc_scaled, axis=-1)
    
    return X_input

# --- MAIN APP ---
def main():
    st.title("🎙️ Deepfake Voice Detection System")
    st.markdown("Upload an audio snippet below to verify if it is **REAL** or **AI-GENERATED (FAKE)**.")
    st.divider()

    try:
        model, scaler = load_assets()
    except Exception as e:
        st.error(f"Error loading system assets: {e}")
        return

    st.subheader("Upload Audio")
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        if st.button("Predict Audio Authenticity", type="primary", use_container_width=True):
            with st.spinner("Analyzing audio..."):
                try:
                    # Preprocess and Predict: Save to temporary file to avoid librosa OS errors with Streamlit UploadedFile
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                        
                    try:
                        X_input = preprocess_audio(tmp_path, scaler)
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                            
                    prediction = model.predict(X_input, verbose=0)
                    score = float(prediction[0][0])
                    
                    prob_real = score
                    prob_fake = 1.0 - prob_real
                    threshold = 0.5
                    
                    st.divider()
                    st.subheader("Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    if prob_real >= threshold:
                        st.success("✅ **VERDICT: GENUINE VOICE (REAL)**")
                        with col1:
                            st.metric("Confidence (REAL)", f"{prob_real * 100:.2f}%")
                        with col2:
                            st.metric("P(fake)", f"{prob_fake * 100:.2f}%")
                        st.progress(prob_real)
                    else:
                        st.error("🚨 **VERDICT: AI-GENERATED (FAKE)**")
                        with col1:
                            st.metric("Confidence (FAKE)", f"{prob_fake * 100:.2f}%")
                        with col2:
                            st.metric("P(real)", f"{prob_real * 100:.2f}%")
                        st.progress(prob_fake)
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
