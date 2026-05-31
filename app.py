import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import pandas as pd
from predict import predict_emotion

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Advanced Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.stApp {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: white;
}

.metric-box {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# TITLE
# ==========================================

st.title("🎤 Advanced Emotion Recognition from Speech")

st.markdown("""
Deep Learning based Speech Emotion Recognition System using:

- CNN + MFCC Features
- Speech Signal Processing
- Audio Feature Visualization
- Emotion Prediction Pipeline
""")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Project Information")

st.sidebar.info("""
### Supported Emotions

- Angry
- Calm
- Disgust
- Fearful
- Happy
- Neutral
- Sad
- Surprised
""")

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload WAV Audio File",
    type=["wav"]
)

# ==========================================
# PROCESS AUDIO
# ==========================================

if uploaded_file is not None:

    # Save Temporary File
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:

        tmp.write(uploaded_file.read())

        temp_audio_path = tmp.name

    # ==========================================
    # AUDIO PLAYER
    # ==========================================

    st.subheader("Uploaded Audio")

    st.audio(temp_audio_path)

    # ==========================================
    # PREDICTION
    # ==========================================

    emotion, confidence = predict_emotion(temp_audio_path)

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"Predicted Emotion: {emotion}")

    with col2:
        st.info(f"Confidence: {confidence:.2f}%")

    # ==========================================
    # LOAD AUDIO
    # ==========================================

    audio, sr = librosa.load(temp_audio_path)

    duration = librosa.get_duration(y=audio, sr=sr)

    # ==========================================
    # AUDIO INFORMATION
    # ==========================================

    st.subheader("Audio Information")

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.metric("Sample Rate", f"{sr} Hz")

    with info_col2:
        st.metric("Duration", f"{duration:.2f} sec")

    with info_col3:
        st.metric("Samples", len(audio))

    # ==========================================
    # WAVEFORM
    # ==========================================

    st.subheader("Waveform")

    fig, ax = plt.subplots(figsize=(12, 4))

    librosa.display.waveshow(audio, sr=sr)

    plt.title("Audio Waveform")

    st.pyplot(fig)

    # ==========================================
    # MFCC
    # ==========================================

    st.subheader("MFCC Features")

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    librosa.display.specshow(
        mfcc,
        x_axis='time',
        sr=sr
    )

    plt.colorbar()

    plt.title("MFCC")

    st.pyplot(fig2)

    # ==========================================
    # MEL SPECTROGRAM
    # ==========================================

    st.subheader("Mel Spectrogram")

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig3, ax3 = plt.subplots(figsize=(12, 5))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis='time',
        y_axis='mel'
    )

    plt.colorbar(format='%+2.0f dB')

    plt.title("Mel Spectrogram")

    st.pyplot(fig3)

    # ==========================================
    # CHROMA FEATURES
    # ==========================================

    st.subheader("Chroma Features")

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr
    )

    fig4, ax4 = plt.subplots(figsize=(12, 5))

    librosa.display.specshow(
        chroma,
        x_axis='time',
        y_axis='chroma',
        sr=sr
    )

    plt.colorbar()

    plt.title("Chroma Features")

    st.pyplot(fig4)

    # ==========================================
    # SPECTRAL CONTRAST
    # ==========================================

    st.subheader("Spectral Contrast")

    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr
    )

    fig5, ax5 = plt.subplots(figsize=(12, 5))

    librosa.display.specshow(
        contrast,
        x_axis='time',
        sr=sr
    )

    plt.colorbar()

    plt.title("Spectral Contrast")

    st.pyplot(fig5)

    # ==========================================
    # TONNETZ FEATURES
    # ==========================================

    st.subheader("Tonnetz Features")

    harmonic = librosa.effects.harmonic(audio)

    tonnetz = librosa.feature.tonnetz(
        y=harmonic,
        sr=sr
    )

    fig6, ax6 = plt.subplots(figsize=(12, 5))

    librosa.display.specshow(
        tonnetz,
        x_axis='time',
        sr=sr
    )

    plt.colorbar()

    plt.title("Tonnetz Features")

    st.pyplot(fig6)

    # ==========================================
    # ZERO CROSSING RATE
    # ==========================================

    st.subheader("Zero Crossing Rate")

    zcr = librosa.feature.zero_crossing_rate(audio)[0]

    fig7, ax7 = plt.subplots(figsize=(12, 4))

    ax7.plot(zcr)

    plt.title("Zero Crossing Rate")

    st.pyplot(fig7)

    # ==========================================
    # RMS ENERGY
    # ==========================================

    st.subheader("RMS Energy")

    rms = librosa.feature.rms(y=audio)[0]

    fig8, ax8 = plt.subplots(figsize=(12, 4))

    ax8.plot(rms)

    plt.title("RMS Energy")

    st.pyplot(fig8)

    # ==========================================
    # SPECTRAL CENTROID
    # ==========================================

    st.subheader("Spectral Centroid")

    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr
    )[0]

    fig9, ax9 = plt.subplots(figsize=(12, 4))

    ax9.plot(centroid)

    plt.title("Spectral Centroid")

    st.pyplot(fig9)

    # ==========================================
    # FEATURE STATISTICS
    # ==========================================

    st.subheader("Feature Statistics")

    feature_stats = pd.DataFrame({
        "Feature": [
            "MFCC Mean",
            "Chroma Mean",
            "Mel Mean",
            "ZCR Mean",
            "RMS Mean",
            "Spectral Centroid Mean"
        ],
        "Value": [
            np.mean(mfcc),
            np.mean(chroma),
            np.mean(mel),
            np.mean(zcr),
            np.mean(rms),
            np.mean(centroid)
        ]
    })

    st.dataframe(
        feature_stats,
        use_container_width=True
    )

    # ==========================================
    # FOOTER
    # ==========================================

    st.markdown("---")

    st.markdown("""
    ### Project Pipeline

    Audio Upload
    → Feature Extraction
    → MFCC + Spectral Features
    → CNN Deep Learning Model
    → Emotion Prediction
    """)

else:

    st.warning("Please upload a WAV audio file.")