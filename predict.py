import numpy as np
import joblib
from tensorflow.keras.models import load_model
from utils import extract_features

# ======================================
# Load Model
# ======================================

model = load_model("models/cnn_emotion_model.h5")

encoder = joblib.load("label_encoder.pkl")

scaler = joblib.load("scaler.pkl")

# ======================================
# Predict Function
# ======================================

def predict_emotion(audio_path):

    features = extract_features(audio_path)

    features = scaler.transform([features])

    features = features[..., np.newaxis]

    prediction = model.predict(features)

    predicted_class = np.argmax(prediction)

    emotion = encoder.inverse_transform([predicted_class])[0]

    confidence = np.max(prediction) * 100

    return emotion, confidence


# ======================================
# Example
# ======================================

if __name__ == "__main__":

    path = input("Enter audio file path: ")

    emotion, confidence = predict_emotion(path)

    print(f"\nPredicted Emotion: {emotion}")
    print(f"Confidence: {confidence:.2f}%")