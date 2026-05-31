import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    LSTM,
    Reshape
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from utils import extract_features, EMOTIONS
import librosa.display

# ======================================
# Dataset Path
# ======================================

DATASET_PATH = "data"

# ======================================
# Load Dataset
# ======================================

features = []
labels = []

print("\nLoading dataset...\n")

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor)

    if not os.path.isdir(actor_path):
        continue

    for file in tqdm(os.listdir(actor_path)):

        file_path = os.path.join(actor_path, file)

        try:
            emotion_code = file.split("-")[2]

            if emotion_code in EMOTIONS:

                emotion = EMOTIONS[emotion_code]

                feature = extract_features(file_path)

                if feature is not None:
                    features.append(feature)
                    labels.append(emotion)

        except:
            continue

# ======================================
# Convert to Arrays
# ======================================

X = np.array(features)
y = np.array(labels)

print("\nDataset Loaded")
print("Features Shape:", X.shape)

# ======================================
# Encode Labels
# ======================================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

joblib.dump(encoder, "label_encoder.pkl")

y_categorical = to_categorical(y_encoded)

# ======================================
# Normalize Features
# ======================================

scaler = StandardScaler()

X = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

# ======================================
# Train-Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_categorical
)

# ======================================
# CNN Input Shape
# ======================================

X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

# ======================================
# CNN Model
# ======================================

cnn_model = Sequential([

    Conv1D(
        64,
        kernel_size=3,
        activation='relu',
        input_shape=(X_train_cnn.shape[1], 1)
    ),

    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(
        128,
        kernel_size=3,
        activation='relu'
    ),

    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),

    Dense(256, activation='relu'),
    Dropout(0.4),

    Dense(y_categorical.shape[1], activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.summary()

# ======================================
# Callbacks
# ======================================

callbacks = [

    EarlyStopping(
        patience=10,
        restore_best_weights=True
    ),

    ReduceLROnPlateau(
        patience=5,
        factor=0.5
    ),

    ModelCheckpoint(
        "models/cnn_emotion_model.h5",
        save_best_only=True
    )
]

# ======================================
# Train CNN
# ======================================

history = cnn_model.fit(
    X_train_cnn,
    y_train,
    validation_data=(X_test_cnn, y_test),
    epochs=50,
    batch_size=32,
    callbacks=callbacks
)

# ======================================
# Evaluate
# ======================================

loss, accuracy = cnn_model.evaluate(X_test_cnn, y_test)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# ======================================
# Predictions
# ======================================

y_pred = cnn_model.predict(X_test_cnn)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# ======================================
# Classification Report
# ======================================

print("\nClassification Report:\n")

print(classification_report(
    y_true,
    y_pred_classes,
    target_names=encoder.classes_
))

# ======================================
# Confusion Matrix
# ======================================

cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

os.makedirs("outputs", exist_ok=True)

plt.savefig("outputs/confusion_matrix.png")

# ======================================
# Training Curves
# ======================================

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(['Train', 'Validation'])

# Loss
plt.subplot(1, 2, 2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(['Train', 'Validation'])

plt.tight_layout()

plt.savefig("outputs/training_curves.png")

print("\nTraining completed successfully!")

# ======================================
# Emotion Distribution
# ======================================

plt.figure(figsize=(10, 6))

emotion_counts = pd.Series(labels).value_counts()

sns.barplot(
    x=emotion_counts.index,
    y=emotion_counts.values
)

plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "outputs/emotion_distribution.png",
    dpi=300,
    bbox_inches='tight'
)

plt.show()

# ======================================
# MFCC Visualization
# ======================================

sample_audio_path = os.path.join(
    DATASET_PATH,
    "Actor_01",
    os.listdir(os.path.join(DATASET_PATH, "Actor_01"))[0]
)

audio, sr = librosa.load(sample_audio_path)

mfccs = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

plt.figure(figsize=(12, 6))

librosa.display.specshow(
    mfccs,
    x_axis='time',
    sr=sr
)

plt.colorbar()

plt.title("MFCC Feature Visualization")

plt.tight_layout()

plt.savefig(
    "outputs/mfcc_visualization.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# ======================================
# Model Comparison
# ======================================

models = ['CNN']
accuracies = [float(accuracy * 100)]

plt.figure(figsize=(8, 5))

sns.barplot(
    x=models,
    y=accuracies
)

plt.ylabel("Accuracy (%)")

plt.title("Model Comparison")

plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    "outputs/model_comparison.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()