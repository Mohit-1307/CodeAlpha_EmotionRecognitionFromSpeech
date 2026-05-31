import librosa
import numpy as np

# Emotion Labels
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def extract_features(file_path, duration=3, offset=0.5):
    """
    Extract audio features from speech file
    """

    try:
        audio, sample_rate = librosa.load(
            file_path,
            duration=duration,
            offset=offset
        )

        # =========================
        # MFCC
        # =========================
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        mfcc_mean = np.mean(mfcc.T, axis=0)
        mfcc_std = np.std(mfcc.T, axis=0)

        # =========================
        # Delta MFCC
        # =========================
        delta_mfcc = librosa.feature.delta(mfcc)

        delta_mean = np.mean(delta_mfcc.T, axis=0)
        delta_std = np.std(delta_mfcc.T, axis=0)

        # =========================
        # Chroma
        # =========================
        stft = np.abs(librosa.stft(audio))

        chroma = librosa.feature.chroma_stft(
            S=stft,
            sr=sample_rate
        )

        chroma_mean = np.mean(chroma.T, axis=0)

        # =========================
        # Mel Spectrogram
        # =========================
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate
        )

        mel_mean = np.mean(mel.T, axis=0)

        # =========================
        # ZCR
        # =========================
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr)

        # =========================
        # RMS Energy
        # =========================
        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms)

        # =========================
        # Final Feature Vector
        # =========================
        features = np.hstack([
            mfcc_mean,
            mfcc_std,
            delta_mean,
            delta_std,
            chroma_mean,
            mel_mean,
            zcr_mean,
            rms_mean
        ])

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None