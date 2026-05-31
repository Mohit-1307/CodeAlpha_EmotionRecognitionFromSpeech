import os
import pandas as pd

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

data = []

dataset_path = "data"

for actor_folder in os.listdir(dataset_path):
    actor_path = os.path.join(dataset_path, actor_folder)

    if os.path.isdir(actor_path):

        for file in os.listdir(actor_path):

            if file.endswith(".wav"):

                parts = file.split("-")

                emotion_code = parts[2]

                emotion = emotion_map.get(emotion_code)

                full_path = os.path.join(actor_path, file)

                data.append([full_path, emotion])

df = pd.DataFrame(data, columns=["path", "emotion"])

df.to_csv("data/labels.csv", index=False)

print("labels.csv created successfully!")
print(df.head())