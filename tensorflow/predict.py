import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

THRESHOLD = 0.5

def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)

def analyze_audio(input_audio_path):
    model = tf.keras.models.load_model("model.keras")
    print("Model loaded...")
    
    if not os.path.exists(input_audio_path):
        print("Error: The specified file does not exist.")

    elif not input_audio_path.lower().endswith(".wav"):
        print("Error: The specified file is not a .wav file.")

    print("Analyzing audio file...")
    mfcc_features = extract_mfcc_features(input_audio_path)
    print(mfcc_features)

    if mfcc_features is not None:
        scaler = StandardScaler()
        mfcc_features_scaled = scaler.fit_transform(mfcc_features.reshape(1, -1))
        print(mfcc_features_scaled)
        prediction = model.predict(mfcc_features_scaled)
        os.remove(input_audio_path)
        print(prediction)
        if prediction[0][0] < THRESHOLD:
            print("The input audio is real.") # print also bcz return is not shown on terminal
            return "The input audio is real."
        else:
            print("The input audio is fake.")
            return "The input audio is fake."
        
    else:
        print("Error: Unable to process the input audio.")
        return "Error: Unable to process the input audio."

if __name__ == "__main__":

    input_file = input("Enter the path of the .wav file to analyze: ")
    analyze_audio(input_file)
