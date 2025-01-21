import os
from train import extract_mfcc_features
import joblib

def analyze_audio(input_audio_path):
    model_filename = "model.pkl"
    scaler_filename = "scaler.pkl"
    svm_classifier = joblib.load(model_filename)
    scaler = joblib.load(scaler_filename)

    if not os.path.exists(input_audio_path):
        print("Error: The specified file does not exist.")
    elif not input_audio_path.lower().endswith(".wav"):
        print("Error: The specified file is not a .wav file.")

    mfcc_features = extract_mfcc_features(input_audio_path)

    if mfcc_features is not None:
        mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))
        prediction = svm_classifier.predict(mfcc_features_scaled)
        os.remove(input_audio_path)
        if prediction[0] == 0:
            return "The input audio is classified as real."
        else:
            return "The input audio is classified as fake."
    else:
        return "Error: Unable to process the input audio."

if __name__ == "__main__":
    user_input_file = input("Enter the path of the .wav file to analyze: ")
    result = analyze_audio(user_input_file)
    print(result)