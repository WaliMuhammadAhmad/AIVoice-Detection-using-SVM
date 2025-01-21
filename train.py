import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)

def create_dataset(directory, label):
    X, y = [], []
    audio_files = glob.glob(os.path.join(directory, "*.wav"))
    for audio_path in audio_files:
        mfcc_features = extract_mfcc_features(audio_path)
        if mfcc_features is not None:
            X.append(mfcc_features)
            y.append(label)
        else:
            print(f"Skipping audio file {audio_path}")

    print("Number of samples in", directory, ":", len(X))
    print("Filenames in", directory, ":", [os.path.basename(path) for path in audio_files])
    return X, y

def train_model(X, y):
    unique_classes = np.unique(y)
    print("Unique classes in y_train:", unique_classes)

    if len(unique_classes) < 2:
        raise ValueError("Atleast 2 set is required to train")

    print("Size of X:", X.shape)
    print("Size of y:", y.shape)

    class_counts = np.bincount(y)
    if np.min(class_counts) < 2:
        print("Combining both classes into one for training")
        X_train, y_train = X, y
        X_test, y_test = None, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        print("Size of X_train:", X_train.shape)
        print("Size of X_test:", X_test.shape)
        print("Size of y_train:", y_train.shape)
        print("Size of y_test:", y_test.shape)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)

        svm_classifier = SVC(kernel='linear', random_state=42)
        svm_classifier.fit(X_train_scaled, y_train)

        y_pred = svm_classifier.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        confusion_mtx = confusion_matrix(y_test, y_pred)

        print("Accuracy:", accuracy)
        print("Confusion Matrix:")
        print(confusion_mtx)
    else:
        print("Insufficient samples for stratified splitting. Combine both classes into one for training.")
        print("Train on all available data.")

        svm_classifier = SVC(kernel='linear', random_state=42)
        svm_classifier.fit(X_train_scaled, y_train)

    model_filename = "model.pkl"
    scaler_filename = "scaler.pkl"
    joblib.dump(svm_classifier, model_filename)
    joblib.dump(scaler, scaler_filename)

def main():
    real_dir = "dataset/real"
    fake_dir = "dataset/fake"

    X_real, y_real = create_dataset(real_dir, label=0)
    X_fake, y_fake = create_dataset(fake_dir, label=1)

    if len(X_real) < 2 or len(X_fake) < 2:
        print("Each class should have at least two samples for stratified splitting.")
        print("Combining both classes into one for training.")
        X = np.vstack((X_real, X_fake))
        y = np.hstack((y_real, y_fake))
    else:
        X = np.vstack((X_real, X_fake))
        y = np.hstack((y_real, y_fake))

    train_model(X, y)

if __name__ == "__main__":
    main()