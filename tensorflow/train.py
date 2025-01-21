import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

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
    return X, y

def build_model(input_shape):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Size of X_train:", X_train.shape)
    print("Size of X_test:", X_test.shape)
    print("Size of y_train:", y_train.shape)
    print("Size of y_test:", y_test.shape)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and train the model
    model = build_model(X_train_scaled.shape[1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Save the trained model and scaler
    model.save("model.keras")  # Use the new .keras format
    print("Model saved as model.keras")

    # Evaluate the model
    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    confusion_mtx = confusion_matrix(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(confusion_mtx)

def load_and_evaluate_model(model_path, X_test, y_test):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    print("Model loaded from", model_path)

    # Recompile the model to avoid compiled metrics warning
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Evaluate the model
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    confusion_mtx = confusion_matrix(y_test, y_pred)

    print("Accuracy after loading model:", accuracy)
    print("Confusion Matrix after loading model:")
    print(confusion_mtx)

def main():
    real_dir = "dataset/real"
    fake_dir = "dataset/fake"

    X_real, y_real = create_dataset(real_dir, label=0)
    X_fake, y_fake = create_dataset(fake_dir, label=1)

    X = np.vstack((X_real, X_fake))
    y = np.hstack((y_real, y_fake))

    train_model(X, y)

    # Example of loading and evaluating the model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    load_and_evaluate_model("model.keras", X_scaled, y)

if __name__ == "__main__":
    main()