# Audio Classification Model

This repository contains a project to classify audio files as either real or faked using MFCC (Mel-frequency cepstral coefficients) features and a neural network model built with TensorFlow and Keras.

## 📂 Dataset

The dataset should be organized into two directories:

- **dataset/real**: Contains real audio samples in `.wav` format.
- **dataset/fake**: Contains faked audio samples in `.wav` format.

## 🛠️ Setup

### Prerequisites

You can install the required packages using pip in env:

```bash
pip install numpy tensorflow keras librosa scikit-learn
```

### Project Structure

```
├── dataset
│   ├── real
│   │   └── *.wav
│   └── fake
│       └── *.wav
├── model.py
├── model.h5
└── README.md
```

## 🚀 Getting Started

### 1. Data Preparation

Ensure your dataset is correctly placed in the `dataset/real` and `dataset/fake` directories. Each `.wav` file will be processed to extract MFCC features.

### 2. Training the Model

Run the `model.py` script to train the model:

```bash
python model.py
```

This script performs the following steps:

1. **Extract MFCC Features**: From the audio files in the dataset.
2. **Create Dataset**: Combine the features and labels for real and faked audio.
3. **Train the Model**: Train a neural network model on the extracted features.
4. **Save the Model**: The trained model is saved as `model.h5`.

### 3. Analyzing New Audio Files

After training the model, you can use it to classify new audio files:

When prompted, enter the path of the `.wav` file you want to analyze. The model will predict whether the audio is real or faked.

## 📊 Evaluation

The model's performance is evaluated using accuracy and a confusion matrix, which are printed during training:

```
Accuracy: 0.95
```

## 🔧 Functions Overview

- **extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512)**: Extracts MFCC features from the given audio file.
- **create_dataset(directory, label)**: Creates a dataset by extracting MFCC features from all `.wav` files in the given directory.
- **build_model(input_shape)**: Builds a neural network model with the specified input shape.
- **train_model(X, y)**: Trains the neural network model on the provided features and labels.
- **analyze_audio(input_audio_path)**: Analyzes a given audio file to predict whether it is real or faked.

## 📝 Notes

- Ensure the audio files are in `.wav` format and 22500hH.
- Adjust the parameters such as `n_mfcc`, `n_fft`, and `hop_length` if necessary to better suit your data.
