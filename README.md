# AI Voice Detection

This project presents an SVM model trained to distinguish between real human and AI-generated voices.

## About the Project

The AI Voice Detection project uses Support Vector Machines (SVM) algo to classify audio samples as real human or AI-generated voices.
The trained model file can detect any type of AI-generated audio, such as Google Assistant Voice, Alexa Voice, and any other AI-generated Text-to-Speech Voice. 
This is a binary classification problem so the model takes real and fake audio as the training dataset. After training you can use predict.py to predict the voice.
The model's performance can be increased by adding and diverse dataset. 

*Note: *tensorflow implementation is also provided in the [folder](tensorflow/). The results might not be good bcz it is still under work!**

## Dataset

The dataset used for training and evaluation comprises 70 audio samples, with 35 samples each for real human and AI-generated voices.
All the audios are `.wav` file format, have *44100 Hz* sample rate and mono channel. More details are listed below:

- Real Audio Dataset
  There are 30 real audios recorded by 4 male voices reading 4 English passages. All the audio has a length of 30 seconds (15 minutes in total)
- Fake Audio Dataset
  There are 30 Text-to-Speech fake audios recorded by different male voices reading an English passage. The passage is also provided in the repository.
  All the male voices are recorded with different accents of English. All the audio has a length of 29-30 seconds (14.90 minutes in total)
- Test Dataset
  The test data set has 5 real and 5 fake audio files. The fake voices in the test dataset are Generated from a different method than the fake audios used in the Training Dataset. All the audios have a length ranging from 19-30 seconds.
  
*The dataset used to train and test this model is provided inside the repository.*

## Getting Started

To train and test the AI Voice Detection model on your local machine, follow these steps:

1. **Clone the Repository:**
   ```sh
   git clone https://github.com/WaliMuhammadAhmad/AIVoice-Detection.git
   ```

3. **Install Dependencies:**
   Install the required libraries in `requirements.txt`:
   ```sh
   pip install -r requirements.txt
   ```
**Docker config is also provided so If you are comfortable with docker then use them**
```sh
docker compose up
```
*Note: make sure you have docker installed and running!*
*dev container config is also provided so you can open vs code inside vs container using vs code Extensions (suggested)*
4. **Train the Model:**
   Run the training script to train the SVM model:
   ```sh
   python train.py
   ```

5. **Test the Model:**
   Use the trained model to predict whether an audio sample is real or fake:
   ```sh
   python predict.py
   ```
   predict.py will ask to enter the path of the *.wav file. Enter the file path for prediction.


## Acknowledgements

I would like to extend my heartfelt gratitude to the following individuals for their invaluable contribution to the dataset used in this project:

- Hassan Abbas
- Himanshu

## Contributing

Contributions are welcome! Whether it's improving model performance, or adding more data, your input is valued. Feel free to open issues or pull requests.

## Contact

For any questions or collaborations, you can reach me at [wali.muhammad.ahmad@gmail.com].
