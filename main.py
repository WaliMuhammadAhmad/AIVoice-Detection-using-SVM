import os
import librosa
import sklearn
import joblib
import numpy as np

# It is recommended to run this before training the model to check the version

# Check versions of the imported libraries
print("Libraries Versions:")
print("librosa version:", librosa.__version__)
print("numpy version:", np.__version__)
print("scikit-learn version:", sklearn.__version__)
print("joblib version:", joblib.__version__)

# Print a message indicating that the files are ready to use
print("\nAll required files are imported and ready to use!")
