import os
import pickle
import librosa
import numpy as np
import tensorflow as tf
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Load the model from the .pkl file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the CSV file with class labels
audiofiles = pd.read_csv(r'C:\Users\rivva\gunshot_detect\gunshot-detection\UrbanSound8K.csv')

# Extract unique class labels based on classID and map them to the corresponding names
classes = dict(zip(audiofiles['classID'], audiofiles['class']))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return "No file uploaded" , 400 
    audio_file = request.files['audio_file']

    
    temp_file_path = 'temp.wav'
    audio_file.save(temp_file_path)

    audiodata, sample_rate = librosa.load(temp_file_path, res_type='kaiser_fast')
    mels = np.mean(librosa.feature.melspectrogram(y=audiodata, sr=sample_rate).T, axis=0)


    X = np.array(mels).reshape(1, -1)

    classid = np.argmax(model.predict(X))

    
    predicted_class = classes[classid] if classid < len(classes) else "Unknown"


    os.remove(temp_file_path)

    return f"Predicted class: {predicted_class}"

if __name__ == '__main__':
    app.run(debug=True)
