import os
import pickle
import librosa
import numpy as np
import tensorflow as tf
import pandas as pd
from scipy.optimize import least_squares
from localisation import locate_gunshot, create_plotly_3d_plot, MIC_POSITIONS 
from flask import Flask, request, jsonify, render_template

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

audiofiles = pd.read_csv(r'C:\Users\rivva\gunshot_detect\gunshot-detection\UrbanSound8K.csv')

classes = dict(zip(audiofiles['classID'], audiofiles['class']))

app = Flask(__name__)




@app.route('/')
def home():
    return render_template('index2.html')

@app.route('/locate', methods=['POST'])
def locate():
    data = request.get_json()
    toa = data.get('toa')
    if not toa:
        return jsonify({'error': 'ToA values are missing'}), 400

    try:
        # Process the ToA values and determine the location
        location = locate_gunshot(toa)  # Function from your localisation module
        plot_html = create_plotly_3d_plot(location, MIC_POSITIONS)  # Function from your localisation module
        return jsonify({'location': location.tolist(), 'plot_html': plot_html})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
