from flask import Flask, render_template, request
import os
import time
from keras.models import model_from_json
from pickle import dump,load
import librosa
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))


# load json and create model
json_file = open('static/models/ann750less36.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("static/models/ann750less36.h5")
print("Loaded model from disk")

scaler = load(open('static/models/scaler750less36.pkl', 'rb'))

print("Model and scalar loaded successfully")

def perform_predict_less(filename):

    y, sr = librosa.load(filename,duration=.75)
    rmse = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    # Added features 
    poly_features=librosa.feature.poly_features(y=y, sr=sr)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    tempogram = librosa.beat.tempogram(y=y,sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tonnetz_f=librosa.feature.tonnetz(y=y, sr=sr)
    #tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sample_rate)

    to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} {np.mean(poly_features)} {np.mean(chroma_cqt)} {np.mean(chroma_cens)} {np.mean(tempogram)} {np.mean(pitches)} {np.mean(onset_env)} {np.mean(tonnetz_f)}'    

    for e in mfcc:
        to_append += f' {np.mean(e)}'

    d = np.fromstring(to_append, sep=' ')
    d1=[]
    d1.append(d)
    d_S = scaler.transform(d1)   
    p = loaded_model.predict_classes(d_S)
    return p


@app.route('/',methods = ['POST','GET'])
def index():
    if request.method == "POST":
    
        path = os.path.abspath(basedir+'/static/audio/')

        if not os.path.exists(path):
            os.makedirs(path)
        app.config['UPLOAD_FOLDER'] = path

        if 'audio_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        if request.method == 'POST':
            file = request.files['audio_file']
            f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

            # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
            file.save(f)
            file_path = "static/audio/"+file.filename
            #start = time.clock() #start time
            pred = perform_predict_less(file_path)[0]
            #required_time = time.clock()-start #end time

            

            if pred == 0:
                pred_label = "Live"
            else:
                pred_label = "Voice"
            print("Prediction: ",pred)
            #print("Required time: ",required_time)
            return render_template("index.html",pred = pred_label,required_time = 5,file_path=file_path)



    return render_template("index.html")

@app.route('/perform_predict', methods=['GET', 'POST'])
def perform_predict():


    return "this is result route"

if __name__ == '__main__':
    app.run()