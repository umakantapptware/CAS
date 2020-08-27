from flask import Flask, render_template, request
import os
import time
from tensorflow.keras.models import model_from_json
from pickle import dump,load
import librosa
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))


# load json and create model
json_file = open('static/models/ann.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("static/models/ann.h5")
print("Loaded model from disk")

scaler = load(open('static/models/scaler.pkl', 'rb'))

print("Model and scalar loaded successfully")


def perform_predict_less(filename):

    y, sr = librosa.load(filename,duration=.75)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
    #to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)} {np.mean(poly_features)} {np.mean(chroma_cqt)} {np.mean(chroma_cens)} {np.mean(tempogram)} {np.mean(pitches)} {np.mean(onset_env)} {np.mean(tonnetz_f)}'    
    to_append = ''
    for e in mfcc:
        to_append += f'{np.mean(e)} '

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
            start = time.perf_counter() #start time
            pred = perform_predict_less(file_path)[0]
            required_time = time.perf_counter()-start #end time

            

            if pred == 0:
                pred_label = "Live"
            else:
                pred_label = "Voice"
            #print("Required time: ",required_time)
            return render_template("index.html",pred = pred_label,required_time = required_time,file_path=file_path)



    return render_template("index.html")

@app.route('/perform_predict', methods=['GET', 'POST'])
def perform_predict():


    return "this is result route"

if __name__ == '__main__':
    app.run(debug=True)