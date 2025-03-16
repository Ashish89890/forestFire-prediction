from flask import Flask     
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)       # Initialize the Flask application
app=application
ridge_model = pickle.load(open('pickles/ridge.pkl', 'rb'))
scaler = pickle.load(open('pickles/scaler.pkl', 'rb'))
@app.route('/')                      # Homepage
@app.route('/predict', methods=['GET','POST'])  # This will be called from UI
def predict_():
    if request.method == 'POST':
        temp=float(request.form.get('temp'))
        rh=float(request.form.get('rh'))
        ws=float(request.form.get('ws'))
        rain=float(request.form.get('rain'))
        ffmc=float(request.form.get('ffmc'))
        dmc=float(request.form.get('dmc'))
        isi=float(request.form.get('isi'))
        classes=float(request.form.get('classes'))
        

        data = [[temp,rh,ws,rain,ffmc,dmc,isi,classes]]
        data = scaler.transform(data)
        results = ridge_model.predict(data)
        return render_template('home.html',result=results[0]) 

    else:
     return render_template('home.html')
def index():
    return render_template('index.html')        
if __name__ == '__main__':
    app.run(host="0.0.0.0")             # Start the server
