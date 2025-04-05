from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load model and scaler
model = load_model("autoencoder_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")
threshold = 0.041812  # You can update this based on training results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'csv_file' in request.files:
        df = pd.read_csv(request.files['csv_file'])
        scaled = scaler.transform(df)
        recon = model.predict(scaled)
        mse = np.mean(np.square(scaled - recon), axis=1)
        results = ['Anomaly' if err > threshold else 'Normal' for err in mse]
        df['Prediction'] = results
        return render_template('index.html', tables=[df.to_html(classes='data', header="true")], results=results)

#  THIS LINE IS MANDATORY
if __name__ == "__main__":
    app.run(debug=True)

