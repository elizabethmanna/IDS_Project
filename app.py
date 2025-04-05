
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the scaler
try:
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    scaler = None
    print("Error: scaler.pkl not found.")

threshold = 0.041812  # Adjust as needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if scaler is None:
            return "Scaler file not found on server.", 500

        model = load_model("autoencoder_model.h5", compile=False)

        if 'csv_file' not in request.files:
            return "No file part", 400

        file = request.files['csv_file']
        if file.filename == '':
            return "No selected file", 400

        df = pd.read_csv(file)
        if df.empty:
            return "Uploaded CSV is empty or invalid", 400

        scaled = scaler.transform(df)
        recon = model.predict(scaled)
        mse = np.mean(np.square(scaled - recon), axis=1)
        results = ['Anomaly' if err > threshold else 'Normal' for err in mse]
        df['Prediction'] = results

        return render_template('index.html', tables=[df.to_html(classes='data')], results=results)

    except FileNotFoundError as fnf_error:
        return f"Model file not found: {str(fnf_error)}", 500
    except Exception as e:
        print("ERROR:", e)
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
