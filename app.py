from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load scaler and set threshold globally
scaler = joblib.load("scaler.pkl")
threshold = 0.041812  # Update if needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = load_model("autoencoder_model.h5", compile=False)

    try:
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

    except Exception as e:
        print("ERROR:", e)
        return f"An error occurred: {str(e)}", 500

# For local dev and Render deployment
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 for local
    app.run(host='0.0.0.0', port=port, debug=True)
