from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os  # ✅ Added to access environment variables for port

app = Flask(__name__)

# Load only the scaler and threshold globally
scaler = joblib.load("scaler.pkl")
threshold = 0.041812  # Adjust if needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lazy load the model here instead of globally
    from tensorflow.keras.models import load_model
    model = load_model("autoencoder_model.h5", compile=False)

    if 'csv_file' in request.files:
        df = pd.read_csv(request.files['csv_file'])
        scaled = scaler.transform(df)
        recon = model.predict(scaled)
        mse = np.mean(np.square(scaled - recon), axis=1)
        results = ['Anomaly' if err > threshold else 'Normal' for err in mse]
        df['Prediction'] = results
        return render_template('index.html', tables=[df.to_html(classes=df['Prediction'])], results=results)

    return "No file uploaded", 400

# ✅ Required for Render: bind to 0.0.0.0 and use PORT env variable
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Render provides PORT
    app.run(host='0.0.0.0', port=port)
