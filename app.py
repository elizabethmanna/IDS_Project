from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os  # ✅ Needed for environment variable PORT access

app = Flask(__name__)

# Load scaler and threshold globally
scaler = joblib.load("scaler.pkl")
threshold = 0.041812  # Adjust if needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Lazy load the model to save memory
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

# ✅ Ensure correct host and port for Render deployment
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render provides a dynamic port
    app.run(host='0.0.0.0', port=port)
