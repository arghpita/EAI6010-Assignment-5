from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from flask_cors import CORS  # Import CORS

# Enable CORS for all routes
app = Flask(__name__)
CORS(app)  # This will allow all domains to access the API

# Define the LSTM model class
class SunspotLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(SunspotLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Load the trained model
model = SunspotLSTM()
model.load_state_dict(torch.load('sunspot_lstm.pth', map_location=torch.device('cpu')))
model.eval()

# Load the scaler
scaler = joblib.load('scaler.pkl')

# Simple route for testing
@app.route('/')
def home():
    return "Flask app is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json["sequence"]
    if not data or len(data) != 12:
        return jsonify({"error": "Invalid input. Expecting an array of length 12."}), 400
    
    # Normalize input
    data = np.array(data).reshape(-1, 1)
    data = scaler.transform(data)
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        prediction = model(data).numpy()
    
    # Inverse transform to original scale
    prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten().tolist()
    return jsonify({"predicted_sunspot_number": prediction})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port, debug=True)
