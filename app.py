import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Define the LSTM model architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size=13, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 25),
            nn.ReLU(),
            nn.Linear(25, output_size)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Load the trained model weights
model = LSTMModel()
model.load_state_dict(torch.load('lstm_air_quality_pytorch.pt', map_location=torch.device('cpu')))
model.eval()

# Load the scaler
import joblib
scaler = joblib.load('scaler.pkl')

# Define feature names
feature_names = ['PM2.5','NO','NO2','NOx','NH3','SO2','CO','Ozone','RH','WS','WD','BP','AT']

# Streamlit UI
st.title("PM2.5 Prediction with LSTM Model")

st.write("Enter values for 3 time steps, each with 13 features (air quality and meteorological variables).")

features = []
for i in range(3):
    st.subheader(f"Time Step {i+1}")
    row = []
    cols = st.columns(4)
    for j, feature in enumerate(feature_names):
        with cols[j % 4]:
            value = st.number_input(f"{feature} (µg/m³ or unit)", value=0.0, format="%.4f", key=f"t{i}_f{j}")
            row.append(value)
    features.append(row)

# Convert features to numpy array
input_data = np.array(features, dtype=np.float32)  # Shape: (3, 13)

# Predict button
if st.button("Predict PM2.5"):
    try:
        if input_data.shape != (3, 13):
            st.error("Input must be a 3x13 array (3 time steps, 13 features).")
        else:
            # Scale each time step independently
            scaled_features = []
            for t in range(3):
                time_step = input_data[t].reshape(1, -1)  # Shape: (1, 13)
                scaled_time_step = scaler.transform(time_step)
                scaled_features.append(scaled_time_step)

            input_scaled = np.stack(scaled_features, axis=1)  # Shape: (1, 3, 13)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

            # Make prediction
            with torch.no_grad():
                prediction = model(input_tensor).item()

            st.success(f"Predicted PM2.5: {prediction:.4f} µg/m³")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Show input data for verification
if st.checkbox("Show input data"):
    st.write("Input Data (3x13):")
    st.write(pd.DataFrame(input_data, columns=feature_names))
