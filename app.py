import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define feature names
feature_names = ['PM2.5','NO','NO2','NOx','NH3','SO2','CO','Ozone','RH','WS','WD','BP','AT']

# Streamlit app title
st.title("PM2.5 Prediction with XGBoost")

# Instructions
st.write("Enter values for 3 time steps, each with 13 features (air quality and meteorological variables).")

# Create input fields for 3 time steps, each with 13 named features
features = []
for i in range(3):
    st.subheader(f"Time Step {i+1}")
    row = []
    cols = st.columns(4)  # Arrange inputs in 4 columns
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
        # Validate input shape
        if input_data.shape != (3, 13):
            st.error("Input must be a 3x13 array (3 time steps, 13 features).")
        else:
            # Scale each time step's 13 features separately
            scaled_features = []
            for t in range(3):
                time_step = input_data[t].reshape(1, -1)  # Shape: (1, 13)
                scaled_time_step = scaler.transform(time_step)  # Scale 13 features
                scaled_features.append(scaled_time_step)
            
            # Concatenate and flatten scaled features
            input_flat = np.concatenate(scaled_features, axis=1)  # Shape: (1, 39)
            
            # Make prediction
            prediction = model.predict(input_flat)[0]

            # Display prediction
            st.success(f"Predicted PM2.5: {prediction:.4f} µg/m³")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Optional: Display input data for verification
if st.checkbox("Show input data"):
    st.write("Input Data (3x13):")
    st.write(pd.DataFrame(input_data, columns=feature_names))