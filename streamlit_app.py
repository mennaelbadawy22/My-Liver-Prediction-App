import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder # You might need this if categorical inputs are used
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Liver Disease Prediction App", layout="centered")

# --- Load Pre-trained Model, Scaler, and Feature Names ---
try:
    best_model = joblib.load('best_liver_prediction_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_features = joblib.load('model_features.pkl') # Load the exact feature order
except FileNotFoundError:
    st.error("Model, scaler, or feature names file not found. Please ensure 'best_liver_prediction_model.pkl', 'scaler.pkl', and 'model_features.pkl' are in the same directory.")
    st.stop() # Stop the app if essential files are missing

# --- Streamlit UI ---
st.title("🏥 Liver Disease Prediction")
st.markdown("""
    This app predicts the likelihood of liver disease based on various health parameters.
    Please enter the patient's details below:
""")

# --- Input Fields for User Data (Matching your dataset columns) ---
st.header("Patient Medical Parameters")

# Define input fields based on your dataset columns
# Use the exact column names from your original dataset, even with special characters
# Ensure the order matches the order in model_features.pkl *before* feature engineering
# Let's map your actual column names to more user-friendly labels if needed
# You can refer to the columns printed by repr() in the previous debugging step.

# Example mapping (adjust if your original dataset columns have simpler names after loading)
# For the input widgets, it's often easier to use simple keys and then map to the complex names
input_data = {}

col1, col2 = st.columns(2)

with col1:
    input_data['Age of the patient'] = st.number_input("Age (years)", min_value=1, max_value=120, value=40)
    input_data['Gender of the patient'] = st.selectbox("Gender", options=['Male', 'Female'])
    input_data['Total Bilirubin'] = st.number_input("Total Bilirubin (mg/dL)", min_value=0.1, max_value=100.0, value=1.0, format="%.1f")
    input_data['Direct Bilirubin'] = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, max_value=50.0, value=0.4, format="%.1f")
    input_data['Â\xa0Alkphos Alkaline Phosphotase'] = st.number_input("Alkaline Phosphatase (ALP) (IU/L)", min_value=10, max_value=2000, value=150)

with col2:
    input_data['Â\xa0Sgpt Alamine Aminotransferase'] = st.number_input("Alamine Aminotransferase (ALT/SGPT) (IU/L)", min_value=5, max_value=1000, value=30)
    input_data['Sgot Aspartate Aminotransferase'] = st.number_input("Aspartate Aminotransferase (AST/SGOT) (IU/L)", min_value=5, max_value=1000, value=40)
    input_data['Total Protiens'] = st.number_input("Total Proteins (g/dL)", min_value=2.0, max_value=10.0, value=7.0, format="%.1f")
    input_data['Â\xa0ALB Albumin'] = st.number_input("Albumin (g/dL)", min_value=1.0, max_value=6.0, value=3.5, format="%.1f")
    input_data['A/G Ratio Albumin and Globulin Ratio'] = st.number_input("Albumin/Globulin Ratio", min_value=0.1, max_value=3.0, value=1.0, format="%.2f")


# --- Preprocessing and Feature Engineering Function ---
def preprocess_and_engineer(df):
    # Apply the same preprocessing and feature engineering steps as in your training script

    # 1. Handle categorical 'Gender of the patient' using LabelEncoder
    # Assuming 'Gender of the patient' was encoded during training: 'Male' -> 1, 'Female' -> 0 or vice versa
    # You need to ensure the LabelEncoder is fitted to both 'Male' and 'Female'
    # If it was 1 for Male and 0 for Female in your training, ensure consistency.
    # For simplicity, we can manually map it here if it's binary.
    gender_mapping = {'Male': 1, 'Female': 0} # Adjust based on your training encoding
    df['Gender of the patient'] = df['Gender of the patient'].map(gender_mapping).astype(int)

    # 2. Feature Engineering: AST_ALT_Ratio
    # Use the exact column names as defined in the training script
    df['AST_ALT_Ratio'] = df['Sgot Aspartate Aminotransferase'] / df['Â\xa0Sgpt Alamine Aminotransferase'].replace(0, 1e-6)

    # 3. Feature Engineering: Albumin_Globulin_Ratio
    # This might be redundant if 'A/G Ratio Albumin and Globulin Ratio' column already exists and is used
    # If you engineered it from 'Â\xa0ALB Albumin' and 'Total Protiens' during training, re-create it.
    # Otherwise, ensure 'A/G Ratio Albumin and Globulin Ratio' is used directly.
    # From your script, you have 'A/G Ratio Albumin and Globulin Ratio' as an input.
    # If you created a new 'Albumin_Globulin_Ratio' in your feature engineering, make sure to add it here.
    # Based on your script, 'A/G Ratio Albumin and Globulin Ratio' is already a feature.
    # So, if you explicitly created `Albumin_Globulin_Ratio` in FE, add this:
    df['Albumin_Globulin_Ratio'] = df['Â\xa0ALB Albumin'] / (df['Total Protiens'] - df['Â\xa0ALB Albumin'])

    # Ensure all required columns (original + engineered) are present and in the correct order
    # Drop the original A/G Ratio if you used the engineered one, or vice-versa.
    # It's best to align with `model_features` saved during training.

    # Ensure the order of columns matches `model_features`
    # This is critical! If `model_features` contains original columns first, then engineered,
    # create the DataFrame in that order.
    # The `model_features` list saved will contain columns after all preprocessing.
    # It's safest to construct the dataframe with all the original inputs, then add engineered features,
    # then select and reorder columns according to `model_features`.
    processed_df = df[model_features] # Select and reorder columns
    return processed_df

# --- Prediction Logic ---
if st.button("Predict Liver Disease"):
    # Convert input_data dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])

    # Apply preprocessing and feature engineering
    try:
        processed_input = preprocess_and_engineer(input_df.copy()) # Use .copy() to avoid SettingWithCopyWarning
        # Apply the same scaling as used during training
        scaled_input = scaler.transform(processed_input)

        # Make prediction
        prediction = best_model.predict(scaled_input)
        prediction_proba = best_model.predict_proba(scaled_input)[:, 1] # Probability of being positive (1)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error(f"The model predicts **Liver Disease is Likely** (Probability: {prediction_proba[0]:.2f}) ⚠️")
            st.write("Please consult a medical professional for diagnosis and treatment.")
        else:
            st.success(f"The model predicts **Liver Disease is Unlikely** (Probability: {prediction_proba[0]:.2f}) ✅")
            st.write("Maintain a healthy lifestyle and regular check-ups.")

        st.markdown("---")
        st.subheader("Input Data Summary:")
        st.dataframe(input_df)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please check your input values and try again.")
        st.write("Debug info: The column names in your input might not match the expected format.")
        st.write(f"Expected model features: {model_features}")
        st.write(f"Input DataFrame columns before processing: {input_df.columns.tolist()}")
        # Display the input df to see what's going into the processing step
        st.dataframe(input_df)
        st.write(f"Processed input columns: {processed_input.columns.tolist()}")