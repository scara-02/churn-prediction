import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# --- Streamlit App ---
st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title('🧠 Customer Churn Prediction App')

st.markdown("""
Use this interactive tool to estimate a customer's churn probability based on their demographic and account data.
""")

# --- User Input Section ---
with st.form("user_form"):
    st.subheader("📋 Input Customer Details")
    
    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 92, step=1)
        credit_score = st.number_input('💳 Credit Score', min_value=0, max_value=1000, value=600)

    with col2:
        balance = st.number_input('🏦 Balance', min_value=0.0, format="%.2f")
        estimated_salary = st.number_input('💼 Estimated Salary', min_value=0.0, format="%.2f")
        tenure = st.slider('📅 Tenure (years)', 0, 10)
        num_of_products = st.slider('📦 Number of Products', 1, 4)
        has_cr_card = st.radio('💳 Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        is_active_member = st.radio('🟢 Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

    submitted = st.form_submit_button("🔍 Predict")

# --- Prediction Logic ---
if submitted:
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale input
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.markdown(f"### 🔎 Churn Probability: `{prediction_proba:.2%}`")

    if prediction_proba > 0.5:
        st.error("⚠️ The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **not likely to churn**.")
