# churn-prediction

📊 Customer Churn Prediction App
A web-based interactive tool powered by Streamlit and a Deep Learning (ANN) model that predicts the likelihood of a customer churning from a bank. With an accuracy of 93%, the model leverages user demographic and banking activity to make predictions in real-time.

🔍 Project Overview
Customer churn is a key indicator of business health in sectors like banking, telecom, and SaaS. Retaining existing customers is far more cost-effective than acquiring new ones.

This project aims to:

Build a deep learning model to predict churn.

Use structured customer data including demographic, credit, and account usage details.

Create an intuitive and responsive Streamlit web app to allow interactive predictions.

🚀 Key Features
✅ 93% Model Accuracy using Artificial Neural Networks (ANN)

🌐 Streamlit UI for easy access via browser

🔄 Real-time predictions with user-friendly inputs

📦 Trained model saved as .h5 (Keras)

📁 Supports loading saved encoders (LabelEncoder & OneHotEncoder) and scaler (StandardScaler)

📊 Probabilistic output for interpretability

🔧 Encapsulation of full ML pipeline: encoding → scaling → prediction

🧠 Model Architecture
The model is a fully connected feedforward neural network (ANN) built using TensorFlow/Keras.

Architecture Summary:
Input Layer: 11 features (after encoding)

Hidden Layers:

Dense (units=64), activation='relu'

Dense (units=32), activation='relu'

Output Layer:

Dense (units=1), activation='sigmoid' (for binary classification)

Loss Function: binary_crossentropy

Optimizer: adam

Metrics: accuracy

Trained on a cleaned and preprocessed version of the Bank Customer Churn dataset.




## 📁 File Structure
'''
📦 churn-prediction-app/
├── app.py # Streamlit App
├── model.h5 # Trained ANN model
├── scaler.pkl # StandardScaler object
├── label_encoder_gender.pkl # LabelEncoder for gender
├── onehot_encoder_geo.pkl # OneHotEncoder for geography
├── requirements.txt # Python dependencies
└── README.md # Project documentation
'''


⚙️ How to Run Locally
🔧 Prerequisites
Python ≥ 3.8

tensorflow, streamlit, pandas, scikit-learn, etc.

🧪 Step-by-step Setup
'''
# 1. Clone the repository
git clone https://github.com/yourusername/churn-prediction-app.git
cd churn-prediction-app

# 2. (Optional) Create a virtual environment
conda create -n churnenv python=3.8
conda activate churnenv

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run app.py
'''
