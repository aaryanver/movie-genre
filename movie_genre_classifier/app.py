import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or Scaler not found. Please run train.py first.")
        return None, None

model, scaler = load_model_and_scaler()

# Configure the Streamlit page
st.set_page_config(page_title="Movie Genre Classifier", page_icon="🎬", layout="centered")

# Custom CSS for dark theme and UI aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #121212;
        color: #ffffff;
    }
    h1 {
        text-align: center;
        color: #f39c12;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        width: 100%;
        background-color: #e74c3c;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s;
        border: none;
    }
    .stButton>button:hover {
        background-color: #c0392b;
        color: #f1c40f;
        box-shadow: 0px 4px 15px rgba(231, 76, 60, 0.4);
    }
    .result-text {
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        color: #2ecc71;
        margin-top: 20px;
        padding: 20px;
        border-radius: 10px;
        background-color: #1e1e1e;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Genre Classifier")

st.markdown("<p style='text-align: center; color: #aaaaaa;'>Predict the genre of a movie based on its duration, rating, and votes!</p>", unsafe_allow_html=True)

st.divider()

# Input Sliders
duration = st.slider("Duration (minutes)", min_value=60, max_value=200, value=120, step=1)
rating = st.slider("Rating (IMDB rating)", min_value=5.0, max_value=10.0, value=7.5, step=0.1)
votes = st.slider("Votes (number of votes)", min_value=10000, max_value=1000000, value=500000, step=10000)

# Prediction button
if st.button("Predict Genre"):
    if model is not None and scaler is not None:
        # Prepare input data
        input_data = np.array([[duration, rating, votes]])
        
        # Scale input
        scaled_input = scaler.transform(input_data)
        
        # Predict genre
        prediction = model.predict(scaled_input)[0]
        
        # Display result
        st.markdown(f"<div class='result-text'>Predicted Genre: {prediction} 🎬</div>", unsafe_allow_html=True)
    else:
        st.error("Cannot predict without a trained model.")
