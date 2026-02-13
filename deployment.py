import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load assets
@st.cache_resource
def load_assets():
    model = joblib.load('./models/rf_model.joblib')
    scaler = joblib.load('scaler.joblib')
    encoders = joblib.load('count_encoders.joblib')
    return model, scaler, encoders

model, scaler, encoders = load_assets()

st.title("🎬 Movie Success Predictor")
st.markdown("Enter movie details to predict success based on training data patterns.")

# Layout with two columns
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Year", 1900, 2030, 2024)
    duration = st.number_input("Duration (minutes)", 1, 500, 120)
    mpa = st.text_input("MPA Rating", "PG-13")
    writers = st.text_input("Writers", "Christopher Nolan")
    directors = st.text_input("Directors", "Christopher Nolan")
    stars = st.text_input("Stars", "Leonardo DiCaprio")

with col2:
    countries = st.text_input("Country of Origin", "United States")
    locations = st.text_input("Filming Locations", "Los Angeles")
    prod_company = st.text_input("Production Company", "Warner Bros.")
    genres = st.text_input("Genres", "Action")
    languages = st.text_input("Languages", "English")
    release_decade = (year // 10) * 10

# 1. Define the exact feature names in the exact order used in your notebook
feature_names = [
    'Year', 'Duration', 'MPA', 'writers', 'directors', 'stars', 
    'countries_origin', 'filming_locations', 'production_company', 
    'genres', 'Languages', 'release_decade'
]

# 2. Inside your "Predict" button logic:
if st.button("Predict Success"):
    # Create the dictionary as we did before
    raw_data = {
        'Year': year,
        'Duration': duration,
        'MPA': mpa,
        'writers': writers,
        'directors': directors,
        'stars': stars,
        'countries_origin': countries,
        'filming_locations': locations,
        'production_company': prod_company,
        'genres': genres,
        'Languages': languages,
        'release_decade': release_decade
    } 
    
    # Create the DataFrame with the feature names
    input_df = pd.DataFrame([raw_data], columns=feature_names)

    # 3. Apply the Count Encoding mapping
    for col, count_map in encoders.items():
        val = str(input_df[col].iloc[0])
        input_df[col] = count_map.get(val, 1)

    # 4. Scale and Predict
    # By keeping it as a DataFrame, the warning disappears
    scaled_input = scaler.transform(input_df)
    
    # Wrap scaled_input back into a DataFrame to keep names for the model
    scaled_input_df = pd.DataFrame(scaled_input, columns=feature_names)
    
    prediction = model.predict(scaled_input_df)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.success(f"🚀 High Success Probability! ({probability:.2%})")
    else:
        st.error(f"📉 Low Success Probability. ({probability:.2%})")
