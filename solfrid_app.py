import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# 1. LOAD ASSETS
@st.cache_resource
def load_assets():
    # Load LSTM and XGBoost models
    model_lstm = tf.keras.models.load_model('model_lstm.h5', compile=False) 
    suit_model = joblib.load('suitability_model.pkl') # Make sure you have this file!

    # Load Encoders and Scaler
    crop_encoder = joblib.load('crop_encoder.pkl')
    county_encoder = joblib.load('county_encoder.pkl')
    scaler = joblib.load('scaler.pkl')

    # Load Price Data
    price_df = pd.read_csv('cleaned_price_data.csv')
    price_df['County'] = price_df['County'].str.lower().str.strip()

    return model_lstm, suit_model, crop_encoder, county_encoder, scaler, price_df

# Unpack all assets
model_lstm, suit_model, crop_encoder, county_encoder, scaler, price_df = load_assets()

# 2. XGBOOST SUITABILITY LOGIC
def get_suitability_recommendations(county, temp_max, temp_min, rainfall, humidity, soil_ph):
    # Use the loaded crop_encoder to get the list of crops
    all_crops = crop_encoder.classes_

    try:
        county_enc = county_encoder.transform([county])[0]
    except:
        return []

    recommendations = []

    # Define columns exactly as they were during XGBoost training
    cols = ['County_Enc', 'Crop_Enc', 'Temp_max', 'Temp_min', 'Rainfall', 'Humidity', 'Soil_pH']

    for crop_name in all_crops:
        try:
            crop_enc = crop_encoder.transform([crop_name])[0]

            # Create feature row
            features = pd.DataFrame([[
                county_enc, crop_enc, temp_max, temp_min, rainfall, humidity, soil_ph
            ]], columns=cols)

            # Get prediction
            is_suitable = suit_model.predict(features)[0]

            if is_suitable == 1:
                # Use predict_proba to get the Match Score
                probability = suit_model.predict_proba(features)[0][1]
                recommendations.append({
                    "Crop": crop_name,
                    "Match_Score": round(probability * 100, 2)
                })
        except:
            continue 

    return sorted(recommendations, key=lambda x: x['Match_Score'], reverse=True)

# 3. HYBRID LOGIC
def test_hybrid_system(county_name, temp, rain, ph, humidity):
    search_county = county_name.lower().strip()

    # 1. Environment Suitability (XGBoost)
    suitable_crops = get_suitability_recommendations(
        county=search_county, 
        temp_max=temp, 
        temp_min=temp-5, 
        rainfall=rain, 
        humidity=humidity, 
        soil_ph=ph
    )

    if not suitable_crops:
        return []

    final_recommendations = []

    # 2. Predict Prices for survivors (LSTM)
    # We already have county_id from the encoder
    county_id = county_encoder.transform([search_county])[0]

    for item in suitable_crops:
        crop_name = item['Crop']
        suit_score = item['Match_Score'] / 100

        history_df = price_df[(price_df['Commodity'] == crop_name) & 
                              (price_df['County'] == search_county)].iloc[-30:]

        if len(history_df) == 30:
            history_vals = history_df['Wholesale'].values.reshape(1, 30, 1)
            try:
                crop_id = crop_encoder.transform([crop_name])[0]
                meta = np.array([[crop_id, county_id]])

                pred_scaled = model_lstm.predict([history_vals, meta], verbose=0)

                # Inverse Scale (Dummy Array Fix)
                dummy = np.zeros((1, 2))
                dummy[0, 0] = pred_scaled[0, 0]
                pred_price = scaler.inverse_transform(dummy)[0, 0]

                final_recommendations.append({
                    "Crop": crop_name,
                    "Suitability": f"{item['Match_Score']}%",
                    "Forecasted_Price": round(pred_price, 2),
                    "Index": round(suit_score * pred_price, 2)
                })
            except:
                continue

    return sorted(final_recommendations, key=lambda x: x['Index'], reverse=True)

# 4. INTERFACE
st.set_page_config(page_title="SolFrid Advisor", page_icon="🌱", layout="wide")
st.title("🌱 SolFrid: Hybrid Crop Recommendation System")
st.markdown("---")

with st.sidebar:
    st.header("Environmental Conditions")
    c = st.selectbox("Select County", options=list(county_encoder.classes_))
    t = st.slider("Temperature (°C)", 10, 45, 25)
    r = st.number_input("Annualized Rainfall (mm)", value=1000)
    p = st.slider("Soil pH", 3.0, 9.0, 6.5)
    h = st.slider("Humidity (%)", 10, 100, 60)

if st.button("Generate Hybrid Recommendations"):
    with st.spinner("Analyzing biological and economic data..."):
        results = test_hybrid_system(c, t, r, p, h)

    if results:
        df = pd.DataFrame(results)
        st.success(f"Top Recommendation for {c}: **{results[0]['Crop']}**")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(df, use_container_width=True)
        with col2:
            st.bar_chart(data=df, x="Crop", y="Index")
    else:
        st.error("No suitable crops found or missing price data for this county.")
