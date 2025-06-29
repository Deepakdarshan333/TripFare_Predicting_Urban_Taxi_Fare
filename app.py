

import streamlit as st
import pickle
import numpy as np
from datetime import datetime
import pandas as pd

# Load the trained model
try:
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    model_loaded = True
except Exception as e:
    st.error("❌ Error loading model or matching features:")
    st.error(str(e))
    model_loaded = False
st.title("🚖 TripFare Prediction App")
st.write("✅ Streamlit UI loaded without model")


# ------------------------ Haversine Function ------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# ------------------------ Streamlit UI ------------------------

st.title("🚖 TripFare Prediction App")
st.markdown("Predict NYC taxi fare based on trip details")

# Input fields
pickup_date = st.date_input("Pickup Date", datetime.now().date())
pickup_time = st.time_input("Pickup Time", datetime.now().time())
pickup_datetime = datetime.combine(pickup_date, pickup_time)

passenger_count = st.slider("Passenger Count", 1, 6, 1)
pickup_lat = st.number_input("Pickup Latitude", value=40.7614327)
pickup_lon = st.number_input("Pickup Longitude", value=-73.9798156)
dropoff_lat = st.number_input("Dropoff Latitude", value=40.6413111)
dropoff_lon = st.number_input("Dropoff Longitude", value=-73.7803331)
rate_code = st.selectbox("RatecodeID", [1, 2, 3, 4, 5, 6])
payment_type = st.selectbox("Payment Type", ['Credit card', 'Cash', 'No charge', 'Dispute', 'Unknown', 'Voided trip'])
store_and_fwd = st.radio("Store and Forward Flag", ['N', 'Y'])

# Feature engineering
pickup_day = pickup_datetime.strftime("%A")
hour_of_day = pickup_datetime.hour
am_pm = "AM" if hour_of_day < 12 else "PM"
is_weekend = 1 if pickup_day in ['Saturday', 'Sunday'] else 0
is_night = 1 if hour_of_day <= 5 or hour_of_day >= 22 else 0
distance = haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
log_distance = np.log1p(distance)

# Encode categorical
payment_map = {'Credit card': 1, 'Cash': 2, 'No charge': 3, 'Dispute': 4, 'Unknown': 5, 'Voided trip': 6}
store_map = {'N': 0, 'Y': 1}

# One-hot encodings
day_dummies = pd.get_dummies(pd.Series(pickup_day), prefix='pickup_day')
ampm_dummies = pd.get_dummies(pd.Series(am_pm), prefix='am_pm')

# Final input
input_data = {
    'passenger_count': passenger_count,
    'RatecodeID': rate_code,
    'payment_type': payment_map[payment_type],
    'store_and_fwd_flag': store_map[store_and_fwd],
    'extra': 0.5,
    'mta_tax': 0.5,
    'tip_amount': 1.0,
    'tolls_amount': 0.0,
    'improvement_surcharge': 0.3,
    'fare_amount': 7.0,
    'log_trip_distance': log_distance,
    'hour_of_day': hour_of_day,
    'is_weekend': is_weekend,
    'is_night': is_night
}

# Merge encoded dummies
input_data.update(day_dummies.to_dict(orient='records')[0])
input_data.update(ampm_dummies.to_dict(orient='records')[0])

# Create DataFrame and fill missing one-hot columns
input_df = pd.DataFrame([input_data])
try:
    expected_cols = model.feature_names_in_
    st.write("✅ Model loaded successfully. Features expected:")
    st.write(expected_cols)

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_cols]
except Exception as e:
    st.error("❌ Error loading model or matching features:")
    st.error(e)

# Predict
if st.button("Predict Fare"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Fare: ${prediction:.2f}")
