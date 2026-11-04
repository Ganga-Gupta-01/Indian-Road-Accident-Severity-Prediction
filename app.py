import streamlit as st
import pandas as pd
import joblib

# Load dataset and model
df = pd.read_csv("data/indian_roads_accidents_dataset.csv")
model = joblib.load("model/road_accident_model.pkl")

st.set_page_config(
    page_title="Indian Road Accident Severity Predictor",
    page_icon="🚦"
)

st.title("🚦 Indian Road Accident Severity Predictor")

st.write(
    "Predict whether an accident is Minor, Major, or Fatal based on accident details."
)

# Numerical Inputs
hour = st.slider("Hour", 0, 23, 12)

lanes = st.number_input(
    "Lanes",
    min_value=1,
    max_value=10,
    value=3
)

temperature = st.number_input(
    "Temperature",
    value=25.0
)

vehicles_involved = st.number_input(
    "Vehicles Involved",
    min_value=1,
    value=1
)

casualties = st.number_input(
    "Casualties",
    min_value=0,
    value=1
)

risk_score = st.number_input(
    "Risk Score",
    min_value=0.0,
    value=1.0
)

# Dataset-Based Dropdowns
day_of_week = st.selectbox(
    "Day Of Week",
    sorted(df["day_of_week"].dropna().unique())
)

road_type = st.selectbox(
    "Road Type",
    sorted(df["road_type"].dropna().unique())
)

weather = st.selectbox(
    "Weather",
    sorted(df["weather"].dropna().unique())
)

visibility = st.selectbox(
    "Visibility",
    sorted(df["visibility"].dropna().unique())
)

traffic_density = st.selectbox(
    "Traffic Density",
    sorted(df["traffic_density"].dropna().unique())
)

festival = st.selectbox(
    "Festival",
    sorted(df["festival"].fillna("No Festival").unique())
)

# Binary Fields
is_weekend = st.selectbox(
    "Weekend",
    [0, 1]
)

traffic_signal = st.selectbox(
    "Traffic Signal",
    [0, 1]
)

is_peak_hour = st.selectbox(
    "Peak Hour",
    [0, 1]
)

if st.button("Predict Severity"):

    data = pd.DataFrame({
        "hour": [hour],
        "lanes": [lanes],
        "temperature": [temperature],
        "vehicles_involved": [vehicles_involved],
        "casualties": [casualties],
        "risk_score": [risk_score],
        "day_of_week": [day_of_week],
        "road_type": [road_type],
        "weather": [weather],
        "visibility": [visibility],
        "traffic_density": [traffic_density],
        "festival": [festival],
        "is_weekend": [is_weekend],
        "traffic_signal": [traffic_signal],
        "is_peak_hour": [is_peak_hour]
    })

    prediction = model.predict(data)

    st.subheader("Prediction")
    st.success(prediction[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(data)

        st.subheader("Prediction Probabilities")

        prob_df = pd.DataFrame(
            probabilities,
            columns=model.classes_
        )

        st.dataframe(prob_df)