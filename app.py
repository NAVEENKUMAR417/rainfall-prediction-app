# =========================================
# Rainfall Prediction Web Application
# =========================================

# -------- Import Required Libraries --------
import streamlit as st
import pandas as pd
import requests
import pickle
import os

from streamlit_js_eval import get_geolocation


# -------- Streamlit Page Configuration --------
st.set_page_config(
    page_title="Rainfall Prediction App",
    page_icon="🌧",
    layout="centered"
)

st.title("🌧 Rainfall Prediction System")
st.write("Predict rainfall using live weather data and machine learning")


# -------- Load Trained Machine Learning Model --------
try:

    model = pickle.load(open("rainfall_model.pkl", "rb"))

except:
    st.error("Trained model file not found")
    st.stop()


# -------- Load Dataset --------
try:

    base_dir = os.path.dirname(__file__)

    dataset_path = os.path.join(
        base_dir,
        "data",
        "rainfall.csv"
    )

    dataset = pd.read_csv(dataset_path)

except:
    st.error("Dataset file not found")
    st.stop()


# -------- Data Cleaning --------
dataset.columns = dataset.columns.str.strip().str.lower()

if "day" in dataset.columns:
    dataset.drop("day", axis=1, inplace=True)

dataset = dataset.fillna(
    dataset.mean(numeric_only=True)
)

# -------- Feature Columns --------
X = dataset.drop("rainfall", axis=1)


# -------- OpenWeather API Key --------
API_KEY = "7855e5a98109d729a418083474e5b9b5"


# =========================================
# User Interface
# =========================================

st.subheader("Choose Location Method")

gps_button = st.button(
    "📍 Use Current Location"
)

st.markdown("---")

st.subheader("Or Enter Location Manually")

col1, col2, col3 = st.columns(3)

with col1:
    city = st.text_input("City")

with col2:
    state = st.text_input("State")

with col3:
    country = st.text_input("Country Code")

manual_predict = st.button(
    "Predict Rainfall"
)


# =========================================
# Weather API URL Generation
# =========================================

weather_url = None
location_name = ""


# -------- GPS Based Location --------
if gps_button:

    location = get_geolocation()

    if location is None:

        st.warning(
            "Please allow browser location access"
        )

        st.stop()

    latitude = location["coords"]["latitude"]
    longitude = location["coords"]["longitude"]

    location_name = (
        f"Latitude: {latitude}, "
        f"Longitude: {longitude}"
    )

    st.info(f"Detected Location: {location_name}")

    weather_url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"lat={latitude}&lon={longitude}"
        f"&appid={API_KEY}&units=metric"
    )


# -------- Manual Location --------
elif manual_predict:

    if city.strip() == "" or country.strip() == "":

        st.warning(
            "City and Country Code are required"
        )

        st.stop()

    if state.strip() != "":

        location_name = (
            f"{city},{state},{country}"
        )

    else:

        location_name = (
            f"{city},{country}"
        )

    st.info(f"Selected Location: {location_name}")

    weather_url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"q={location_name}"
        f"&appid={API_KEY}&units=metric"
    )


# =========================================
# Fetch Weather Data
# =========================================

if weather_url:

    try:

        response = requests.get(weather_url)

        weather_data = response.json()

        if response.status_code != 200:

            st.error(
                weather_data.get(
                    "message",
                    "Weather API Error"
                )
            )

            st.stop()

        # -------- Extract Weather Information --------
        temperature = weather_data["main"]["temp"]

        humidity = weather_data["main"]["humidity"]

        pressure = weather_data["main"]["pressure"]

        wind_speed = weather_data["wind"]["speed"]

        cloud_coverage = weather_data["clouds"]["all"]

        wind_direction = (
            weather_data["wind"].get("deg", 0)
        )

        # -------- Generate Additional Features --------
        max_temp = temperature + 2

        min_temp = temperature - 2

        dew_point = temperature - 3

        sunshine = 5

        # -------- Prepare Input Data --------
        input_data = pd.DataFrame([[

            pressure,
            max_temp,
            temperature,
            min_temp,
            dew_point,
            humidity,
            cloud_coverage,
            sunshine,
            wind_direction,
            wind_speed

        ]], columns=X.columns)

        # -------- Make Prediction --------
        prediction = model.predict(input_data)

        # -------- Display Weather Details --------
        st.markdown("---")

        st.subheader("Current Weather Details")

        st.write(f"🌡 Temperature: {temperature} °C")

        st.write(f"💧 Humidity: {humidity}%")

        st.write(f"🌬 Wind Speed: {wind_speed} m/s")

        st.write(f"☁ Cloud Coverage: {cloud_coverage}%")

        # -------- Prediction Result --------
        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction[0] == 1:

            st.success(
                f"🌧 Rainfall is likely expected in {location_name}"
            )

        else:

            st.success(
                f"☀ No rainfall expected in {location_name}"
            )

    except:

        st.error(
            "Error occurred while processing weather data"
        )