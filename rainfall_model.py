# ==============================
# Rainfall Prediction System
# ==============================

# -------- Import Required Libraries --------
import pandas as pd
import requests
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# -------- Load Dataset --------
print("Loading dataset...")

try:
    current_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(current_dir, "data", "rainfall.csv")

    dataset = pd.read_csv(dataset_path)

    print("Dataset loaded successfully")

except FileNotFoundError:
    print("Dataset file not found")
    exit()


# -------- Data Preprocessing --------
print("Processing dataset...")

# Clean column names
dataset.columns = dataset.columns.str.strip().str.lower()

# Convert rainfall values into binary values
if "rainfall" not in dataset.columns:
    print("Rainfall column missing in dataset")
    exit()

dataset["rainfall"] = (
    dataset["rainfall"]
    .astype(str)
    .str.lower()
    .map({
        "yes": 1,
        "no": 0
    })
)

# Remove unnecessary column
if "day" in dataset.columns:
    dataset.drop("day", axis=1, inplace=True)

# Fill missing values
dataset.fillna(dataset.mean(numeric_only=True), inplace=True)

print("Dataset cleaned successfully")


# -------- Prepare Features and Labels --------
X = dataset.drop("rainfall", axis=1)
y = dataset["rainfall"]


# -------- Split Dataset --------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -------- Train Machine Learning Model --------
print("Training model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully")


# -------- Save Model --------
model_file = "rainfall_model.pkl"

with open(model_file, "wb") as file:
    pickle.dump(model, file)

print(f"Model saved successfully as '{model_file}'")


# -------- Function to Detect User Location --------
def detect_location():

    try:
        response = requests.get("http://ip-api.com/json/")
        location_data = response.json()

        latitude = location_data.get("lat")
        longitude = location_data.get("lon")

        city = location_data.get("city")
        country = location_data.get("countryCode")

        location_name = f"{city}, {country}"

        return latitude, longitude, location_name

    except:
        return None, None, None


# -------- User Choice --------
print("\nChoose Weather Input Method")
print("1. Detect Current Location Automatically")
print("2. Enter City Manually")

choice = input("Enter your choice (1 or 2): ")


# -------- Weather API Key --------
API_KEY = "7855e5a98109d729a418083474e5b9b5"


# -------- Generate API URL --------
if choice == "1":

    lat, lon, city_name = detect_location()

    if lat is None:
        print("Could not detect current location")

        city_name = input("Enter city name: ")

        api_url = (
            f"https://api.openweathermap.org/data/2.5/weather?"
            f"q={city_name}&appid={API_KEY}&units=metric"
        )

    else:
        print(f"Detected Location: {city_name}")

        api_url = (
            f"https://api.openweathermap.org/data/2.5/weather?"
            f"lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        )

else:

    city_name = input("Enter city name: ")

    api_url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"q={city_name}&appid={API_KEY}&units=metric"
    )


# -------- Fetch Weather Data --------
print("\nFetching weather data...")

response = requests.get(api_url)
weather_data = response.json()

if response.status_code != 200:
    print("Error fetching weather data")
    print(weather_data.get("message", "Unknown error"))
    exit()

print("Weather data fetched successfully")


# -------- Extract Required Weather Features --------
try:

    temperature = weather_data["main"]["temp"]
    humidity = weather_data["main"]["humidity"]
    pressure = weather_data["main"]["pressure"]

    wind_speed = weather_data["wind"]["speed"]

    cloud_coverage = weather_data["clouds"]["all"]

    max_temp = temperature + 2
    min_temp = temperature - 2

    dew_point = temperature - 3

    sunshine = 5

    wind_direction = weather_data["wind"].get("deg", 0)

except KeyError:
    print("Required weather data missing")
    exit()


# -------- Prepare Input For Prediction --------
input_values = pd.DataFrame([[
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
prediction = model.predict(input_values)


# -------- Display Result --------
print("\n========== Prediction Result ==========")

if prediction[0] == 1:
    print(f"Rain is likely expected in {city_name}")
else:
    print(f"No rainfall expected in {city_name}")

print("=======================================")