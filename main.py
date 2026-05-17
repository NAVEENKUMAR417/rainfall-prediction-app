import pickle
import requests
import pandas as pd

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput


# -------- Load Trained Model --------
model = pickle.load(open("rainfall_model.pkl", "rb"))

# -------- OpenWeather API Key --------
API_KEY = "7855e5a98109d729a418083474e5b9b5"


class RainfallApp(App):

    def build(self):

        self.layout = BoxLayout(
            orientation='vertical',
            padding=20,
            spacing=15
        )

        self.title_label = Label(
            text="Rainfall Prediction App",
            font_size=28,
            size_hint=(1, 0.3)
        )

        self.city_input = TextInput(
            hint_text="Enter City Name",
            multiline=False
        )

        self.predict_button = Button(
            text="Predict Rainfall",
            size_hint=(1, 0.4)
        )

        self.predict_button.bind(
            on_press=self.predict_rainfall
        )

        self.result_label = Label(
            text="Prediction Result",
            font_size=24
        )

        self.layout.add_widget(self.title_label)
        self.layout.add_widget(self.city_input)
        self.layout.add_widget(self.predict_button)
        self.layout.add_widget(self.result_label)

        return self.layout


    def predict_rainfall(self, instance):

        city = self.city_input.text.strip()

        if city == "":
            self.result_label.text = "Please enter city name"
            return

        try:

            # -------- Fetch Weather Data --------
            api_url = (
                f"https://api.openweathermap.org/data/2.5/weather?"
                f"q={city}&appid={API_KEY}&units=metric"
            )

            response = requests.get(api_url)
            weather = response.json()

            if response.status_code != 200:
                self.result_label.text = "City not found"
                return

            # -------- Extract Weather Features --------
            temperature = weather["main"]["temp"]
            humidity = weather["main"]["humidity"]
            pressure = weather["main"]["pressure"]

            wind_speed = weather["wind"]["speed"]

            cloud = weather["clouds"]["all"]

            max_temp = temperature + 2
            min_temp = temperature - 2

            dew_point = temperature - 3

            sunshine = 5

            wind_direction = weather["wind"].get("deg", 0)

            # -------- Prepare Input Data --------
            input_data = pd.DataFrame([[

                pressure,
                max_temp,
                temperature,
                min_temp,
                dew_point,
                humidity,
                cloud,
                sunshine,
                wind_direction,
                wind_speed

            ]], columns=[

                'pressure',
                'maxtemp',
                'temperature',
                'mintemp',
                'dewpoint',
                'humidity',
                'cloud',
                'sunshine',
                'winddirection',
                'windspeed'

            ])

            # -------- Prediction --------
            prediction = model.predict(input_data)

            # -------- Show Result --------
            if prediction[0] == 1:
                self.result_label.text = f"Rain Expected in {city}"
            else:
                self.result_label.text = f"No Rain Expected in {city}"

        except:
            self.result_label.text = "Error occurred"


RainfallApp().run()