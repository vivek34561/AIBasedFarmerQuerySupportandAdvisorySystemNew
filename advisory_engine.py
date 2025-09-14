# advisory_engine.py

import os
import requests
import sqlite3
from twilio.rest import Client
from dotenv import load_dotenv
load_dotenv()
from config import KERALA_DISTRICTS
# --- Configuration ---
# Store these in your .env file and load them using load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
DB_PATH = "chatbot.db" # Path to your SQLite database

# A map of districts to their rough coordinates for the weather API

def get_weather_data(lat, lon):
    """Fetches 5-day weather forecast from OpenWeatherMap."""
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        # For simplicity, we'll just average the next 24 hours of data
        forecasts = response.json()['list'][:8] # Next 24 hours (8 * 3-hour forecasts)
        avg_temp = sum(f['main']['temp'] for f in forecasts) / len(forecasts)
        avg_humidity = sum(f['main']['humidity'] for f in forecasts) / len(forecasts)
        # Check for rain in any of the next 8 periods
        will_rain = any(f['weather'][0]['main'] == 'Rain' for f in forecasts)
        return {"temp": avg_temp, "humidity": avg_humidity, "rain": will_rain}
    except requests.RequestException as e:
        print(f"Error fetching weather: {e}")
        return None

def apply_rules(weather_data, crop_data):
    """Applies rules to generate alerts. crop_data is your agricultural data."""
    alerts = []
    if not weather_data:
        return alerts

    # Rule 1: Fungal disease risk for Banana crops
    if (weather_data['humidity'] > 60 and 2 <= weather_data['temp'] <= 40):
        alerts.append({
            "crop": "Banana",
            "message": "കാലാവസ്ഥ: വാഴയ്ക്ക് കുമിൾ രോഗത്തിന് സാധ്യത. ഉയർന്ന ഈർപ്പം അപകടസാധ്യത വർദ്ധിപ്പിക്കുന്നു. പ്രതിരോധ നടപടികൾ സ്വീകരിക്കുക."
        })
        
    # Rule 2: Heavy rain warning
    if weather_data['rain']:
        alerts.append({
            "crop": "All", # This alert applies to all crops
            "message": "കാലാവസ്ഥ മുന്നറിയിപ്പ്: അടുത്ത 24 മണിക്കൂറിനുള്ളിൽ മഴയ്ക്ക് സാധ്യത. കീടനാശിനി തളിക്കുന്നത് ഒഴിവാക്കുക."
        })
        
    # Add more rules based on your agricultural data...
    return alerts

def send_sms_alert(phone_number, message):
    """Sends an SMS using Twilio."""
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER]):
        print("Twilio credentials not set. Skipping SMS.")
        return
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        client.messages.create(
            to=phone_number,
            from_=TWILIO_PHONE_NUMBER,
            body=message
        )
        print(f"SMS sent to {phone_number}")
    except Exception as e:
        print(f"Failed to send SMS to {phone_number}: {e}")

def run_advisory_check():
    """The main function to be scheduled."""
    print("--- Running Daily Advisory Check ---")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for district, coords in KERALA_DISTRICTS.items():
        print(f"Checking for {district}...")
        weather_data = get_weather_data(coords['lat'], coords['lon'])
        
        # Here you would load your specific crop data for the district
        crop_data = {} # Placeholder for your data
        
        active_alerts = apply_rules(weather_data, crop_data)

        if not active_alerts:
            continue

        for alert in active_alerts:
            # Find users who should receive this alert
            if alert['crop'] == "All":
                cursor.execute("SELECT phone_number FROM users WHERE district=?", (district,))
            else:
                cursor.execute("SELECT phone_number FROM users WHERE district=? AND primary_crop=?", (district, alert['crop']))
            
            target_users = cursor.fetchall()
            for user_phone in target_users:
                send_sms_alert(user_phone[0], alert['message'])
    
    conn.close()
    print("--- Advisory Check Complete ---")