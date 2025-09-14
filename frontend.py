import streamlit as st
import requests
from typing import List, Dict
import os
import uuid
import json
import pandas as pd
import io  # <<< ADDED IMPORT
import pickle
# To be installed: pip install streamlit-webrtc st_audiorec pandas gtts
import st_audiorec
from datetime import datetime
from gtts import gTTS
import numpy as np  # <<< ADDED IMPORT
from config import KERALA_DISTRICTS , KERALA_DISTRICT_COORDS
# <<< ADDED IMPORT

# -------------------- Configuration --------------------
API_URL = "http://localhost:8000"
MODEL_PATH = "models/" 
# -------------------- Utility Functions --------------------
CROP_SCHEDULES = {
    "Paddy-I (Kharif) - North Kerala": [
        {'stage': 'Sowing', 'start_week': 17},
        {'stage': 'Transplanting', 'start_week': 21},
        {'stage': 'Vegetative Growth', 'start_week': 24},
        {'stage': 'Flowering', 'start_week': 30},
        {'stage': 'Grain Formation', 'start_week': 33},
        {'stage': 'Harvesting', 'start_week': 36}
    ],
    "Paddy-II (Rabi) - Central Kerala": [
        {'stage': 'Sowing', 'start_week': 31},
        {'stage': 'Transplanting', 'start_week': 35},
        {'stage': 'Vegetative Growth', 'start_week': 38},
        {'stage': 'Flowering', 'start_week': 42},
        {'stage': 'Grain Formation', 'start_week': 45},
        {'stage': 'Harvesting', 'start_week': 49}
    ]
    # You can add more crops here by extracting data from the PDF
}


def schedule_notification(activity: str, date_str: str):
    """Calls the backend to schedule an SMS notification."""
    try:
        payload = {"activity_name": activity, "notify_date": date_str}
        response = requests.post(f"{API_URL}/schedule-notification", json=payload, timeout=15)
        response.raise_for_status()
        st.success(f"✅ Reminder set! You'll be notified on {date_str}.")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to schedule notification: {e}")


def fetch_mandi_prices(selected_date, district, crop, market=None) -> List[Dict]:
    """Fetches mandi price data from our backend API."""
    params = {
        "arrival_date": selected_date.strftime("%Y-%m-%d"),
        "district": district,
        "commodity": crop
    }
    # <<< CHANGE START: Add market to params if it's provided >>>
    if market:
        params["market"] = market
    # <<< CHANGE END >>>

    try:
        response = requests.get(f"{API_URL}/mandi-prices", params=params, timeout=20)
        response.raise_for_status()
        result = response.json()
        st.info(result['message'])
        return result.get('data', [])
    except requests.exceptions.HTTPError as e:
        st.error(f"Error: The server returned status code {e.response.status_code}.")
        try:
            st.error(f"Details: {e.response.json().get('detail')}")
        except json.JSONDecodeError:
            st.error(f"Details: {e.response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the server: {e}")
    return []

def get_weather_data(lat: float, lon: float, api_key: str) -> Dict:
    """
    Fetches weather data for the next 24 hours, including temp, humidity,
    rainfall, max temp, and max wind speed.
    """
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        forecast_list = data.get('list', [])
        if not forecast_list:
            return None

        # Get current temp/humidity from the first forecast period
        current_forecast = forecast_list[0]
        temperature = current_forecast.get("main", {}).get("temp")
        humidity = current_forecast.get("main", {}).get("humidity")

        # Calculate totals and maximums over the next 24 hours (8 periods)
        total_rainfall = 0
        max_temp = -100 # Initialize with a very low number
        max_wind_speed = 0

        for period in forecast_list[:8]:
            # Sum rainfall
            total_rainfall += period.get('rain', {}).get('3h', 0)
            # Find max temperature
            if period.get("main", {}).get("temp_max", -100) > max_temp:
                max_temp = period["main"]["temp_max"]
            # Find max wind speed
            if period.get("wind", {}).get("speed", 0) > max_wind_speed:
                max_wind_speed = period["wind"]["speed"]

        if temperature is not None and humidity is not None:
            return {
                "temperature": temperature,
                "humidity": humidity,
                "total_rainfall": total_rainfall,
                "max_temp": max_temp,
                "wind_speed": max_wind_speed * 3.6  # Convert m/s to km/h
            }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            # Give a specific, helpful error for the most common problem
            st.error("Invalid OpenWeather API Key. Please check the key in the sidebar and make sure it is active.", icon="🔑")
        else:
            st.error(f"Weather API request failed with status code {e.response.status_code}.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the weather service. Check your internet connection.")
        return None
def check_for_alerts(weather_data: Dict) -> List:
    """Checks weather data against predefined thresholds to generate alerts."""
    alerts = []
    if not weather_data:
        return alerts

    # Rule 1: Heavy Rain Warning
    if weather_data.get("total_rainfall", 0) > 10:
        alerts.append((
            "warning",
            f"**Heavy Rain Warning:** {weather_data['total_rainfall']:.1f} mm of rain expected in the next 24 hours. Ensure proper drainage to avoid waterlogging."
        ))

    # Rule 2: Heat Stress Alert
    if weather_data.get("max_temp", 0) > 10:
        alerts.append((
            "error",
            f"**Heat Stress Alert:** Temperature may reach {weather_data['max_temp']:.1f}°C. Provide irrigation to crops to reduce heat stress."
        ))

    # Rule 3: High Wind Advisory
    if weather_data.get("wind_speed", 0) > 20:
        alerts.append((
            "warning",
            f"**High Wind Advisory:** Wind speeds may reach {weather_data['wind_speed']:.1f} km/h. Protect young or vulnerable plants."
        ))
    
    return alerts

@st.cache_resource(show_spinner="Loading crop prediction model...")
def load_crop_model():
    """
    Loads the preprocessor and model from disk.
    Uses caching to ensure this is only done once per session.
    """
    try:
        preprocessor_path = os.path.join(MODEL_PATH, "preprocessor.pkl")
        model_path = os.path.join(MODEL_PATH, "model.pkl")

        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        return preprocessor, model
    except FileNotFoundError:
        st.error(f"Model or preprocessor file not found in '{MODEL_PATH}' directory. Please make sure 'preprocessor.pkl' and 'model.pkl' are present.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the models: {e}")
        return None, None
    
    
# <<< NEW FUNCTION START: Text-to-Speech conversion
@st.cache_data(show_spinner="Generating audio...")
def text_to_speech(text: str, language: str) -> bytes:
    """Converts text to speech using gTTS and returns audio bytes."""
    lang_code_map = {
        "English": "en", "Malayalam": "ml", "Hindi": "hi",
        "Spanish": "es", "French": "fr", "German": "de",
        "Chinese": "zh-CN", "Arabic": "ar"
    }
    lang_code = lang_code_map.get(language, 'en')

    try:
        tts = gTTS(text=text, lang=lang_code, tld="co.in", slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.read()
    except Exception as e:
        st.error(f"Failed to generate audio: {e}")
        return None
# <<< NEW FUNCTION END

@st.cache_data(show_spinner=False)
def get_all_threads() -> List[str]:
    try:
        response = requests.get(f"{API_URL}/threads", timeout=5)
        response.raise_for_status()
        return response.json().get("threads", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch threads: {e}")
        return []

@st.cache_data(show_spinner=False)
def get_documents() -> List[str]:
    try:
        response = requests.get(f"{API_URL}/documents", timeout=5)
        response.raise_for_status()
        return response.json().get("documents", [])
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch document list: {e}")
        return []

@st.cache_data(show_spinner=False)
def get_processing_status() -> Dict:
    try:
        response = requests.get(f"{API_URL}/processing_status", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch processing status: {e}")
        return {"total_documents": 0, "processed_chunks": 0}

def create_new_thread() -> str:
    try:
        response = requests.post(f"{API_URL}/new_thread", timeout=5)
        response.raise_for_status()
        return response.json().get("thread_id", "")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to create new thread: {e}")
        return ""

def load_thread(thread_id: str) -> List[Dict]:
    try:
        response = requests.get(f"{API_URL}/load_thread/{thread_id}", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to load thread {thread_id}: {e}")
        return []

def send_message(thread_id: str, messages: List[Dict], language: str = "English") -> List[Dict]:
    try:
        payload = {"thread_id": thread_id, "messages": messages, "language": language}
        response = requests.post(f"{API_URL}/chat", json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to send message: {e}")
        return []


def log_feedback(thread_id: str, msg_index: int, rating: int):
    """Logs feedback for a specific message."""
    try:
        payload = {"thread_id": thread_id, "message_index": msg_index, "rating": rating}
        requests.post(f"{API_URL}/log-feedback", json=payload, timeout=10)
    except requests.exceptions.RequestException as e:
        # Fail silently on the frontend or show a minor error
        print(f"Could not log feedback: {e}")

def escalate_to_expert(thread_id: str):
    """Calls the backend to escalate a conversation."""
    try:
        payload = {"thread_id": thread_id}
        response = requests.post(f"{API_URL}/escalate-query", json=payload, timeout=30)
        response.raise_for_status()
        st.success("✅ Your query has been sent to an expert for review!")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to escalate query: {e}")


# -------------------- Session State Initialization --------------------
# (Combined the duplicated blocks for cleanliness)
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = create_new_thread() or str(uuid.uuid4())
if 'message_history' not in st.session_state:
    st.session_state.message_history = []
if 'chat_threads' not in st.session_state:
    all_threads = get_all_threads()
    updated_threads = [st.session_state.thread_id]
    for thread in all_threads:
        if thread not in updated_threads:
            updated_threads.append(thread)
    st.session_state.chat_threads = updated_threads
if 'voice_lang' not in st.session_state:
    st.session_state.voice_lang = "English"
if 'last_transcription' not in st.session_state:
    st.session_state.last_transcription = ""

if 'feedback' not in st.session_state:
    st.session_state.feedback = {}

if 'temp_val' not in st.session_state:
    st.session_state.temp_val = 25.5
    
if 'humidity_val' not in st.session_state:
    st.session_state.humidity_val = 75.0
    
if 'rainfall_val' not in st.session_state:
    st.session_state.rainfall_val = 200.0

if 'planner_result' not in st.session_state:
    st.session_state.planner_result = None

# -------------------- Sidebar Controls --------------------
with st.sidebar:
    st.title("⚙️ Controls")
    st.divider()

    if st.button("➕ New Chat", use_container_width=True):
        new_id = create_new_thread()
        if new_id:
            st.session_state.thread_id = new_id
        else:
            st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.message_history = []
        st.session_state.chat_threads = get_all_threads() or [st.session_state.thread_id]
        st.rerun()

    language = st.selectbox(
        "Response Language",
        ["English", "Malayalam", "Hindi", "Spanish", "French", "German", "Chinese", "Arabic"],
        index=0
    )
    
    st.divider()
    st.subheader("🌦️ Weather Data")
    api_key_input = st.text_input("Enter OpenWeather API Key", type="password", help="Get a free key from openweathermap.org")
    selected_district = st.selectbox(
        "Select Your District",
        options=list(KERALA_DISTRICT_COORDS.keys()),
        index=0 # Default to the first district
    )
    
    st.divider()
    st.subheader("🚨 Proactive Alerts")
    if not api_key_input:
        st.info("Enter your OpenWeather API key above to activate alerts.")
    else:
        # Cache the alert check to avoid calling the API on every single interaction
        @st.cache_data(ttl=600) # Cache results for 10 minutes
        def get_alerts_for_district(district, key):
            coords = KERALA_DISTRICT_COORDS.get(district)
            if coords:
                weather_data = get_weather_data(coords[0], coords[1], key)
                if weather_data:
                    return check_for_alerts(weather_data)
            return []

        # Run the alert check
        alerts = get_alerts_for_district(selected_district, api_key_input)

        if not alerts:
            st.success("✅ Conditions look good. No alerts for your district.")
        else:
            for alert_type, message in alerts:
                if alert_type == "error":
                    st.error(message, icon="🔥")
                elif alert_type == "warning":
                    st.warning(message, icon="⚠️")
                else:
                    st.info(message, icon="ℹ️")
    
    st.divider()
    st.subheader("🎤 Voice Input")
    st.info("Record your question here.")
    audio_bytes = st_audiorec.st_audiorec()

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        if st.button("Send Voice Query", use_container_width=True, type="primary"):
            files = {"file": ("voice_query.wav", audio_bytes, "audio/wav")}
            params = {"language": language}
            try:
                response = requests.post(f"{API_URL}/voice_query", files=files, data=params, timeout=60)
                if response.status_code == 200:
                    result = response.json()
                    transcription = result.get("transcription")
                    if transcription:
                        st.session_state.voice_input = transcription
                        st.success("Voice query transcribed!")
                        st.rerun()
                    else:
                        st.error("Failed to get transcription from response.")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to voice service: {e}")

    st.divider()
    st.subheader("📚 Knowledge Base")
    with st.expander("System Status", expanded=False):
        status = get_processing_status()
        total_docs = status.get("total_documents", 0)
        processed_chunks = status.get("processed_chunks", 0)
        if total_docs > 0 and processed_chunks > 0:
            st.success("🟢 Ready")
        elif total_docs > 0:
            st.warning("🟠 Processing...")
        else:
            st.info("⚪ No documents loaded")
        st.metric("Documents", total_docs)
        st.metric("Chunks", processed_chunks)
    st.divider()
    st.subheader("My Conversations")
    for idx, thread_id in enumerate(st.session_state.chat_threads[:10]):
        short_id = thread_id[:8]
        if st.button(f"💬 {short_id}...", key=f"chat_{idx}", use_container_width=True):
            st.session_state.thread_id = thread_id
            st.session_state.message_history = load_thread(thread_id)
            st.rerun()

# -------------------- Main UI --------------------
st.set_page_config(page_title="Digital Krishi Officer", page_icon="🌾", layout="wide")
st.title("🌾 Digital Krishi Officer - കൃഷി സഹായി")
st.caption("AI-powered farming assistant for Kerala farmers")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Ask Expert",
    "🌿 Crop Disease Detection",
    "📊 Crop Recommender",
    "🗓️ Crop Schedule Planner",
    "📈 Mandi Prices"
])

# -------------------- Chatbot Tab --------------------
with tab1:
    chat_container = st.container(height=400)

    
    with chat_container:
        # We use enumerate to get an index for unique keys
        for idx, msg in enumerate(st.session_state.message_history):
            role = msg.get('role', 'assistant')
            content = msg.get('content', '')
            with st.chat_message(role):
                st.markdown(content)
                # Add the "Read Aloud" button only for assistant messages
                if role == 'assistant':
                    feedback_key = f"feedback_{idx}"
                    feedback_given = st.session_state.feedback.get(feedback_key)
                    # We create 5 columns to neatly space out all the buttons
                    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 2, 5])
                    
                    # Column 1: Read Aloud Button
                    with col1:
                        if st.button("🔊 Read", key=f"read_aloud_{idx}"):
                            audio_bytes = text_to_speech(content, language)
                            if audio_bytes:
                                # Display the audio player right below the button
                                st.audio(audio_bytes, format="audio/mp3")

                    # Column 2: Thumbs Up Button
                    with col2:
                        if st.button("👍", key=f"thumbs_up_{idx}", disabled=bool(feedback_given)):
                            st.session_state.feedback[feedback_key] = "👍"
                            log_feedback(st.session_state.thread_id, idx, 1)
                            st.rerun()
                    
                    # Column 3: Thumbs Down Button
                    with col3:
                        if st.button("👎", key=f"thumbs_down_{idx}", disabled=bool(feedback_given)):
                            st.session_state.feedback[feedback_key] = "👎"
                            log_feedback(st.session_state.thread_id, idx, -1)
                            st.rerun()

                    # Column 4: Conditional Escalate Button
                    # This only appears if the user has clicked "👎"
                    if st.session_state.feedback.get(feedback_key) == "👎":
                        with col4:
                            if st.button("Escalate", key=f"escalate_{idx}"):
                                escalate_to_expert(st.session_state.thread_id)
                                st.session_state.feedback[feedback_key] = "escalated"
                    
                    # Display confirmation messages below the main content
                    if st.session_state.feedback.get(feedback_key) == "👍":
                        st.caption("Thanks for your feedback!")
                    if st.session_state.feedback.get(feedback_key) == "escalated":
                        st.caption("This query has been escalated to an expert.")
    # <<< MODIFICATION END

    user_input = None
    if 'voice_input' in st.session_state and st.session_state.voice_input:
        user_input = st.session_state.pop('voice_input')
    elif 'pending_input' in st.session_state and st.session_state.pending_input:
        user_input = st.session_state.pop('pending_input')
    else:
        user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state.message_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            ai_messages = send_message(
                st.session_state.thread_id,
                st.session_state.message_history,
                language
            )

        if ai_messages:
            for ai_msg in ai_messages:
                with st.chat_message("assistant"):
                    st.markdown(ai_msg.get('content', ''))
                st.session_state.message_history.append(ai_msg)
            
            get_all_threads.clear()
            all_threads = get_all_threads()
            updated_threads = [st.session_state.thread_id]
            for thread in all_threads:
                if thread not in updated_threads:
                    updated_threads.append(thread)
            st.session_state.chat_threads = updated_threads
            st.rerun()

# -------------------- Disease Prediction Tab --------------------
# (This section remains unchanged)
with tab2:
    st.header("🔬 Disease Prediction from Medical Images")
    st.info("Upload a medical image for AI-powered disease prediction")

    upload_col, result_col = st.columns([1, 1])

    with upload_col:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a medical image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    try:
                        uploaded_file.seek(0)
                        file_tuple = (
                            uploaded_file.name,
                            uploaded_file.read(),
                            uploaded_file.type or "application/octet-stream"
                        )
                        response = requests.post(
                            f"{API_URL}/api/predict-disease/",
                            files={"file": file_tuple},
                            timeout=60
                        )

                        if response.status_code == 200:
                            st.session_state.prediction_results = response.json()
                            st.success("Analysis complete!")
                        else:
                            st.error(f"Analysis failed: {response.status_code}")
                            if response.text:
                                st.error(f"Error details: {response.text}")

                    except requests.exceptions.RequestException as e:
                        st.error(f"Request failed: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

    with result_col:
        st.subheader("Analysis Results")

        if 'prediction_results' in st.session_state and st.session_state.prediction_results:
            results = st.session_state.prediction_results
            prediction = results.get('prediction', 'No prediction available')
            probabilities = results.get("probabilities", [])

            st.metric("Predicted Disease", prediction)

            if probabilities:
                try:
                    top_prob = max(probabilities)
                    st.metric("Confidence", f"{top_prob:.2%}")

                    st.divider()
                    st.subheader("Probability Distribution")

                    if len(probabilities) > 1:
                        labels = [f"Class {i}" for i in range(len(probabilities))]
                        df = pd.DataFrame({'Class': labels, 'Probability': probabilities})
                        df = df.sort_values('Probability', ascending=False)
                        st.bar_chart(df.set_index('Class'))

                        st.subheader("Top Predictions")
                        for _, row in df.head(3).iterrows():
                            st.write(f"**{row['Class']}**: {row['Probability']:.2%}")
                    else:
                        st.info("Single class prediction")

                except Exception as e:
                    st.error(f"Error processing probabilities: {e}")
                    st.json(probabilities)
            else:
                st.warning("No probability information available")

            st.divider()
            if st.button("💬 Ask about this prediction in chat"):
                question = f"I just received a disease prediction of '{prediction}' from an uploaded image. Can you tell me more about this condition and its remedies?"
                st.session_state.pending_input = question
                st.info("Go to the 'Ask Expert' tab to see the question.")
                st.rerun()
        else:
            st.info("Upload and analyze an image to see results here")

# -------------------- Dashboard Tab --------------------
with tab3:
    st.header("🌱 Find the Best Crop for Your Land")
    st.info("Enter your soil and environmental conditions below to get a crop recommendation.")

    # Load the preprocessor and model
    preprocessor, model = load_crop_model()

    if preprocessor and model:
        if st.button("Fetch Live Weather Data for Selected District", use_container_width=True):
            if not api_key_input:
                st.error("Please enter your OpenWeather API key in the sidebar.")
            else:
                with st.spinner(f"Fetching weather for {selected_district}..."):
                    coords = KERALA_DISTRICT_COORDS[selected_district]
                    weather_data = get_weather_data(coords[0], coords[1], api_key_input)
                    if weather_data:
                        # Update session state, which will automatically update the widgets
                        st.session_state.temp_val = round(weather_data["temperature"], 1)
                        st.session_state.humidity_val = round(weather_data["humidity"], 1)
                        st.session_state.rainfall_val = round(weather_data["total_rainfall"], 1)
                        st.success("✅ Weather data fetched and updated below!")
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            n_val = st.number_input("Nitrogen (N) Ratio in Soil", min_value=0, max_value=150, value=90, help="e.g., 90 kg/ha")
            p_val = st.number_input("Phosphorous (P) Ratio in Soil", min_value=0, max_value=150, value=45, help="e.g., 45 kg/ha")
            k_val = st.number_input("Potassium (K) Ratio in Soil", min_value=0, max_value=210, value=45, help="e.g., 45 kg/ha")
            ph_val = st.number_input("Soil pH Value", min_value=0.0, max_value=14.0, value=6.5, step=0.1, help="Scale from 0-14")

        with col2:
            temp_val = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=float(st.session_state.temp_val), step=0.1)
    
            humidity_val = st.slider("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=float(st.session_state.humidity_val), step=0.1)

            rainfall_val = st.number_input("Seasonal Rainfall (mm)",min_value=0.0,max_value=400.0,value=200.0, step=0.1,help="Enter the average rainfall for the entire crop season in mm (e.g., 250).")
        st.divider()

        if st.button("🌾 Predict Best Crop", use_container_width=True, type="primary"):
            try:
                # Create a DataFrame from the inputs in the correct order
                input_data = pd.DataFrame({
                    'N': [n_val],
                    'P': [p_val],
                    'K': [k_val],
                    'temperature': [temp_val],
                    'humidity': [humidity_val],
                    'ph': [ph_val],
                    'rainfall': [rainfall_val]
                })

                # 1. Scale the input data using the loaded preprocessor
                scaled_data = preprocessor.transform(input_data)
                
                # 2. Make a prediction using the loaded model
                prediction = model.predict(scaled_data)
                
                # 3. Display the result
                predicted_crop = prediction[0]
                st.success(f"### The recommended crop for these conditions is: **{predicted_crop.title()}**")

                # Add a button to ask the chatbot about this crop
                if st.button(f"💬 Ask about growing {predicted_crop.title()}"):
                    question = f"The model recommended I grow '{predicted_crop}'. Can you give me a guide on how to cultivate it in Kerala?"
                    st.session_state.pending_input = question
                    st.info("Go to the 'Ask Expert' tab to see the question.")
                    st.rerun()

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Crop recommendation model is currently unavailable. Please check the server logs.")



with tab4:
    st.header("🗓️ Crop Activity Planner (AI-Powered)")
    st.info(
        "Select a crop and your last activity. The AI assistant will read the "
        "Crop Weather Calendar to find the next step in your schedule."
    )

    crop_options = [
        "Paddy-I (Early Kharif)", "Paddy-II (Rabi)", "Coconut",
        "Pepper", "Banana", "Coffee", "Tapioca"
    ]
    col1, col2 = st.columns([1, 1])

    with col1:
        selected_crop_name = st.selectbox(
            "1. Select your Crop Type", options=crop_options
        )
        last_activity = st.text_input(
            "2. What activity did you just complete?",
            placeholder="e.g., Sowing, Transplanting, Flowering"
        )
        activity_date = st.date_input(
            "3. On what date did you complete it?", value=datetime.today()
        )

    with col2:
        st.write("### AI-Generated Next Steps")
        # --- PART 1: This button now ONLY fetches and saves the result ---
        if st.button("🤖 Ask Assistant for Next Step"):
            if not last_activity:
                st.warning("Please enter the activity you completed.")
            else:
                with st.spinner("Analyzing your schedule..."):
                    st.session_state.planner_result = None # Clear previous results
                    prompt = f"""
                    You are a highly precise agricultural schedule analyst. Your mission is to analyze the "Crop Weather Calendar for Kerala" document and respond ONLY with a valid JSON object. Do not include any conversational text or explanations.

                    First, analyze the schedule for the crop "{selected_crop_name}". Determine if it is a **linear schedule** (with clear, sequential steps like Paddy) or a **continuous schedule** (a perennial crop with overlapping activities like Coconut).

                    Based on your analysis, you MUST use one of the following three JSON formats for your response.

                    ---
                    **FORMAT 1: Use this for LINEAR schedules.**
                    If you find a clear, sequential next step for the activity "{last_activity}".

                    {{
                      "type": "linear_schedule",
                      "status": "On Schedule / Early / Late",
                      "next_activity": "Name of the next sequential stage",
                      "advice": "Provide specific, actionable advice for this next activity.",
                      "next_date": "YYYY-MM-DD"
                    }}

                    ---
                    **FORMAT 2: Use this for CONTINUOUS schedules.**
                    If you find the crop "{selected_crop_name}", but it does not have a clear sequential step after "{last_activity}" because it's a perennial crop.

                    {{
                      "type": "continuous_schedule",
                      "advice": "Provide general farm management advice relevant to the '{last_activity}' stage for this crop. For example: 'After flowering, focus on nutrient management and irrigation for coconut trees.'",
                      "details": "A single 'next step' is not applicable for this continuous-cycle crop."
                    }}

                    ---
                    **FORMAT 3: Use this if you CANNOT FIND the crop.**
                    If the document does not contain any information about the crop "{selected_crop_name}".

                    {{
                      "type": "error",
                      "message": "Could not find any information for '{selected_crop_name}' in the knowledge base."
                    }}

                    ---
                    **User's Request:**
                    - Crop: "{selected_crop_name}"
                    - Activity Just Completed: "{last_activity}"
                    - Date of Completion: "{activity_date.strftime('%Y-%m-%d')}"
                    """

                    ai_response_list = send_message(
                        thread_id=str(uuid.uuid4()),
                        messages=[{"role": "user", "content": prompt}],
                        language=language
                    )

                    if ai_response_list and 'content' in ai_response_list[0]:
                        try:
                            response_text = ai_response_list[0]['content']
                            json_start = response_text.find('{')
                            json_end = response_text.rfind('}') + 1
                            json_string = response_text[json_start:json_end]
                            st.session_state.planner_result = json.loads(json_string)
                        except (json.JSONDecodeError, ValueError):
                            st.session_state.planner_result = {"type": "error", "message": "The AI assistant returned an invalid format. Please try again."}
                    else:
                        st.session_state.planner_result = {"type": "error", "message": "Failed to get a response from the AI assistant."}

    # --- PART 2: This block now handles ALL displaying by reading from session_state ---
    if st.session_state.planner_result:
        data = st.session_state.planner_result
        response_type = data.get("type")

        if response_type == "linear_schedule":
            status = data.get("status", "N/A")
            next_activity = data.get("next_activity", "N/A")
            next_date_str = data.get("next_date")
            advice = data.get("advice", "")

            st.info(f"**Status for '{last_activity.title()}':** {status}")
            st.metric(label="Next Activity to Perform", value=next_activity)
            if next_date_str:
                next_date_obj = datetime.strptime(next_date_str, "%Y-%m-%d")
                st.metric(label="Ideal Start Date", value=next_date_obj.strftime("%B %d, %Y"))
            st.write(f"**Assistant's Advice:** {advice}")
            st.divider()

            if next_date_str and next_activity not in ["N/A", ""]:
                if st.button("🔔 Notify Me on this Date", use_container_width=True, type="primary"):
                    schedule_notification(activity=next_activity, date_str=next_date_str)

        elif response_type == "continuous_schedule":
            advice = data.get("advice", "No specific advice available.")
            details = data.get("details", "")
            st.success(f"**General Advice for {selected_crop_name}:**")
            st.write(advice)
            st.caption(details)

        elif response_type == "error":
            st.error(data.get("message", "An unknown error occurred."))
            
        else:
            st.warning("Received an unusual response from the assistant.")
            st.json(data)




with tab5:
    st.header("📈 Real-Time Mandi Price Lookup")
    st.info("Select your district, crop, and date to get the latest market prices from Agmarknet.")

    kerala_crops = [
        "Paddy", "Coconut", "Pepper", "Banana", "Rubber", "Cashew", "Tapioca",
        "Arecanut", "Cardamom", "Ginger", "Turmeric", "Coffee"
    ]

    # We will use two columns for a cleaner layout
    col1, col2 = st.columns(2)
    with col1:
        selected_district = st.selectbox(
            "1. Select Your District",
            options=list(KERALA_DISTRICTS),
            key="mandi_district"
        )
        selected_crop = st.selectbox(
            "2. Select Your Crop",
            options=kerala_crops,
            key="mandi_crop"
        )
    with col2:
        selected_date = st.date_input("3. Select Date", value=datetime.today())
        # <<< CHANGE START: Add new optional input for Market >>>
        selected_market = st.text_input("4. Enter Market Name (Optional)",
                                        placeholder="e.g., Koduvayoor")
        # <<< CHANGE END >>>

    if st.button("🔍 Fetch Market Prices", use_container_width=True, type="primary"):
        if not selected_crop:
            st.warning("Please select a crop.")
        else:
            with st.spinner(f"Fetching prices for {selected_crop}..."):
                # <<< CHANGE START: Pass the market to the function >>>
                price_data = fetch_mandi_prices(selected_date,
                                                selected_district,
                                                selected_crop,
                                                selected_market)
                if price_data:
                    # Convert data to a pandas DataFrame for easier handling
                    df = pd.DataFrame(price_data)
                    st.session_state.mandi_data = df # Save to session state
                else:
                    st.session_state.mandi_data = pd.DataFrame() # Clear old data if none found

    # Display the results if they exist in the session state
    if 'mandi_data' in st.session_state and not st.session_state.mandi_data.empty:
        df = st.session_state.mandi_data
        st.divider()
        st.subheader(f"Prices for '{selected_crop}' in {selected_district} on {selected_date.strftime('%d %B, %Y')}")

        # --- Display Key Metrics ---
        avg_modal_price = df['modal_price'].mean()
        max_price = df['max_price'].max()
        market_with_max_price = df.loc[df['max_price'].idxmax()]['market']

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric(
            label="Average Modal Price (per Quintal)",
            value=f"₹ {avg_modal_price:,.2f}"
        )
        metric_col2.metric(
            label="Highest Price Found",
            value=f"₹ {max_price:,.2f}",
            help=f"In {market_with_max_price} market"
        )

        # --- Display Data Table ---
        st.dataframe(
            df[['market', 'min_price', 'max_price', 'modal_price']],
            use_container_width=True
        )

        # --- Display Bar Chart ---
        st.subheader("Price Comparison Across Markets")
        chart_data = df.set_index('market')[['min_price', 'max_price', 'modal_price']]
        st.bar_chart(chart_data)                    