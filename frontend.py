import streamlit as st
import requests
from typing import List, Dict
import os
import uuid
import json

# To be installed: pip install streamlit-webrtc
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import speech_recognition as sr
import av

# -------------------- Configuration --------------------
API_URL = "http://localhost:8000"

# -------------------- Utility Functions --------------------
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
        response = requests.post(f"{API_URL}/chat", json=payload, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to send message: {e}")
        return []

# -------------------- Session State Initialization --------------------
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = create_new_thread() or str(uuid.uuid4())

if 'message_history' not in st.session_state:
    st.session_state.message_history = []

if 'chat_threads' not in st.session_state:
    threads = get_all_threads()
    st.session_state.chat_threads = threads if threads else [st.session_state.thread_id]

# store voice language and latest transcription
if 'voice_lang' not in st.session_state:
    st.session_state.voice_lang = "English"
if 'last_transcription' not in st.session_state:
    st.session_state.last_transcription = ""

# -------------------- Main UI --------------------
st.set_page_config(page_title="RAG-Enhanced Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 RAG-Enhanced Chatbot")
st.info("This chatbot uses RAG (documents in the `documents` folder) plus LLM chat threads.")

# Sidebar
st.sidebar.title("LangGraph Chatbot Controls")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    new_id = create_new_thread()
    if new_id:
        st.session_state.thread_id = new_id
    else:
        st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.message_history = []
    # refresh threads list (clear cache if needed)
    st.session_state.chat_threads = get_all_threads() or [st.session_state.thread_id]
    st.rerun()

st.sidebar.header("My Conversations")
for idx, thread_id in enumerate(st.session_state.chat_threads):
    short = thread_id[:8]
    if st.sidebar.button(f"💬 Chat: {short}...", key=f"chat_{idx}"):
        st.session_state.thread_id = thread_id
        st.session_state.message_history = load_thread(thread_id)
        st.rerun()


st.sidebar.markdown("---")
st.sidebar.header("📚 RAG Knowledge Base")
with st.sidebar.expander("System Status", expanded=True):
    status = get_processing_status()
    total_docs = status.get("total_documents", 0)
    processed_chunks = status.get("processed_chunks", 0)

    if total_docs > 0 and processed_chunks > 0:
        st.markdown("**Status:** 🟢 Ready")
    elif total_docs > 0:
        st.markdown("**Status:** 🟠 Processing...")
    else:
        st.markdown("**Status:** ⚪ Idle / No documents")

    st.markdown(f"**Documents Found:** {total_docs}")
    st.markdown(f"**Processed Chunks:** {processed_chunks}")

    st.subheader("Existing Documents")
    documents = get_documents()
    if documents:
        for doc in documents:
            st.markdown(f"- {doc}")
    else:
        st.markdown("- No documents found.")

# Conversation area
chat_col, side_col = st.columns([3, 1])

with chat_col:
    st.header("Conversation")
    # display history
    for msg in st.session_state.message_history:
        role = msg.get('role', 'assistant')
        content = msg.get('content', '')
        with st.chat_message(role):
            st.markdown(content)

    # Text input area
    user_input = st.chat_input("Type your message here...")

    # Button to use latest transcription (if any)
    if st.session_state.last_transcription:
        if st.button("Use latest transcription as input"):
            user_input = st.session_state.last_transcription

    if user_input:
        # append to history and render
        st.session_state.message_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # send to backend and get reply
        with st.spinner("Thinking..."):
            ai_messages = send_message(st.session_state.thread_id, st.session_state.message_history, st.session_state.voice_lang)

        if ai_messages:
            for ai_msg in ai_messages:
                with st.chat_message("assistant"):
                    st.markdown(ai_msg.get('content', ''))
                st.session_state.message_history.append(ai_msg)

with side_col:
    st.header("Voice Input")
    st.write("Select recognition language and click Start to speak.")
    language = st.selectbox(
        "Speech language",
        ["English", "Malayalam", "Hindi", "Spanish", "French", "German", "Chinese", "Arabic"],
        index=["English", "Malayalam", "Hindi", "Spanish", "French", "German", "Chinese", "Arabic"].index(st.session_state.voice_lang) if st.session_state.voice_lang in ["English", "Malayalam", "Hindi", "Spanish", "French", "German", "Chinese", "Arabic"] else 0
    )
    # keep voice_lang in session state
    st.session_state.voice_lang = language

    st.write("Click Start and speak. After stopping, click 'Use latest transcription...' to copy it to the chat input.")

    class AudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.recognizer = sr.Recognizer()
            self.audio_data = None
            self.transcribed_text = ""

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            raw_audio = frame.to_ndarray(format="s16", layout="mono")
            self.audio_data = sr.AudioData(raw_audio.tobytes(), frame.sample_rate, 2)
            # We do not block the UI; put transcription into session_state when possible
            try:
                lang_code = {
                    "English": "en-US", "Hindi": "hi-IN", "Malayalam": "ml-IN", "Spanish": "es-ES",
                    "French": "fr-FR", "German": "de-DE", "Chinese": "zh-CN", "Arabic": "ar-SA"
                }.get(st.session_state.voice_lang, "en-US")
                text = self.recognizer.recognize_google(self.audio_data, language=lang_code)
                # Save last transcription to session state
                st.session_state.last_transcription = text
            except sr.UnknownValueError:
                # do not spam errors — store empty
                st.session_state.last_transcription = ""
            except sr.RequestError:
                st.session_state.last_transcription = ""
            return frame

    # Start/webrtc streamer
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

# -------------------- Prediction Section --------------------
st.markdown("---")
st.header("🦠 Disease Prediction")
st.info("Upload an image (jpg/jpeg/png). The image will be sent to the backend model for prediction.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if uploaded_file:
    with st.spinner("Analyzing image..."):
        try:
            # Build proper multipart file tuple expected by requests
            file_tuple = (uploaded_file.name, uploaded_file.read(), uploaded_file.type or "application/octet-stream")
            response = requests.post(f"{API_URL}/api/predict-disease/", files={"file": file_tuple}, timeout=60)
            if response.status_code == 200:
                data = response.json()
                st.subheader("Prediction Result:")
                st.write(f"**Predicted Disease:** {data.get('prediction', 'No prediction available')}")
                probabilities = data.get("probabilities", [])
                if probabilities:
                    try:
                        # display top probability
                        top = max(probabilities)
                        st.write(f"**Top Probability:** {top:.2%}")
                    except Exception:
                        st.write("Probabilities:", probabilities)
                    st.markdown("---")
                    st.write("**All Probabilities:**")
                    st.json(probabilities)
                else:
                    st.write("No probabilities available.")
            else:
                st.error("An error occurred during prediction.")
                st.write(f"Status Code: {response.status_code}")
                try:
                    st.write(f"Error Message: {response.json().get('detail', 'N/A')}")
                except Exception:
                    st.write("Could not decode error message.")
        except requests.exceptions.RequestException as e:
            st.error(f"Prediction request failed: {e}")
