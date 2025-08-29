import streamlit as st
import requests
from typing import List, Dict

# -------------------- Backend API URL --------------------
API_URL = "http://localhost:8000"  # Change if your FastAPI server is running elsewhere

# -------------------- Utility Functions --------------------
def get_all_threads() -> List[str]:
    response = requests.get(f"{API_URL}/threads")
    if response.status_code == 200:
        return response.json()["threads"]
    return []

def create_new_thread() -> str:
    response = requests.post(f"{API_URL}/new_thread")
    if response.status_code == 200:
        return response.json()["thread_id"]
    return ""

def load_thread(thread_id: str) -> List[Dict]:
    response = requests.get(f"{API_URL}/load_thread/{thread_id}")
    if response.status_code == 200:
        return response.json()
    return []

def send_message(thread_id: str, messages: List[Dict], language: str = "English") -> List[Dict]:
    payload = {"thread_id": thread_id, "messages": messages, "language": language}
    response = requests.post(f"{API_URL}/chat", json=payload)
    if response.status_code == 200:
        return response.json()
    return []

# -------------------- Streamlit Session State --------------------
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = create_new_thread()

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()

if st.session_state['thread_id'] not in st.session_state['chat_threads']:
    st.session_state['chat_threads'].append(st.session_state['thread_id'])

# -------------------- Sidebar --------------------
st.sidebar.title("LangGraph Chatbot")
language = st.sidebar.selectbox(
    "Select response language",
    ["English", 'Malayalam', "Hindi", "Spanish", "French", "German", "Chinese", "Arabic"],
    index=0
)

if st.sidebar.button("New Chat"):
    new_thread_id = create_new_thread()
    st.session_state['thread_id'] = new_thread_id
    st.session_state['message_history'] = []
    if new_thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(new_thread_id)

st.sidebar.header("My Conversations")
for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        st.session_state['message_history'] = load_thread(thread_id)

# -------------------- Chat Messages --------------------
for msg in st.session_state['message_history']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])

user_input = st.chat_input("Type here")
if user_input:
    st.session_state['message_history'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    ai_messages = send_message(st.session_state['thread_id'], st.session_state['message_history'], language)
    
    for ai_msg in ai_messages:
        st.session_state['message_history'].append(ai_msg)
        with st.chat_message("assistant"):
            st.text(ai_msg['content'])

# -------------------- Prediction ------------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    import shutil, os
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    with open(file_path, "rb") as f:
        response = requests.post(f"{API_URL}/api/predict-disease/", files={"file": f})
    if response.status_code == 200:
        data = response.json()
        st.subheader("Predicted Disease:")
        st.write(data.get("prediction", "No prediction available"))
        probs = data.get("probabilities", [])
        if probs:
            st.write("Top Probability:", max(probs))
        else:
            st.write("No probabilities found")
    else:
        st.error("Error in prediction")
    os.remove(file_path)
