import streamlit as st
import requests
from typing import List, Dict
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import os
print(sr.__version__)
# -------------------- Backend API URL --------------------
API_URL = "http://localhost:8000"

# -------------------- Utility Functions --------------------
def get_all_threads() -> List[str]:
    response = requests.get(f"{API_URL}/threads")
    return response.json().get("threads", []) if response.status_code == 200 else []

def upload_document(file) -> Dict:
    """Upload a document to the RAG system"""
    files = {"file": (file.name, file, file.type)}
    response = requests.post(f"{API_URL}/upload_document", files=files)
    return response.json() if response.status_code == 200 else {"success": False, "message": "Upload failed"}

def get_documents() -> List[str]:
    """Get list of documents in the RAG system"""
    response = requests.get(f"{API_URL}/documents")
    return response.json().get("documents", []) if response.status_code == 200 else []

def refresh_documents() -> Dict:
    """Refresh the document vector store"""
    response = requests.post(f"{API_URL}/refresh_documents")
    return response.json() if response.status_code == 200 else {"success": False, "message": "Refresh failed"}

def force_process_documents() -> Dict:
    """Force process all documents in the folder"""
    response = requests.post(f"{API_URL}/force_process_documents")
    return response.json() if response.status_code == 200 else {"success": False, "message": "Force processing failed"}

def get_processing_status() -> Dict:
    """Get the current processing status"""
    response = requests.get(f"{API_URL}/processing_status")
    return response.json() if response.status_code == 200 else {"error": "Failed to get status"}

def create_new_thread() -> str:
    response = requests.post(f"{API_URL}/new_thread")
    return response.json().get("thread_id", "") if response.status_code == 200 else ""

def load_thread(thread_id: str) -> List[Dict]:
    response = requests.get(f"{API_URL}/load_thread/{thread_id}")
    return response.json() if response.status_code == 200 else []

def send_message(thread_id: str, messages: List[Dict], language: str = "English") -> List[Dict]:
    payload = {"thread_id": thread_id, "messages": messages, "language": language}
    response = requests.post(f"{API_URL}/chat", json=payload)
    return response.json() if response.status_code == 200 else []

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
    ["English", "Malayalam", "Hindi", "Spanish", "French", "German", "Chinese", "Arabic"]
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

# -------------------- Document Management --------------------
st.sidebar.header("📚 Document Management")

# Document upload
st.sidebar.subheader("Upload Document")
uploaded_doc = st.sidebar.file_uploader(
    "Upload documents for RAG",
    type=["pdf", "txt", "docx"],
    help="Upload PDF, TXT, or DOCX files to enhance the chatbot's knowledge base"
)

if uploaded_doc:
    if st.sidebar.button("Upload to Knowledge Base"):
        with st.sidebar.spinner("Uploading and processing document..."):
            result = upload_document(uploaded_doc)
            if result.get("success", False):
                st.sidebar.success(f"✅ {result.get('message', 'Document uploaded successfully!')}")
                st.rerun()
            else:
                st.sidebar.error(f"❌ {result.get('message', 'Upload failed')}")

# Document processing status
st.sidebar.subheader("📊 Processing Status")
try:
    status = get_processing_status()
    if "error" not in status:
        processed = status.get("processed_count", 0)
        total = status.get("total_count", 0)
        new = status.get("new_count", 0)
        
        # Status indicator
        if new == 0:
            st.sidebar.success(f"✅ Up to date! ({processed}/{total} documents processed)")
        else:
            st.sidebar.warning(f"⚠️ {new} new documents need processing ({processed}/{total} processed)")
        
        # Show processed documents
        if processed > 0:
            st.sidebar.write(f"📄 **Processed documents ({processed}):**")
            for doc in status.get("processed_documents", [])[:5]:  # Show first 5
                st.sidebar.write(f"• {doc}")
            if processed > 5:
                st.sidebar.write(f"... and {processed - 5} more")
        
        # Show new documents
        if new > 0:
            st.sidebar.write(f"🆕 **New documents ({new}):**")
            for doc in status.get("new_documents", []):
                st.sidebar.write(f"• {doc}")
    else:
        st.sidebar.error(f"Error getting status: {status.get('error')}")
except Exception as e:
    st.sidebar.error(f"Error loading status: {e}")

# List all documents in folder
st.sidebar.subheader("📁 All Documents in Folder")
try:
    documents = get_documents()
    if documents:
        st.sidebar.write(f"📄 **{len(documents)} documents found:**")
        for doc in documents:
            st.sidebar.write(f"• {doc}")
    else:
        st.sidebar.write("📭 No documents in folder")
except Exception as e:
    st.sidebar.error(f"Error loading documents: {e}")

# Document processing buttons
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("🔄 Refresh"):
        with st.sidebar.spinner("Refreshing knowledge base..."):
            result = refresh_documents()
            if result.get("success", False):
                st.sidebar.success("✅ Knowledge base refreshed!")
            else:
                st.sidebar.error("❌ Failed to refresh knowledge base")

with col2:
    if st.button("⚡ Force Process"):
        with st.sidebar.spinner("Processing all documents..."):
            result = force_process_documents()
            if result.get("success", False):
                st.sidebar.success("✅ All documents processed!")
                st.rerun()
            else:
                st.sidebar.error("❌ Failed to process documents")

# -------------------- Main Chat Area --------------------
st.title("🤖 RAG-Enhanced Chatbot")
st.info("💡 **How it works:** This chatbot uses Retrieval Augmented Generation (RAG) to answer questions based on your uploaded documents. Upload PDFs, Word docs, or text files to the knowledge base, and the chatbot will use them to provide contextual answers!")

# -------------------- Chat Messages --------------------
for msg in st.session_state['message_history']:
    with st.chat_message(msg['role']):
        st.text(msg['content'])

# -------------------- Voice Input --------------------
st.subheader("Voice Input")
st.write("Click start and speak. Your speech will be converted to text.")

recognizer = sr.Recognizer()

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recognized_text = ""

    def recv(self, frame):
        return frame

LANGUAGE_CODES = {
    "English": "en-US",
    "Hindi": "hi-IN",
    "Malayalam": "ml-IN",
    "Spanish": "es-ES",
    "French": "fr-FR",
    "German": "de-DE",
    "Chinese": "zh-CN",
    "Arabic": "ar-SA"
}
webrtc_ctx = webrtc_streamer(
    key="langgraph-voice",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
)

if webrtc_ctx.audio_receiver:
    try:
        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        if audio_frames:
            import wave
            temp_file = "temp_audio.wav"
            with wave.open(temp_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                for frame in audio_frames:
                    wf.writeframes(frame.to_bytes())
            with sr.AudioFile(temp_file) as source:
                audio_data = recognizer.record(source)
                # Use selected language code
                lang_code = LANGUAGE_CODES.get(language, "en-US")
                user_input = recognizer.recognize_google(audio_data, language=lang_code)
            os.remove(temp_file)
            st.success(f"Recognized text: {user_input}")
        else:
            user_input = st.chat_input("Type here")
    except Exception as e:
        st.error(f"Speech recognition error: {e}")
        user_input = st.chat_input("Type here")
else:
    user_input = st.chat_input("Type here")

# -------------------- Send Chat --------------------
# -------------------- Send Chat --------------------
# -------------------- Send Chat --------------------
# -------------------- Send Chat --------------------
if user_input:
    # Append user message
    st.session_state['message_history'].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # Send the FULL message history to backend for context
    ai_messages = send_message(
        st.session_state['thread_id'],
        st.session_state['message_history'],
        language
    )

    # Show assistant responses
    for ai_msg in ai_messages:
        with st.chat_message("assistant"):
            st.text(ai_msg['content'])
        st.session_state['message_history'].append(ai_msg)


# -------------------- Prediction ------------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    import shutil
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
