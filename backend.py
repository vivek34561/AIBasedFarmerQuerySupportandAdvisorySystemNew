from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, TypedDict, Annotated
import uuid, sqlite3, shutil, os
import tempfile
import json
import traceback
from typing import Optional
import requests
from datetime import date
import speech_recognition as sr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager
from advisory_engine import run_advisory_check # Import your main function
from pinecone import Pinecone
from models.prediction import PredictionPipeline
import logging
import sqlite3
from datetime import datetime
from twilio.rest import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()




def send_sms_notification(phone_number: str, message: str):
    """Sends an SMS using Twilio."""
    try:
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")

        if not all([account_sid, auth_token, twilio_phone_number]):
            logger.error("Twilio credentials are not fully configured in .env file.")
            return

        client = Client(account_sid, auth_token)

        # Ensure the recipient number is in E.164 format (e.g., +919151429036)
        # The number in your DB is already in this format, which is great.
        formatted_phone_number = phone_number

        message = client.messages.create(
            body=message,
            from_=twilio_phone_number,
            to=formatted_phone_number
        )
        logger.info(f"SMS notification sent successfully to {formatted_phone_number}, SID: {message.sid}")
    except Exception as e:
        logger.error(f"Failed to send SMS notification to {phone_number}: {e}")
        
        

# -------------------- RAG Setup --------------------
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
    
    model_name = os.getenv("HUGGINGFACE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    embedding_dimension = len(embeddings.embed_query("test"))
    logger.info(f"Using embedding model: {model_name} with dimension: {embedding_dimension}")
    
    if index_name in pc.list_indexes().names():
        index = pc.Index(index_name)
        index_stats = index.describe_index_stats()
        if index_stats.get('dimension') != embedding_dimension:
            logger.warning(f"Index dimension ({index_stats.get('dimension')}) doesn't match embedding dimension ({embedding_dimension})")
        
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text"
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    else:
        logger.warning(f"Pinecone index '{index_name}' not found. RAG features will be disabled.")
        retriever = None
except Exception as e:
    logger.error(f"Could not initialize Pinecone: {e}. RAG features will be disabled.")
    retriever = None

# -------------------- LLM Setup --------------------
from langchain_groq import ChatGroq

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0.7,
)

# -------------------- LangGraph Setup --------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # Keep track of language per-thread
    language: str 

def retrieve_context(query: str) -> str:
    if retriever is None: return ""
    try:
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return ""

def chat_node(state: ChatState):
    messages = state['messages']
    language = state.get('language', 'English') # Default to English
    
    # Get the last human message for context retrieval
    last_human_msg_content = ""
    if messages and isinstance(messages[-1], HumanMessage):
        last_human_msg_content = messages[-1].content
    
    if not last_human_msg_content:
        return {"messages": [AIMessage(content="I didn't receive a message to respond to.")]}

    # Prepare messages for the LLM
    llm_messages = list(messages)

    # <<< CHANGE START: More robust system prompt management
    # Check if a system message is already present
    has_system_message = any(isinstance(m, SystemMessage) for m in llm_messages)

    # If it's the start of the chat (no system message), add one.
    if not has_system_message:
        context = retrieve_context(last_human_msg_content)
        
        context_prompt = f"Use the following context to answer: {context}" if context else ""
        language_prompt = f"Your primary language for responding is {language}. Provide answers in {language} unless the user explicitly asks for another."

        system_content = f"""You are a helpful agricultural advisory assistant for farmers in Kerala. {language_prompt}
Be accurate, helpful, and provide practical, actionable advice. Consider local Kerala conditions.
If you don't know the answer, say so.
{context_prompt}"""
        
        system_msg = SystemMessage(content=system_content.strip())
        llm_messages.insert(0, system_msg)
    # <<< CHANGE END

    try:
        response = llm.invoke(llm_messages)
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"LLM invoke error: {e}")
        return {"messages": [AIMessage(content=f"Sorry, I encountered an error: {str(e)}")]}

# -------------------- SQLite & Graph Compilation --------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
# Join that directory path with the database filename
db_path = os.path.join(script_dir, "chatbot.db")
print(f"--- Connecting to database at: {db_path} ---") # For debugging

conn = sqlite3.connect(database=db_path, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
def create_db_and_tables():
    print("--- Initializing Database ---")
    # Use a temporary connection for setup to avoid conflicts
    setup_conn = sqlite3.connect(db_path)
    cursor = setup_conn.cursor()

    # Create users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phone_number TEXT NOT NULL UNIQUE,
        district TEXT NOT NULL,
        primary_crop TEXT
    );
    """)
    print("Table 'users' created or already exists.")
    
    # Create escalations table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS escalations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id TEXT NOT NULL,
        chat_history TEXT,
        status TEXT DEFAULT 'open',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    print("Table 'escalations' created or already exists.")
    
    # Create feedback table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id TEXT NOT NULL,
        message_index INTEGER NOT NULL,
        rating INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    print("Table 'feedback' created or already exists.")

    setup_conn.commit()
    setup_conn.close()
create_db_and_tables()    
    
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)


scheduler = BackgroundScheduler()
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Code to run on startup ---
    print("--- Initializing application ---")
    create_db_and_tables()
    
    # Schedule the job to run every day at 7:00 AM Kerala time (IST)
    # scheduler.add_job(run_advisory_check, 'cron', hour=19, minute=44, timezone='Asia/Kolkata')
    scheduler.start()
    print("APScheduler started...")
    
    yield # The application is now ready to run and accept requests
    
    # --- Code to run on shutdown ---
    print("--- Shutting down application ---")
    scheduler.shutdown()
    print("APScheduler shut down.")


# -------------------- FastAPI Setup --------------------
app = FastAPI(title="RAG LangGraph Chatbot API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------- Pydantic Models --------------------
class MandiPrice(BaseModel):
    market: str
    commodity: str
    min_price: float
    max_price: float
    modal_price: float

class MandiPriceResponse(BaseModel):
    data: List[MandiPrice]
    message: str

class NotificationRequest(BaseModel):
    activity_name: str
    notify_date: str # Expecting "YYYY-MM-DD" format
    
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    thread_id: str  # Made thread_id mandatory for clarity
    messages: List[Message]
    language: str = "English"

class NewThreadResponse(BaseModel):
    thread_id: str
    
# ... (other Pydantic models are fine) ...
class ThreadListResponse(BaseModel):
    threads: List[str]

class DocumentListResponse(BaseModel):
    documents: List[str]

class ProcessingStatusResponse(BaseModel):
    total_documents: int
    processed_chunks: int

class UserProfile(BaseModel):
    phone_number: str
    district: str
    primary_crop: Optional[str] = None
class FeedbackRequest(BaseModel):
    thread_id: str
    message_index: int
    rating: int # e.g., 1 for upvote, -1 for downvote

class EscalateRequest(BaseModel):
    thread_id: str
# -------------------- Utility Functions --------------------
def generate_thread_id():
    return str(uuid.uuid4())

def retrieve_all_threads():
    """
    Definitive Version: Correctly sorts conversations using the 'checkpoint_id' column.
    """
    try:
        with conn:
            cursor = conn.cursor()
            # Step 1: Select all thread_ids, correctly sorted by the chronological checkpoint_id.
            cursor.execute(
                "SELECT thread_id FROM checkpoints ORDER BY checkpoint_id DESC"
            )
            rows = cursor.fetchall()

            # Step 2: Create a unique list in Python, which preserves the correct sorted order.
            unique_thread_ids = []
            seen_ids = set()
            for row in rows:
                thread_id = row[0]
                if thread_id not in seen_ids:
                    unique_thread_ids.append(thread_id)
                    seen_ids.add(thread_id)
            
            return unique_thread_ids
            
    except Exception as e:
        if "no such table" in str(e) or "no such column" in str(e):
            logger.warning(f"Database query failed (table or column might be missing): {e}")
            return []
        
        logger.error(f"Failed to retrieve threads directly from database: {e}")
        traceback.print_exc()
        return []


def convert_messages(messages: List[Message]) -> List[BaseMessage]:
    converted = []
    for m in messages:
        if m.role == 'user':
            converted.append(HumanMessage(content=m.content))
        elif m.role == 'assistant':
            converted.append(AIMessage(content=m.content))
    return converted

# -------------------- API Endpoints --------------------
# In backend.py

@app.get("/mandi-prices", response_model=MandiPriceResponse)
def get_mandi_prices(arrival_date: date, district: str, commodity: str, market: Optional[str] = None):
    """
    Fetches crop prices for a given district, commodity, and optional market from data.gov.in.
    """
    api_key = os.getenv("DATA_GOV_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="DATA_GOV_API_KEY is not configured on the server.")

    formatted_date = arrival_date.strftime("%d-%b-%Y")
    api_url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

    params = {
        "api-key": api_key,
        "format": "json",
        "offset": "0",
        "limit": "100",
        "filters[state.keyword]": "Kerala",
        "filters[district]": district,
        "filters[market]": market,
        "filters[commodity]": commodity,
        "filters[variety]": "Palayamthodan",
        "filters[grade]": "Medium"
    }
    if market:
        params["filters[market]"] = market.title()
    # <<< ADD THIS LINE FOR DEBUGGING >>>
    logger.info(f"Requesting Mandi data with params: {params}")

    try:
        response = requests.get(api_url, params=params, timeout=20)
        response.raise_for_status()
        api_data = response.json()

        records = api_data.get("records", [])
        if not records:
            return MandiPriceResponse(data=[], message="No price data found for the selected crop, district, and date.")

        price_data = []
        for record in records:
            try:
                price_data.append(MandiPrice(
                    market=record.get("market", "N/A").strip(),
                    commodity=record.get("commodity", "N/A").strip(),
                    min_price=float(record.get("min_price", 0.0)),
                    max_price=float(record.get("max_price", 0.0)),
                    modal_price=float(record.get("modal_price", 0.0))
                ))
            except (ValueError, TypeError):
                logger.warning(f"Skipping record with invalid price data: {record}")
                continue
        
        return MandiPriceResponse(data=price_data, message="Data fetched successfully.")

    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error fetching mandi data: {e.response.text}")
        # Pass the original error detail from the government portal if available
        error_detail = f"Failed to fetch data from the government portal. Status: {e.response.status_code}"
        raise HTTPException(status_code=500, detail=error_detail)
    except requests.exceptions.RequestException as e:
        logger.error(f"Request Error fetching mandi data: {e}")
        raise HTTPException(status_code=503, detail="Could not connect to the government data portal. Please try again later.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching mandi prices: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


@app.post("/schedule-notification")
def schedule_notification(request: NotificationRequest):
    """Schedules a one-time SMS notification for a user."""
    try:
        # For this implementation, we'll notify the most recently added user.
        # In a multi-user system, you would link this to a logged-in user's ID.
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT phone_number FROM users ORDER BY id DESC LIMIT 1")
            user_row = cursor.fetchone()

        if not user_row:
            raise HTTPException(status_code=404, detail="No registered user found in the database.")

        user_phone_number = user_row[0]
        notification_date = datetime.strptime(request.notify_date, "%Y-%m-%d")

        # Create a user-friendly message for the SMS
        reminder_message = (
            f"Reminder from Digital Krishi Officer: "
            f"It's time to start your next farming activity: '{request.activity_name}'. "
            f"Scheduled to begin around {notification_date.strftime('%B %d, %Y')}."
        )

        # Schedule the job using the existing scheduler instance
        scheduler.add_job(
            send_sms_notification,
            trigger='date',
            run_date=notification_date,
            args=[user_phone_number, reminder_message],
            id=f"notification_{user_phone_number}_{uuid.uuid4()}", # Unique ID for the job
            replace_existing=False
        )
        logger.info(f"Notification scheduled for {user_phone_number} on {request.notify_date}")
        return {"status": "success", "message": f"Notification scheduled for {request.notify_date}."}

    except Exception as e:
        logger.error(f"Failed to schedule notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Optional: Add a testing endpoint to trigger the check manually
@app.post("/trigger-advisory-manually")
def trigger_advisory():
    run_advisory_check()
    return {"status": "Advisory check triggered manually."}


@app.post("/register-user")
def register_user(user: UserProfile):
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (phone_number, district, primary_crop) VALUES (?, ?, ?)",
                (user.phone_number, user.district, user.primary_crop)
            )
            conn.commit()
        return {"status": "success", "message": f"User {user.phone_number} registered."}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Phone number already registered.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-schema")
def debug_schema():
    """
    A temporary endpoint to read the exact schema of the checkpoints table.
    """
    try:
        with conn:
            cursor = conn.cursor()
            # This command asks the database to describe the 'checkpoints' table
            cursor.execute("PRAGMA table_info(checkpoints);")
            schema_info = cursor.fetchall()
            
            if not schema_info:
                return {"error": "Could not retrieve schema. The 'checkpoints' table may not exist."}

            # Format the result into a readable JSON
            columns = [
                {"column_index": row[0], "name": row[1], "type": row[2], "can_be_null": not row[3]}
                for row in schema_info
            ]
            return {"table_name": "checkpoints", "schema": columns}
            
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.post("/chat", response_model=List[Message])
def chat_endpoint(request: ChatRequest):
    # <<< CHANGE START: This is the main fix.
    # We no longer pass the whole history. LangGraph's checkpointer handles that.
    # We only pass the NEWEST message from the user.
    
    thread_id = request.thread_id
    CONFIG = {'configurable': {'thread_id': thread_id}}
    
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided.")
        
    # Extract only the last message from the list sent by the frontend
    last_user_message = request.messages[-1]
    if last_user_message.role != 'user':
        raise HTTPException(status_code=400, detail="Last message must be from the user.")

    # Convert just the new message
    new_message_converted = HumanMessage(content=last_user_message.content)
    
    try:
        # Invoke the chatbot with only the new message and the language preference
        # The checkpointer will load the previous messages for this thread_id automatically
        response = chatbot.invoke(
            {
                'messages': [new_message_converted],
                'language': request.language
            }, 
            config=CONFIG
        )
        
        # The graph's response contains the new AI message(s).
        # We find the last AIMessage, which is our reply.
        ai_reply = None
        for msg in reversed(response['messages']):
            if isinstance(msg, AIMessage):
                ai_reply = {"role": "assistant", "content": msg.content}
                break
        
        if ai_reply:
            return [ai_reply]
        else:
            raise HTTPException(status_code=500, detail="AI did not generate a response.")

    except Exception as e:
        logger.error(f"Chat endpoint error for thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    # <<< CHANGE END

@app.post("/new_thread", response_model=NewThreadResponse)
def new_thread():
    thread_id = generate_thread_id()
    return NewThreadResponse(thread_id=thread_id)
    
@app.get("/load_thread/{thread_id}", response_model=List[Message])
def load_thread(thread_id: str):
    try:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        
        # State can be None if thread doesn't exist
        if not state:
            return []
            
        messages = state.values.get('messages', [])
        formatted = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            elif isinstance(msg, AIMessage):
                role = 'assistant'
            # We don't need to send the system message to the frontend
            elif isinstance(msg, SystemMessage):
                continue
            else:
                continue
            formatted.append({'role': role, 'content': msg.content})
        return formatted
    except Exception as e:
        # Catch cases where the thread might not exist in the checkpointer
        logger.error(f"Could not load thread {thread_id}: {e}")
        return []

# ... (The rest of your endpoints like /voice_query, /documents, etc., are fine and remain unchanged) ...
@app.post("/voice_query")
async def voice_query(file: UploadFile = File(...), language: str = Form("Malayalam")):
    """
    Accepts a voice query (audio file), transcribes it, and returns RAG chatbot answer.
    """
    try:
        # Save uploaded audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            shutil.copyfileobj(file.file, temp_audio)
            temp_audio_path = temp_audio.name

        # Transcribe audio
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)
            # Use Malayalam or selected language
            lang_code = {
                "Malayalam": "ml-IN", "English": "en-US", "Hindi": "hi-IN", "Spanish": "es-ES", "French": "fr-FR", "German": "de-DE", "Chinese": "zh-CN", "Arabic": "ar-SA"
            }.get(language, "ml-IN")
            try:
                query_text = recognizer.recognize_google(audio_data, language=lang_code)
            except sr.UnknownValueError:
                os.remove(temp_audio_path)
                raise HTTPException(status_code=400, detail="Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                os.remove(temp_audio_path)
                raise HTTPException(status_code=503, detail=f"Could not request results from Google Speech Recognition service; {e}")
            # <<< CHANGE END

        os.remove(temp_audio_path)

        # Pass transcribed text to RAG chatbot
        # Use a new thread for each voice query (or you can use session)
        thread_id = str(uuid.uuid4())
        messages = [HumanMessage(content=query_text)]
        CONFIG = {'configurable': {'thread_id': thread_id}}
        response = chatbot.invoke({'messages': messages, 'language': language}, config=CONFIG)
        
        ai_messages = response['messages']
        # Find the last AI message in the response
        answer = "No answer generated."
        for msg in reversed(ai_messages):
            if isinstance(msg, AIMessage):
                answer = msg.content
                break
        return {"transcription": query_text}
    except Exception as e:
        # Clean up temp file in case of an early error
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        logger.error(f"Voice query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice query failed: {str(e)}")

@app.get("/threads", response_model=ThreadListResponse)
def get_all_threads():
    threads = retrieve_all_threads()
    return ThreadListResponse(threads=threads)
    
@app.get("/documents", response_model=DocumentListResponse)
def get_documents():
    """Get list of documents in the knowledge base"""
    try:
        documents_folder = "documents"
        if os.path.exists(documents_folder):
            documents = [f for f in os.listdir(documents_folder) if os.path.isfile(os.path.join(documents_folder, f))]
            return DocumentListResponse(documents=documents)
        return DocumentListResponse(documents=[])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/processing_status", response_model=ProcessingStatusResponse)
def get_processing_status():
    """Get the status of document processing"""
    try:
        documents_folder = "documents"
        total_docs = 0
        if os.path.exists(documents_folder):
            total_docs = len([f for f in os.listdir(documents_folder) if os.path.isfile(os.path.join(documents_folder, f))])
        
        # Get chunk count from Pinecone if available
        processed_chunks = 0
        if retriever and index_name in pc.list_indexes().names():
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            processed_chunks = stats.get('total_vector_count', 0)
        
        return ProcessingStatusResponse(total_documents=total_docs, processed_chunks=processed_chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@app.post("/api/predict-disease/")
async def predict_disease(file: UploadFile = File(...)):
    try:
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pipeline = PredictionPipeline(temp_file_path)
        result = pipeline.predict()

        os.remove(temp_file_path)
        
        # Ensure result has the expected structure
        if result and isinstance(result, list) and "image" in result[0]:
            return {"prediction": result[0]["image"], "probabilities": result[0].get("probabilities")}
        else:
            return {"error": "Prediction result in unexpected format."}
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Clean up temp file in case of error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/log-feedback")
def log_feedback(request: FeedbackRequest):
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO feedback (thread_id, message_index, rating) VALUES (?, ?, ?)",
                (request.thread_id, request.message_index, request.rating)
            )
            conn.commit()
        return {"status": "success", "message": "Feedback logged."}
    except Exception as e:
        logger.error(f"Feedback logging error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/escalate-query")
def escalate_query(request: EscalateRequest):
    try:
        # Get the full conversation history for the thread
        thread_state = chatbot.get_state(config={'configurable': {'thread_id': request.thread_id}})
        if not thread_state:
            raise HTTPException(status_code=404, detail="Thread not found.")

        messages = thread_state.values.get('messages', [])
        # Convert messages to a JSON-serializable format
        history_list = [{"type": msg.__class__.__name__, "content": msg.content} for msg in messages]
        chat_history_json = json.dumps(history_list, indent=2)

        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO escalations (thread_id, chat_history) VALUES (?, ?)",
                (request.thread_id, chat_history_json)
            )
            conn.commit()
        
        return {"status": "success", "message": "Query has been escalated to an expert."}
    except Exception as e:
        logger.error(f"Escalation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend:app", host="0.0.0.0", port=port)