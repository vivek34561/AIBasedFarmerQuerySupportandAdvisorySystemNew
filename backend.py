from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, TypedDict, Annotated
import uuid, sqlite3, shutil, os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from models.prediction import PredictionPipeline
from rag_system import get_rag_system
from googletrans import Translator
translator = Translator()
load_dotenv()



LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Malayalam": "ml",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh-cn",
    "Arabic": "ar"
}

# -------------------- Chatbot Setup --------------------
llm = ChatOpenAI(temperature=0.7)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage

def chat_node(state: ChatState, config: RunnableConfig):
    messages = state['messages']
    target_language = (
        (config.get("configurable") or {}).get("target_language") or "English"
    )

    latest = messages[-1] if messages else None
    if latest and hasattr(latest, "content"):
        user_text = latest.content
        # translate input to English
        try:
            user_text_en = translator.translate(user_text, dest="en").text
        except Exception:
            user_text_en = user_text

        rag = get_rag_system()
        response_content_en = rag.query(user_text_en)

        # translate back if needed
        if target_language.lower() != "english":
            dest = LANGUAGE_CODES.get(target_language, "en")
            try:
                response_content = translator.translate(response_content_en, dest=dest).text
            except Exception:
                response_content = response_content_en
        else:
            response_content = response_content_en

        return {"messages": [AIMessage(content=response_content)]}

    # fallback (should rarely hit)
    return {"messages": [AIMessage(content="")]}


# -------------------- SQLite Setup --------------------
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

# -------------------- FastAPI Setup --------------------
app = FastAPI(title="LangGraph Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------- Pydantic Models --------------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    thread_id: Optional[str] = None
    messages: List[Message]
    language: Optional[str] = "English"

class NewThreadResponse(BaseModel):
    thread_id: str

class ThreadListResponse(BaseModel):
    threads: List[str]

class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None

class DocumentListResponse(BaseModel):
    documents: List[str]
    count: int

# -------------------- Utility Functions --------------------
def generate_thread_id():
    return str(uuid.uuid4())

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

def convert_to_human(messages: List[Message]):
    return [HumanMessage(content=m.content) for m in messages if m.role == 'user']

# -------------------- API Endpoints --------------------
@app.get("/threads", response_model=ThreadListResponse)
def get_all_threads():
    threads = retrieve_all_threads()
    return ThreadListResponse(threads=threads)

@app.post("/new_thread", response_model=NewThreadResponse)
def new_thread():
    thread_id = generate_thread_id()
    return NewThreadResponse(thread_id=thread_id)

@app.get("/load_thread/{thread_id}", response_model=List[Message])
def load_thread(thread_id: str):
    try:
        messages = chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']
        formatted = []
        for msg in messages:
            role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
            formatted.append({'role': role, 'content': msg.content})
        return formatted
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found: {str(e)}")


from langchain_core.messages import HumanMessage, AIMessage

def convert_to_langchain_messages(messages: List[Message]):
    result = []
    for m in messages:
        if m.role == 'user':
            result.append(HumanMessage(content=m.content))
        elif m.role == 'assistant':
            result.append(AIMessage(content=m.content))
    return result

@app.post("/chat", response_model=List[Message])
def chat_endpoint(request: ChatRequest):
    thread_id = request.thread_id or generate_thread_id()
    CONFIG = {
        'configurable': {
            'thread_id': thread_id,
            'target_language': request.language or "English"
        }
    }
    lc_messages = convert_to_langchain_messages(request.messages)

    response = chatbot.invoke({'messages': lc_messages}, config=CONFIG)
    ai_messages = response['messages']
    return [{"role": "assistant", "content": msg.content} for msg in ai_messages]


@app.post("/upload_document", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the RAG system"""
    try:
        # Check file type
        allowed_extensions = ['.pdf', '.txt', '.docx']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return DocumentUploadResponse(
                success=False,
                message=f"File type {file_ext} not supported. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Save temporary file
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Add to RAG system
        rag = get_rag_system()
        success = rag.add_document(temp_file, file.filename)
        
        # Clean up temp file
        os.remove(temp_file)
        
        if success:
            return DocumentUploadResponse(
                success=True,
                message="Document uploaded and processed successfully",
                filename=file.filename
            )
        else:
            return DocumentUploadResponse(
                success=False,
                message="Failed to process document"
            )
            
    except Exception as e:
        return DocumentUploadResponse(
            success=False,
            message=f"Error uploading document: {str(e)}"
        )

@app.get("/documents", response_model=DocumentListResponse)
def get_documents():
    """Get list of all documents in the RAG system"""
    try:
        rag = get_rag_system()
        documents = rag.list_documents()
        return DocumentListResponse(documents=documents, count=len(documents))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh_documents")
def refresh_documents():
    """Refresh the vector store by reloading all documents"""
    try:
        rag = get_rag_system()
        success = rag.refresh_vector_store()
        if success:
            return {"success": True, "message": "Vector store refreshed successfully"}
        else:
            return {"success": False, "message": "Failed to refresh vector store"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/force_process_documents")
def force_process_documents():
    """Force process all documents in the uploaded_document folder"""
    try:
        rag = get_rag_system()
        success = rag.force_process_all_documents()
        if success:
            return {"success": True, "message": "All documents processed successfully"}
        else:
            return {"success": False, "message": "Failed to process documents"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/processing_status")
def get_processing_status():
    """Get the current processing status of documents"""
    try:
        rag = get_rag_system()
        status = rag.get_processing_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict-disease/")
async def predict_disease(file: UploadFile = File(...)):
    try:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pipeline = PredictionPipeline(temp_file)
        result = pipeline.predict()

        os.remove(temp_file)
        return {"prediction": result[0]["image"], "probabilities": result[0]["probabilities"]}
    except Exception as e:
        return {"error": str(e)}
