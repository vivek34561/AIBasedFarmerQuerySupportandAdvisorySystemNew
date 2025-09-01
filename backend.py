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

load_dotenv()

# -------------------- Chatbot Setup --------------------
llm = ChatOpenAI()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

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

@app.post("/chat", response_model=List[Message])
def chat_endpoint(request: ChatRequest):
    thread_id = request.thread_id or generate_thread_id()
    CONFIG = {'configurable': {'thread_id': thread_id}}
    human_messages = convert_to_human(request.messages)

    # Add instruction for language if not English
    if request.language and request.language.lower() != "english":
        human_messages.append(HumanMessage(content=f"Please respond in {request.language}."))

    try:
        response = chatbot.invoke({'messages': human_messages}, config=CONFIG)
        ai_messages = response['messages']
        return [{"role": "assistant", "content": msg.content} for msg in ai_messages]
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