from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, TypedDict, Annotated
import uuid, sqlite3, shutil, os
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
from pinecone import Pinecone
from models.prediction import PredictionPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# -------------------- RAG Setup --------------------
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
    
    # Initialize Hugging Face embeddings
    # You can choose different models based on your needs
    # Some popular options:
    # - "sentence-transformers/all-MiniLM-L6-v2" (384 dimensions, fast)
    # - "sentence-transformers/all-mpnet-base-v2" (768 dimensions, balanced)
    # - "BAAI/bge-large-en-v1.5" (1024 dimensions, high quality)
    # - "sentence-transformers/all-MiniLM-L12-v2" (384 dimensions)
    
    model_name = os.getenv("HUGGINGFACE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},  # Change to 'cuda' if you have GPU
        encode_kwargs={'normalize_embeddings': True}  # For better similarity search
    )
    
    # Get embedding dimension
    embedding_dimension = len(embeddings.embed_query("test"))
    logger.info(f"Using embedding model: {model_name} with dimension: {embedding_dimension}")
    
    # Check if index exists
    if index_name in pc.list_indexes().names():
        index = pc.Index(index_name)
        # Verify dimension matches
        index_stats = index.describe_index_stats()
        if index_stats.get('dimension') != embedding_dimension:
            logger.warning(f"Index dimension ({index_stats.get('dimension')}) doesn't match embedding dimension ({embedding_dimension})")
            logger.warning("You may need to recreate the index with the correct dimension")
        
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            text_key="text",
            namespace=""  # Use default namespace
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    else:
        logger.warning(f"Pinecone index '{index_name}' not found. RAG features will be disabled.")
        retriever = None
except Exception as e:
    logger.error(f"Could not initialize Pinecone: {e}. RAG features will be disabled.")
    retriever = None

# -------------------- Chatbot Setup --------------------
from langchain_groq import ChatGroq
# ... other imports ...

# -------------------- LLM Setup --------------------
# Option 1: Groq (Recommended)

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    temperature=0.7,
    # You can optionally include other kwargs like timeout or max_retries.
)



# RAG prompt template
system_prompt = """You are a helpful medical assistant. Use the following context to answer questions.
If you don't know the answer based on the context, say so. Be accurate and helpful.

Context: {context}
up
Remember to:
1. Be empathetic and professional
2. Provide accurate information based on the context
3. Suggest consulting healthcare professionals for serious concerns
4. Maintain conversation history context
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: Optional[str]

def retrieve_context(query: str) -> str:
    """Retrieve relevant context from Pinecone"""
    if retriever is None:
        return ""
    
    try:
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return ""

def chat_node(state: ChatState):
    messages = state['messages']
    
    # Get the last human message
    last_human_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content
            break
    
    if last_human_msg:
        # Retrieve context
        context = retrieve_context(last_human_msg)
        
        # Create a prompt with context
        system_msg = SystemMessage(content=system_prompt.format(context=context))
        
        # Include conversation history (last 10 messages for context window)
        chat_history = messages[-10:] if len(messages) > 10 else messages
        
        # Prepare messages for the LLM
        llm_messages = [system_msg] + chat_history
        
        # Get response
        response = llm.invoke(llm_messages)
        
        return {"messages": [response], "context": context}
    
    return {"messages": [AIMessage(content="I didn't receive a message to respond to.")]}

# -------------------- SQLite Setup --------------------
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

# -------------------- FastAPI Setup --------------------
app = FastAPI(title="RAG LangGraph Chatbot API")
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

class DocumentListResponse(BaseModel):
    documents: List[str]

class ProcessingStatusResponse(BaseModel):
    total_documents: int
    processed_chunks: int

# -------------------- Utility Functions --------------------
def generate_thread_id():
    return str(uuid.uuid4())

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)

def convert_messages(messages: List[Message]):
    """Convert messages to appropriate LangChain message types"""
    converted = []
    for m in messages:
        if m.role == 'user':
            converted.append(HumanMessage(content=m.content))
        elif m.role == 'assistant':
            converted.append(AIMessage(content=m.content))
        elif m.role == 'system':
            converted.append(SystemMessage(content=m.content))
    return converted

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
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        if state and state.values:
            messages = state.values.get('messages', [])
            formatted = []
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    role = 'user'
                elif isinstance(msg, AIMessage):
                    role = 'assistant'
                elif isinstance(msg, SystemMessage):
                    role = 'system'
                else:
                    continue
                formatted.append({'role': role, 'content': msg.content})
            return formatted
        return []
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found: {str(e)}")

@app.post("/chat", response_model=List[Message])
def chat_endpoint(request: ChatRequest):
    thread_id = request.thread_id or generate_thread_id()
    CONFIG = {'configurable': {'thread_id': thread_id}}
    
    # Convert all messages to appropriate types
    all_messages = convert_messages(request.messages)
    
    # Add language instruction if needed
    if request.language and request.language.lower() != "english":
        all_messages.append(HumanMessage(content=f"Please respond in {request.language}."))
    
    try:
        # Invoke the chatbot with full message history
        response = chatbot.invoke({'messages': all_messages}, config=CONFIG)
        
        # Extract only new AI messages
        ai_messages = []
        for msg in response['messages']:
            if isinstance(msg, AIMessage):
                ai_messages.append({"role": "assistant", "content": msg.content})
        
        return ai_messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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


@app.post("/add_document")
async def add_document(file: UploadFile = File(...)):
    """Add a single document without reprocessing everything"""
    try:
        # Save file
        file_path = os.path.join("documents", file.filename)
        os.makedirs("documents", exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process just this document
        from documents_manager import DocumentManager
        manager = DocumentManager()
        success = manager.add_single_document(file_path)
        
        if success:
            return {"message": f"Document {file.filename} added successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to process document")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document_status")
def get_document_status():
    """Get status of all documents"""
    try:
        from documents_manager import DocumentManager
        manager = DocumentManager()
        docs = manager.list_documents()
        
        processed = [d for d in docs if d['status'] == 'processed']
        pending = [d for d in docs if d['status'] == 'pending']
        
        return {
            "total": len(docs),
            "processed": len(processed),
            "pending": len(pending),
            "documents": docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sync_documents")
def sync_documents():
    """Process only new/modified documents"""
    try:
        from documents_manager import DocumentManager
        manager = DocumentManager()
        count = manager.sync_documents()
        return {"message": f"Processed {count} new/modified documents"}
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
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pipeline = PredictionPipeline(temp_file)
        result = pipeline.predict()

        os.remove(temp_file)
        return {"prediction": result[0]["image"], "probabilities": result[0]["probabilities"]}
    except Exception as e:
        return {"error": str(e)}

# -------------------- Document Management Endpoints --------------------
@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to be processed and added to the knowledge base"""
    try:
        # Save the uploaded file
        file_path = os.path.join("documents", file.filename)
        os.makedirs("documents", exist_ok=True)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document
        from document_processor import DocumentProcessor
        processor = DocumentProcessor()
        processor.process_documents()
        
        return {"message": f"Document {file.filename} uploaded and processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reindex_documents")
def reindex_documents():
    """Reprocess all documents in the documents folder"""
    try:
        from document_processor import DocumentProcessor
        processor = DocumentProcessor()
        processor.clear_index()
        processor.process_documents()
        return {"message": "All documents reindexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))