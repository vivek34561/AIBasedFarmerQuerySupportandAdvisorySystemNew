import os
import shutil
from typing import List, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self, documents_folder: str = "uploaded_document", chroma_db_path: str = "chroma_db"):
        self.documents_folder = documents_folder
        self.chroma_db_path = chroma_db_path
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0.3)
        
        # Create folders if they don't exist
        os.makedirs(self.documents_folder, exist_ok=True)
        os.makedirs(self.chroma_db_path, exist_ok=True)
        
        # Initialize text splitter with smaller chunks for large documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Reduced from 1000 to handle large documents
            chunk_overlap=100,  # Reduced from 200
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize or load vector store
        self.vector_store = self._initialize_vector_store()
        
        # Create RAG chain
        self.rag_chain = self._create_rag_chain()
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize or load existing ChromaDB vector store"""
        try:
            # Try to load existing vector store
            vector_store = Chroma(
                persist_directory=self.chroma_db_path,
                embedding_function=self.embeddings
            )
            
            # Check if vector store has documents and only process new ones
            if len(vector_store.get()['ids']) == 0:
                print("🆕 Vector store is empty, processing all documents...")
                self._load_and_process_documents(vector_store)
            else:
                print("📚 Vector store has existing documents, checking for new ones...")
                self._check_and_add_new_documents(vector_store)
            
            return vector_store
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            # Create new vector store
            vector_store = Chroma(
                persist_directory=self.chroma_db_path,
                embedding_function=self.embeddings
            )
            self._load_and_process_documents(vector_store)
            return vector_store
    
    def _load_and_process_documents(self, vector_store: Chroma) -> None:
        """Load documents from the documents folder and add to vector store"""
        documents = self._load_documents()
        if documents:
            chunks = self.text_splitter.split_documents(documents)
            if chunks:
                # Process chunks in smaller batches to avoid token limits
                self._add_chunks_in_batches(vector_store, chunks)
    
    def _add_chunks_in_batches(self, vector_store: Chroma, chunks: List[Document], batch_size: int = 50) -> None:
        """Add chunks to vector store in smaller batches to avoid token limits"""
        total_chunks = len(chunks)
        print(f"Processing {total_chunks} chunks in batches of {batch_size}...")
        
        successful_batches = 0
        total_added = 0
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            try:
                print(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
                vector_store.add_documents(batch)
                successful_batches += 1
                total_added += len(batch)
                print(f"✅ Batch {batch_num} processed successfully")
                
            except Exception as e:
                if "max_tokens_per_request" in str(e):
                    print(f"⚠️  Batch {batch_num} too large, splitting smaller...")
                    # If batch is still too large, process individually
                    for j, doc in enumerate(batch):
                        try:
                            vector_store.add_documents([doc])
                            total_added += 1
                            print(f"✅ Added individual document {j+1}/{len(batch)} from batch {batch_num}")
                        except Exception as doc_error:
                            print(f"❌ Failed to add document {j+1}: {str(doc_error)[:100]}...")
                else:
                    print(f"❌ Error processing batch {batch_num}: {str(e)[:100]}...")
        
        print(f"🎉 Processing complete! Added {total_added}/{total_chunks} chunks successfully")
    
    def _check_and_add_new_documents(self, vector_store: Chroma) -> None:
        """Check for new documents and add only new ones to avoid duplicates"""
        try:
            # Get list of documents already in vector store
            existing_docs = set()
            try:
                # Try to get metadata from existing documents
                collection = vector_store._collection
                if collection:
                    # Get document metadata to track filenames
                    results = collection.get(include=['metadatas'])
                    if results and results['metadatas']:
                        for metadata in results['metadatas']:
                            if metadata and 'source' in metadata:
                                filename = os.path.basename(metadata['source'])
                                existing_docs.add(filename)
                                print(f"📋 Found existing document: {filename}")
            except Exception as e:
                print(f"Warning: Could not retrieve existing document metadata: {e}")
            
            # Get current documents in folder
            current_docs = set(self.list_documents())
            print(f"📁 Current documents in folder: {list(current_docs)}")
            
            # Find new documents
            new_docs = current_docs - existing_docs
            
            if new_docs:
                print(f"🆕 Found {len(new_docs)} new documents to process: {list(new_docs)}")
                
                # Process only new documents
                for filename in new_docs:
                    file_path = os.path.join(self.documents_folder, filename)
                    if os.path.exists(file_path):
                        print(f"⚡ Processing new document: {filename}")
                        self._load_single_document(file_path)
                        print(f"✅ Processed new document: {filename}")
            else:
                print("✅ All documents are already processed. No new documents found.")
                
        except Exception as e:
            print(f"Error checking for new documents: {e}")
    
    def _load_documents(self) -> List[Document]:
        """Load documents from the documents folder"""
        documents = []
        
        if not os.path.exists(self.documents_folder):
            print(f"Documents folder {self.documents_folder} does not exist")
            return documents
        
        # Load different file types
        loaders = {
            "*.pdf": PyPDFLoader,
            "*.txt": TextLoader,
            "*.docx": Docx2txtLoader,
        }
        
        for pattern, loader_class in loaders.items():
            try:
                # Use DirectoryLoader for each file type
                loader = DirectoryLoader(
                    self.documents_folder,
                    glob=pattern,
                    loader_cls=loader_class,
                    show_progress=True
                )
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded {len(docs)} documents with pattern {pattern}")
            except Exception as e:
                print(f"Error loading documents with pattern {pattern}: {e}")
        
        return documents
    
    def _create_rag_chain(self):
        """Create the RAG chain for question answering"""
        # Create retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Create prompt template
        template = """You are a helpful assistant that answers questions based on the provided context.
Use the following pieces of context to answer the user's question. If you don't know the answer based on the context, just say that you don't have enough information to answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
        )
        
        return rag_chain
    
    def add_document(self, file_path: str, filename: str) -> bool:
        """Add a new document to the system"""
        try:
            # Copy file to documents folder
            destination = os.path.join(self.documents_folder, filename)
            shutil.copy2(file_path, destination)
            
            # Load and process the new document
            self._load_single_document(destination)
            return True
        except Exception as e:
            print(f"Error adding document {filename}: {e}")
            return False
    
    def _load_single_document(self, file_path: str) -> None:
        """Load and process a single document"""
        try:
            # Determine loader based on file extension
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext == '.txt':
                loader = TextLoader(file_path)
            elif file_ext == '.docx':
                loader = Docx2txtLoader(file_path)
            else:
                print(f"Unsupported file type: {file_ext}")
                return
            
            # Load document
            documents = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add to vector store using batching
            if chunks:
                print(f"📄 Processing {os.path.basename(file_path)} ({len(chunks)} chunks)")
                self._add_chunks_in_batches(self.vector_store, chunks)
            
        except Exception as e:
            print(f"Error processing document {file_path}: {e}")
    
    def query(self, question: str) -> str:
        """Query the RAG system with a question"""
        try:
            response = self.rag_chain.invoke(question)
            return response.content
        except Exception as e:
            print(f"Error querying RAG system: {e}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vector store"""
        try:
            return len(self.vector_store.get()['ids'])
        except:
            return 0
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get detailed processing status"""
        try:
            # Get documents in vector store
            vector_docs = set()
            try:
                collection = self.vector_store._collection
                if collection:
                    results = collection.get(include=['metadatas'])
                    if results and results['metadatas']:
                        for metadata in results['metadatas']:
                            if metadata and 'source' in metadata:
                                filename = os.path.basename(metadata['source'])
                                vector_docs.add(filename)
            except Exception as e:
                print(f"Warning: Could not retrieve vector store metadata: {e}")
            
            # Get documents in folder
            folder_docs = set(self.list_documents())
            
            # Calculate status
            processed = len(vector_docs)
            total = len(folder_docs)
            new_docs = folder_docs - vector_docs
            
            return {
                "processed_documents": list(vector_docs),
                "total_documents": list(folder_docs),
                "new_documents": list(new_docs),
                "processed_count": processed,
                "total_count": total,
                "new_count": len(new_docs),
                "is_up_to_date": len(new_docs) == 0
            }
        except Exception as e:
            return {
                "error": str(e),
                "processed_count": 0,
                "total_count": 0,
                "new_count": 0,
                "is_up_to_date": False
            }
    
    def list_documents(self) -> List[str]:
        """List all documents in the documents folder"""
        if not os.path.exists(self.documents_folder):
            return []
        
        documents = []
        for file in os.listdir(self.documents_folder):
            if file.endswith(('.pdf', '.txt', '.docx')):
                documents.append(file)
        return documents
    
    def refresh_vector_store(self) -> bool:
        """Refresh the vector store by reloading all documents"""
        try:
            # Clear existing vector store
            self.vector_store.delete_collection()
            
            # Reinitialize vector store
            self.vector_store = Chroma(
                persist_directory=self.chroma_db_path,
                embedding_function=self.embeddings
            )
            
            # Reload documents
            self._load_and_process_documents(self.vector_store)
            
            # Recreate RAG chain
            self.rag_chain = self._create_rag_chain()
            
            return True
        except Exception as e:
            print(f"Error refreshing vector store: {e}")
            return False
    
    def force_process_all_documents(self) -> bool:
        """Force process all documents in the folder"""
        try:
            print("🔄 Force processing all documents...")
            
            # Get all documents in folder
            documents = self.list_documents()
            if not documents:
                print("📭 No documents found in folder")
                return False
            
            print(f"📄 Found {len(documents)} documents: {documents}")
            
            # Process each document
            total_chunks = 0
            for filename in documents:
                file_path = os.path.join(self.documents_folder, filename)
                if os.path.exists(file_path):
                    print(f"📖 Processing: {filename}")
                    self._load_single_document(file_path)
                    total_chunks += 1
            
            print(f"✅ Successfully processed {total_chunks} documents")
            
            # Recreate RAG chain with updated vector store
            self.rag_chain = self._create_rag_chain()
            
            return True
            
        except Exception as e:
            print(f"❌ Error force processing documents: {e}")
            return False

# Global RAG system instance
rag_system = None

def get_rag_system() -> RAGSystem:
    """Get or create global RAG system instance"""
    global rag_system
    if rag_system is None:
        rag_system = RAGSystem()
    return rag_system
