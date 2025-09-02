import os
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import logging
import time

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, documents_folder: str = "documents"):
        self.documents_folder = documents_folder
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
        
        # Create index if it doesn't exist
        existing_indexes = self.pc.list_indexes().names()
        if self.index_name not in existing_indexes:
            logger.info(f"Creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embeddings dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=os.getenv("PINECONE_REGION", "us-east-1")
                )
            )
            # Wait for index to be ready
            time.sleep(10)
        
        self.index = self.pc.Index(self.index_name)
        
        # Wait for index to be ready
        logger.info("Waiting for index to be ready...")
        while not self.index.describe_index_stats()['dimension']:
            time.sleep(1)
        
        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="text",
            namespace=""  # Use default namespace
        )
        
    def load_document(self, file_path: str):
        """Load a single document based on its extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif ext in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif ext == '.md':
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                logger.warning(f"Unsupported file type: {ext}")
                return []
            
            return loader.load()
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            return []
    
    def process_documents(self):
        """Process all documents in the documents folder"""
        if not os.path.exists(self.documents_folder):
            logger.warning(f"Documents folder '{self.documents_folder}' does not exist")
            return
            
        all_documents = []
        
        # Walk through all files in the documents folder
        for root, dirs, files in os.walk(self.documents_folder):
            for file in files:
                file_path = os.path.join(root, file)
                logger.info(f"Processing: {file_path}")
                documents = self.load_document(file_path)
                all_documents.extend(documents)
        
        if not all_documents:
            logger.warning("No documents found to process")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        splits = text_splitter.split_documents(all_documents)
        logger.info(f"Created {len(splits)} document chunks")
        
        # Add documents to Pinecone in batches
        batch_size = 100
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            self.vector_store.add_documents(batch)
            logger.info(f"Added batch {i//batch_size + 1}/{(len(splits) + batch_size - 1)//batch_size}")
        
        logger.info("Documents successfully added to Pinecone")
        
    def clear_index(self):
        """Clear all vectors from the index"""
        try:
            # Get index stats
            stats = self.index.describe_index_stats()
            
            # If there are vectors, delete them
            if stats['total_vector_count'] > 0:
                # Delete all vectors in the default namespace
                self.index.delete(delete_all=True, namespace="")
                logger.info("Cleared all vectors from Pinecone index")
            else:
                logger.info("Index is already empty")
        except Exception as e:
            logger.warning(f"Could not clear index: {str(e)}")
            # If clearing fails, we can still continue

if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process_documents()