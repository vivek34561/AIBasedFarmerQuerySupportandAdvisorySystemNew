import os
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
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
        
        # Initialize Hugging Face embeddings
        model_name = os.getenv("HUGGINGFACE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Loading embedding model: {model_name}")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Get embedding dimension
        test_embedding = self.embeddings.embed_query("test")
        self.embedding_dimension = len(test_embedding)
        logger.info(f"Embedding dimension: {self.embedding_dimension}")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
        logger.info(f"Using Pinecone index name: {self.index_name}")
        
        # List all existing indexes
        existing_indexes = self.pc.list_indexes()
        logger.info("Existing Pinecone indexes:")
        for idx in existing_indexes:
            logger.info(f"  - {idx['name']} (dimension: {idx['dimension']})")
        
        # Check if our index exists
        index_names = [idx['name'] for idx in existing_indexes]
        if self.index_name in index_names:
            # Get the existing index info
            existing_index = next(idx for idx in existing_indexes if idx['name'] == self.index_name)
            existing_dim = existing_index['dimension']
            
            if existing_dim != self.embedding_dimension:
                logger.error(f"ERROR: Index '{self.index_name}' exists with dimension {existing_dim}, but embeddings have dimension {self.embedding_dimension}")
                logger.error("Please either:")
                logger.error("1. Delete the existing index and run again")
                logger.error("2. Use a different index name in your .env file")
                logger.error("3. Use a different embedding model that matches the dimension")
                raise ValueError(f"Dimension mismatch: index={existing_dim}, embeddings={self.embedding_dimension}")
            else:
                logger.info(f"Using existing index '{self.index_name}' with matching dimension {existing_dim}")
        else:
            # Create new index
            logger.info(f"Creating new index: {self.index_name} with dimension {self.embedding_dimension}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=os.getenv("PINECONE_REGION", "us-east-1")
                )
            )
            logger.info("Waiting for index to be ready...")
            time.sleep(10)
        
        self.index = self.pc.Index(self.index_name)
        
        # Verify index is ready
        logger.info("Verifying index status...")
        stats = self.index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
        
        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="text",
            namespace=""
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
            os.makedirs(self.documents_folder, exist_ok=True)
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
        
        # Add documents to Pinecone in smaller batches
        batch_size = 20  # Smaller batch size
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            try:
                self.vector_store.add_documents(batch)
                logger.info(f"Added batch {i//batch_size + 1}/{(len(splits) + batch_size - 1)//batch_size}")
                time.sleep(0.5)  # Small delay between batches
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                # Try adding documents one by one
                for j, doc in enumerate(batch):
                    try:
                        self.vector_store.add_documents([doc])
                        logger.info(f"Added document {i+j+1}/{len(splits)} individually")
                    except Exception as e2:
                        logger.error(f"Failed to add document {i+j+1}: {e2}")
        
        logger.info("Documents successfully added to Pinecone")
        
    def clear_index(self):
        """Clear all vectors from the index"""
        try:
            stats = self.index.describe_index_stats()
            if stats.get('total_vector_count', 0) > 0:
                self.index.delete(delete_all=True, namespace="")
                logger.info("Cleared all vectors from Pinecone index")
            else:
                logger.info("Index is already empty")
        except Exception as e:
            logger.warning(f"Could not clear index: {str(e)}")

if __name__ == "__main__":
    processor = DocumentProcessor()
    processor.process_documents()