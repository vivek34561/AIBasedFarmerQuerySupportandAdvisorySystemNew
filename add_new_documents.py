import os
import json
from document_processor import DocumentProcessor
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_FILES_LOG = "processed_documents.json"

def load_processed_files():
    """Load the list of already processed files"""
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, 'r') as f:
            return json.load(f)
    return {}

def save_processed_files(processed_files):
    """Save the list of processed files"""
    with open(PROCESSED_FILES_LOG, 'w') as f:
        json.dump(processed_files, f, indent=2)

def get_new_documents(documents_folder="documents"):
    """Get list of documents that haven't been processed yet"""
    processed_files = load_processed_files()
    new_documents = []
    
    for root, dirs, files in os.walk(documents_folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_stat = os.stat(file_path)
            file_modified_time = file_stat.st_mtime
            
            # Check if file is new or modified
            if file_path not in processed_files or processed_files[file_path]['modified'] < file_modified_time:
                new_documents.append(file_path)
    
    return new_documents

def process_new_documents():
    """Process only new or modified documents"""
    new_docs = get_new_documents()
    
    if not new_docs:
        logger.info("No new documents to process!")
        return
    
    logger.info(f"Found {len(new_docs)} new/modified documents to process:")
    for doc in new_docs:
        logger.info(f"  - {doc}")
    
    # Initialize processor
    processor = DocumentProcessor()
    processed_files = load_processed_files()
    
    # Process each new document
    for doc_path in new_docs:
        try:
            logger.info(f"Processing: {doc_path}")
            
            # Load and process single document
            documents = processor.load_document(doc_path)
            if documents:
                # Split the document
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                splits = text_splitter.split_documents(documents)
                
                # Add to vector store
                processor.vector_store.add_documents(splits)
                
                # Mark as processed
                file_stat = os.stat(doc_path)
                processed_files[doc_path] = {
                    'processed_at': datetime.now().isoformat(),
                    'modified': file_stat.st_mtime,
                    'chunks': len(splits)
                }
                
                logger.info(f"Successfully processed {doc_path} ({len(splits)} chunks)")
            
        except Exception as e:
            logger.error(f"Error processing {doc_path}: {e}")
    
    # Save updated processed files list
    save_processed_files(processed_files)
    logger.info("Document processing complete!")

if __name__ == "__main__":
    process_new_documents()