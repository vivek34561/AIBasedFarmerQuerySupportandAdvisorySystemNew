import os
import json
from typing import List, Dict
from document_processor import DocumentProcessor
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentManager:
    def __init__(self, documents_folder="documents", log_file="processed_documents.json"):
        self.documents_folder = documents_folder
        self.log_file = log_file
        self.processor = DocumentProcessor()
        
    def get_processed_documents(self) -> Dict:
        """Get list of all processed documents"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return {}
    
    def add_single_document(self, file_path: str):
        """Add a single document to the vector store"""
        try:
            logger.info(f"Adding document: {file_path}")
            
            # Load document
            documents = self.processor.load_document(file_path)
            if not documents:
                logger.error(f"Could not load document: {file_path}")
                return False
            
            # Split document
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Add to vector store
            self.processor.vector_store.add_documents(splits)
            
            # Update log
            processed = self.get_processed_documents()
            file_stat = os.stat(file_path)
            processed[file_path] = {
                'processed_at': datetime.now().isoformat(),
                'modified': file_stat.st_mtime,
                'chunks': len(splits)
            }
            
            with open(self.log_file, 'w') as f:
                json.dump(processed, f, indent=2)
            
            logger.info(f"Successfully added {file_path} ({len(splits)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {e}")
            return False
    
    def remove_document(self, file_path: str):
        """Remove a document from the vector store (Note: Pinecone doesn't support selective deletion easily)"""
        logger.warning("Document removal not implemented - would require storing vector IDs")
        # In production, you'd store vector IDs when adding documents
        # Then delete those specific IDs here
    
    def list_documents(self) -> List[Dict]:
        """List all documents and their status"""
        processed = self.get_processed_documents()
        all_docs = []
        
        # Check all files in documents folder
        for root, dirs, files in os.walk(self.documents_folder):
            for file in files:
                file_path = os.path.join(root, file)
                file_stat = os.stat(file_path)
                
                doc_info = {
                    'path': file_path,
                    'size': file_stat.st_size,
                    'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                }
                
                if file_path in processed:
                    doc_info['status'] = 'processed'
                    doc_info['processed_at'] = processed[file_path]['processed_at']
                    doc_info['chunks'] = processed[file_path]['chunks']
                else:
                    doc_info['status'] = 'pending'
                
                all_docs.append(doc_info)
        
        return all_docs
    
    def sync_documents(self):
        """Sync all new and modified documents"""
        processed = self.get_processed_documents()
        new_count = 0
        
        for root, dirs, files in os.walk(self.documents_folder):
            for file in files:
                file_path = os.path.join(root, file)
                file_stat = os.stat(file_path)
                
                # Check if new or modified
                if (file_path not in processed or 
                    processed[file_path]['modified'] < file_stat.st_mtime):
                    
                    if self.add_single_document(file_path):
                        new_count += 1
        
        logger.info(f"Sync complete! Processed {new_count} new/modified documents")
        return new_count

# CLI interface
if __name__ == "__main__":
    import sys
    
    manager = DocumentManager()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python document_manager.py list        - List all documents")
        print("  python document_manager.py sync        - Process new/modified documents")
        print("  python document_manager.py add <file>  - Add specific document")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list":
        docs = manager.list_documents()
        print(f"\nTotal documents: {len(docs)}")
        for doc in docs:
            status = "✓" if doc['status'] == 'processed' else "○"
            print(f"{status} {doc['path']} ({doc.get('chunks', 0)} chunks)")
    
    elif command == "sync":
        count = manager.sync_documents()
        print(f"Processed {count} new/modified documents")
    
    elif command == "add" and len(sys.argv) > 2:
        file_path = sys.argv[2]
        if manager.add_single_document(file_path):
            print(f"Successfully added {file_path}")
        else:
            print(f"Failed to add {file_path}")
    
    else:
        print("Invalid command")