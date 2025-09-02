from document_processor import DocumentProcessor
import os

def initialize_knowledge_base():
    # Create documents folder if it doesn't exist
    os.makedirs("documents", exist_ok=True)
    
    # Create a sample document if the folder is empty
    if not os.listdir("documents"):
        sample_file = os.path.join("documents", "sample.txt")
        with open(sample_file, "w") as f:
            f.write("""Welcome to the Medical Chatbot Knowledge Base.

This is a sample document. Please add your medical documents, research papers, 
and other relevant materials to this folder.

Supported formats:
- PDF (.pdf)
- Text files (.txt)
- Word documents (.doc, .docx)
- Markdown files (.md)

The chatbot will use these documents to provide informed responses to medical queries.
""")
        print("Created sample document in documents folder")
    
    print("Initializing knowledge base...")
    processor = DocumentProcessor()
    
    # Try to clear existing index (if it fails, continue anyway)
    try:
        processor.clear_index()
    except Exception as e:
        print(f"Note: Could not clear existing index: {e}")
    
    # Process all documents
    processor.process_documents()
    
    print("Knowledge base initialized successfully!")

if __name__ == "__main__":
    initialize_knowledge_base()