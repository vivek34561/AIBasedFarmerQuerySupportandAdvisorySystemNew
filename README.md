# RAG-Enhanced Chatbot

A powerful chatbot application that uses Retrieval Augmented Generation (RAG) to provide contextual answers based on your uploaded documents.

## Features

- **Multi-format Document Support**: Upload PDF, TXT, and DOCX files
- **Vector Storage**: Uses ChromaDB for efficient document embedding storage  
- **Intelligent Retrieval**: Finds relevant document sections for each query
- **Contextual Responses**: Generates answers based on retrieved content
- **Multi-language Support**: Supports responses in multiple languages
- **Voice Input**: Speech-to-text functionality for hands-free interaction
- **Thread Management**: Organize conversations in separate threads
- **Disease Prediction**: Bonus feature for image-based disease prediction

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   ```bash
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1. **Start the backend server**:
   ```bash
   uvicorn backend:app --reload
   ```

2. **In a new terminal, start the frontend**:
   ```bash
   streamlit run frontend.py
   ```

3. **Access the application**:
   - Open your browser and go to `http://localhost:8501`
   - The backend API will be running on `http://localhost:8000`

## How to Use the RAG System

1. **Upload Documents**:
   - Use the sidebar "Document Management" section
   - Upload PDF, TXT, or DOCX files
   - Documents will be processed and stored in the vector database

2. **Ask Questions**:
   - Type questions related to your uploaded documents
   - The chatbot will search through your documents and provide relevant answers

3. **Manage Knowledge Base**:
   - View uploaded documents in the sidebar
   - Use the refresh button to reload documents if needed

## Project Structure

```
├── backend.py              # FastAPI backend with RAG integration
├── frontend.py             # Streamlit frontend interface
├── rag_system.py           # Core RAG functionality
├── chatbot_backend.py      # Original chatbot backend (standalone)
├── models/                 # Disease prediction models
├── uploaded_document/      # Document storage directory
├── chroma_db/             # Vector database storage
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## API Endpoints

- `POST /chat` - Send messages to the chatbot
- `POST /upload_document` - Upload documents to the knowledge base
- `GET /documents` - Get list of uploaded documents
- `POST /refresh_documents` - Refresh the vector store
- `GET /threads` - Get all conversation threads
- `POST /new_thread` - Create a new conversation thread

## Troubleshooting

1. **Import Errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`

2. **OpenAI API Issues**: Ensure your API key is correctly set in the `.env` file

3. **Document Processing Errors**: Check that documents are in supported formats (PDF, TXT, DOCX)

4. **Vector Store Issues**: Try using the "Refresh Knowledge Base" button in the sidebar

## Contributing

Feel free to submit issues and enhancement requests!

## License

See LICENSE file for details.