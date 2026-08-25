# RAGBot 🤖

RAGBot is a free AI chatbot that combines Retrieval-Augmented Generation (RAG) with general-purpose question answering.

## Features

- General question answering
- Document-based question answering
- Semantic search
- Hugging Face embeddings
- FAISS vector store
- Hugging Face LLM
- Streamlit chat interface
- Chat history during the current session
- No API key required

## Technology Stack

- Python
- Streamlit
- LangChain
- Hugging Face Sentence Transformers
- Hugging Face Transformers
- FAISS
- PyTorch

## Architecture

User Question
↓
Embedding
↓
FAISS Similarity Search
↓
Relevant Document Context
↓
Hugging Face LLM
↓
Final Answer

If the retrieved document information is not relevant,
RAGBot can use the LLM's general knowledge.

## Project Structure

RAGBot/
│
├── app.py
├── document_loader.py
├── text_splitter.py
├── embeddings.py
├── vector_store.py
├── llm.py
├── rag_pipeline.py
├── sample.txt
├── requirements.txt
├── README.md
└── .gitignore

## Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/RAGBot.git

Move into the project:

cd RAGBot

Install dependencies:

pip install -r requirements.txt

## Run Locally

streamlit run app.py

## Deployment

The application can be deployed using Streamlit Community Cloud.

No API key is required for this version.

## RAG Process

1. Load the document.
2. Split the document into smaller chunks.
3. Convert chunks into embeddings.
4. Store embeddings in FAISS.
5. Convert the user's question into an embedding.
6. Retrieve relevant document chunks.
7. Provide relevant context to the LLM.
8. Generate the final response.

## Future Improvements

- PDF support
- Multiple document uploads
- Better open-source LLM
- Conversation memory
- Source citations
- Improved retrieval
- Web search for real-time information

## License

This project is intended for educational and portfolio purposes.
