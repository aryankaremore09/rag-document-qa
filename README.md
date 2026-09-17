# RAG Document Q&A Bot

A Q&A bot that answers questions from your own PDFs using RAG (retrieval-augmented generation).

## Stack
- OpenAI API (embeddings + chat completion)
- ChromaDB (vector database)
- Streamlit (web interface)
- Recursive chunking via LangChain's text splitter

## How it works
1. `ingest.py` — extracts text from PDFs, chunks it, embeds it, stores it in ChromaDB
2. `ask.py` — terminal-based Q&A over the documents
3. `app.py` — Streamlit web interface for the same bot

## Setup
1. `pip install -r requirements.txt`
2. Add your OpenAI API key to a `.env` file: `OPENAI_API_KEY=your_key_here`
3. Add PDFs to the `docs/` folder
4. Run `python ingest.py` to build the knowledge base
5. Run `streamlit run app.py` to launch the chat interface