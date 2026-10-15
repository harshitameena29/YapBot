cat > README.md <<'EOF'
# YapBot 🤖

Your AI Research Companion.

YapBot is a document question-answering application that uses Retrieval-Augmented Generation (RAG) to answer questions about uploaded PDF documents.

## Features
- Upload and process PDF documents
- Ask questions about document content
- Retrieve relevant passages using ChromaDB
- Generate answers using a Groq language model
- View source passages for answers
- Adjust chunking and retrieval settings

## Tech Stack
- Python
- Streamlit
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq LLM

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Add your Groq API key to a `.env` file:

```text
GROQ_API_KEY=your_api_key_here
```

Start the application:

```bash
streamlit run app.py
```
EOF