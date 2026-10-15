import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DB_ROOT = Path("chroma_db")
LIBRARY_FILE = DB_ROOT / "library.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-120b"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)

st.set_page_config(page_title="yapBot", page_icon="📚", layout="wide")


# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def get_llm():
    return ChatGroq(model=LLM_MODEL)


@st.cache_resource
def get_vectorstore(book_id: str):
    return Chroma(
        persist_directory=str(DB_ROOT / book_id),
        embedding_function=get_embeddings(),
    )


# ----------------------------------------------------------------------------
# Library helpers (keeps track of every book that has been indexed)
# ----------------------------------------------------------------------------
def load_library() -> dict:
    if LIBRARY_FILE.exists():
        try:
            return json.loads(LIBRARY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def save_library(library: dict) -> None:
    DB_ROOT.mkdir(exist_ok=True)
    LIBRARY_FILE.write_text(json.dumps(library, indent=2), encoding="utf-8")


def index_pdf(file_bytes: bytes, filename: str, chunk_size: int, chunk_overlap: int) -> str:
    """Load, split, embed and store a PDF. Returns the book id."""
    book_id = hashlib.md5(file_bytes + f"{chunk_size}-{chunk_overlap}".encode()).hexdigest()[:12]
    library = load_library()

    if book_id in library and (DB_ROOT / book_id).exists():
        return book_id  # already indexed with the same settings

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        documents = PyPDFLoader(tmp_path).load()
    finally:
        os.unlink(tmp_path)

    if not documents:
        raise ValueError("No text could be extracted from this PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata["source"] = filename

    DB_ROOT.mkdir(exist_ok=True)
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(DB_ROOT / book_id),
    )

    library[book_id] = {
        "name": filename,
        "pages": len(documents),
        "chunks": len(chunks),
    }
    save_library(library)
    return book_id


def delete_book(book_id: str) -> None:
    get_vectorstore.clear()  # release file handles before deleting
    shutil.rmtree(DB_ROOT / book_id, ignore_errors=True)
    library = load_library()
    library.pop(book_id, None)
    save_library(library)
    st.session_state.chats.pop(book_id, None)


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "chats" not in st.session_state:
    st.session_state.chats = {}  # book_id -> list of messages
if "active_book" not in st.session_state:
    st.session_state.active_book = None

# ----------------------------------------------------------------------------
# Sidebar: upload + library + settings
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("📚 Your Library")

    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found. Add it to your `.env` file.")

    st.subheader("Upload a book")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    with st.expander("Indexing settings"):
        chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, step=50)

    if st.button("Process book", type="primary", disabled=uploaded is None, use_container_width=True):
        try:
            with st.spinner("Reading, chunking and embedding the book... this can take a while."):
                book_id = index_pdf(uploaded.getvalue(), uploaded.name, chunk_size, chunk_overlap)
            st.session_state.active_book = book_id
            st.success("Book is ready!")
            st.rerun()
        except Exception as e:
            st.error(f"Could not process the PDF: {e}")

    st.divider()

    library = load_library()
    if library:
        st.subheader("Select a book")
        ids = list(library.keys())
        if st.session_state.active_book not in ids:
            st.session_state.active_book = ids[0]

        st.session_state.active_book = st.selectbox(
            "Ask questions about",
            options=ids,
            index=ids.index(st.session_state.active_book),
            format_func=lambda i: library[i]["name"],
        )
        info = library[st.session_state.active_book]
        st.caption(f"{info['pages']} pages · {info['chunks']} chunks")

        st.subheader("Retrieval settings")
        k = st.slider("Chunks to retrieve (k)", 1, 10, 4)
        fetch_k = st.slider("Candidates to consider (fetch_k)", k, 30, max(10, k))
        lambda_mult = st.slider(
            "Diversity (MMR λ)", 0.0, 1.0, 0.5, step=0.1,
            help="0 = maximum diversity, 1 = maximum relevance",
        )
        show_sources = st.toggle("Show source passages", value=True)

        col1, col2 = st.columns(2)
        if col1.button("Clear chat", use_container_width=True):
            st.session_state.chats[st.session_state.active_book] = []
            st.rerun()
        if col2.button("Delete book", use_container_width=True):
            delete_book(st.session_state.active_book)
            st.session_state.active_book = None
            st.rerun()
    else:
        k, fetch_k, lambda_mult, show_sources = 4, 10, 0.5, True

# ----------------------------------------------------------------------------
# Main area: chat
# ----------------------------------------------------------------------------
st.title("🤖 YapBot")
st.caption("Your AI Research Companion")

library = load_library()
book_id = st.session_state.active_book

if not library or book_id is None:
    st.info("👈 Upload a PDF in the sidebar and click **Process book** to get started.")
    st.stop()

st.caption(f"Currently reading: **{library[book_id]['name']}**")

messages = st.session_state.chats.setdefault(book_id, [])


def render_sources(sources):
    with st.expander(f"📄 Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            st.markdown(f"**Passage {i} — page {s['page']}**")
            st.text(s["text"])


for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            render_sources(msg["sources"])

query = st.chat_input("Ask YapBot anything about your document...")

if query:
    messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the book and thinking..."):
                retriever = get_vectorstore(book_id).as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
                )
                docs = retriever.invoke(query)
                context = "\n\n".join(doc.page_content for doc in docs)
                final_prompt = PROMPT.invoke({"context": context, "question": query})
                answer = get_llm().invoke(final_prompt).content

            sources = [
                {
                    "page": int(doc.metadata.get("page", 0)) + 1,
                    "text": doc.page_content,
                }
                for doc in docs
            ]
            st.markdown(answer)
            if show_sources:
                render_sources(sources)
            messages.append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as e:
            err = f"Something went wrong: {e}"
            st.error(err)
            messages.append({"role": "assistant", "content": err})