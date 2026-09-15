"""
RAG-Based Document Assistant - simplified single-file version (Streamlit)

Pipeline:
PDF -> extract text -> clean -> sentence-aware chunking -> embed
-> store in ChromaDB -> retrieve top-5 -> prompt -> Llama 3.2 (Ollama)

Run with:
    pip install streamlit pdfplumber nltk sentence-transformers chromadb ollama
    ollama pull llama3.2
    streamlit run app.py
"""

import streamlit as st
import pdfplumber
import re
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import io
import os


# ---------- Optional OpenAI ----------

USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"

if USE_OPENAI:
    from openai import OpenAI
    openai_client = OpenAI()


# ---------- NLTK ----------

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ---------- Load model + DB once per session ----------

@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def get_client():
    return chromadb.PersistentClient(path="./chroma_db")


model = get_model()
client = get_client()


# ---------- Text processing ----------

def clean_text(text):
    """Collapse extra whitespace/newlines from raw PDF-extracted text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text, max_chunk_size=1000):
    """
    Split text into chunks that never cut a sentence in half,
    unless a single sentence itself exceeds max_chunk_size.
    """

    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:

        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence

        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            if len(sentence) > max_chunk_size:

                for i in range(0, len(sentence), max_chunk_size):
                    piece = sentence[i:i + max_chunk_size].strip()

                    if piece:
                        chunks.append(piece)

                current_chunk = ""

            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ---------- PDF processing ----------

def extract_pdf_text(contents):
    """
    Extract text using pdfplumber, which handles design-heavy PDFs
    (subsetted/custom fonts, ligatures — common in corporate reports
    built in InDesign) far more reliably than pypdf.
    """
    full_text = ""
    with pdfplumber.open(io.BytesIO(contents)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
    return full_text


def process_pdf(uploaded_file):

    # Read uploaded PDF
    contents = uploaded_file.getvalue()

    if not contents:
        raise ValueError("The uploaded file is empty.")

    # Extract text
    full_text = extract_pdf_text(contents)

    # Clean extracted text
    cleaned_text = clean_text(full_text)

    # Prevent empty embeddings
    if not cleaned_text:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "Please upload a text-based PDF. "
            "Scanned/image-only PDFs are not supported."
        )

    # Create chunks
    chunks = chunk_text(cleaned_text)

    # Safety check
    if not chunks:
        raise ValueError(
            "No text chunks could be created from this PDF."
        )

    # Generate embeddings
    embeddings = model.encode(
        chunks,
        show_progress_bar=False
    )

    # Safety check
    if embeddings is None or len(embeddings) == 0:
        raise ValueError(
            "No embeddings were generated for this document."
        )

    # ---------- Collection name ----------

    filename = uploaded_file.name.rsplit(".", 1)[0]

    collection_name = re.sub(
        r"[^a-zA-Z0-9._-]",
        "_",
        filename
    ).strip("._-")

    # ChromaDB collection names must be at least 3 characters
    if len(collection_name) < 3:
        collection_name = "doc_" + collection_name

    # Make collection name safe and reasonably sized
    collection_name = collection_name[:60]

    # Delete old collection with the same name if it exists
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    # Create collection
    collection = client.create_collection(
        name=collection_name
    )

    # Add documents and embeddings
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    return collection, collection_name, len(chunks)


# ---------- Question answering ----------

def ask_question(collection, question, top_k=5):

    # Create embedding for user question
    query_embedding = model.encode(
        question,
        show_progress_bar=False
    ).tolist()

    # Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count())
    )

    retrieved_chunks = results["documents"][0]

    # Build context
    context = "\n\n".join(retrieved_chunks)

    # Prompt
    prompt = f"""You are a helpful document assistant.

Use the context below to answer the user's question accurately.

Context:
{context}

Question:
{question}

Instructions:
- Start with a one-sentence direct answer.
- If there are multiple relevant points, list them as short bullet points.
- Keep the language clear and simple.
- Do not add information that is not supported by the context.
- If the context does not contain enough information, say so clearly instead of guessing.

Answer:"""

    # ---------- OpenAI ----------

    if USE_OPENAI:

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )

        return (
            response.choices[0].message.content,
            retrieved_chunks
        )

    # ---------- Ollama ----------

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return (
        response["message"]["content"],
        retrieved_chunks
    )


# ---------- Streamlit UI ----------

st.title("RAG-Based Document Assistant")

st.caption(
    "Upload one PDF document and ask questions about its content."
)


# ---------- File uploader ----------

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    accept_multiple_files=False
)


# ---------- Detect new document ----------

if uploaded_file:

    current_file_id = (
        uploaded_file.name,
        uploaded_file.size
    )

    previous_file_id = st.session_state.get(
        "file_id"
    )

    # Process only when a new/different file is uploaded
    if current_file_id != previous_file_id:

        # Clear previous document state
        st.session_state.pop("collection", None)
        st.session_state.pop("collection_name", None)
        st.session_state.pop("last_filename", None)

        with st.spinner(
            "Processing document..."
        ):

            try:

                collection, collection_name, n_chunks = process_pdf(
                    uploaded_file
                )

                # Store new document information
                st.session_state.collection = collection
                st.session_state.collection_name = collection_name
                st.session_state.last_filename = uploaded_file.name
                st.session_state.file_id = current_file_id

                st.success(
                    f"Indexed {uploaded_file.name} into {n_chunks} chunks."
                )

            except Exception as e:

                st.error(
                    f"Could not process this PDF: {e}"
                )

                # Make sure an invalid document
                # does not leave stale state
                st.session_state.pop("collection", None)
                st.session_state.pop("collection_name", None)
                st.session_state.pop("last_filename", None)
                st.session_state.pop("file_id", None)


# ---------- Ask questions ----------

if "collection" in st.session_state:

    question = st.text_input(
        "Ask a question about the document"
    )

    if question:

        with st.spinner(
            "Retrieving relevant information and generating answer..."
        ):

            try:

                answer, sources = ask_question(
                    st.session_state.collection,
                    question,
                    top_k=5
                )

                st.markdown("### Answer")

                st.write(answer)

                with st.expander("Source chunks used"):

                    for i, chunk in enumerate(sources, 1):

                        st.markdown(
                            f"**Chunk {i}:** {chunk}"
                        )

            except Exception as e:

                st.error(
                    f"Could not generate an answer: {e}"
                )
