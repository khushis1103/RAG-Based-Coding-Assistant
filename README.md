# RAG-Based Document Assistant

RAG • Streamlit • ChromaDB • Sentence Transformers • Ollama

A Retrieval-Augmented Generation (RAG) application that lets you upload any PDF document and ask natural-language questions about it, with answers grounded strictly in the document's actual content rather than the model's own trained knowledge.

Built to explore how RAG systems are structured in practice — from raw PDF text to a working question-answering app. Demoed and tested on PwC's Global Annual Review 2025.

## How it works

```
PDF Upload → Text Extraction → Cleaning → Sentence-aware Chunking
→ Embeddings (sentence-transformers) → Stored in ChromaDB
→ [User asks a question] → Semantic Retrieval (top-5 relevant chunks)
→ Prompt construction → Local LLM (Llama 3.2 via Ollama) → Grounded Answer
```

The pipeline works on **any PDF you upload** — there is no document-specific logic anywhere in it (no keyword lists, no domain-specific parsing). It's a general-purpose document Q&A system; PwC's Global Annual Review 2025 was used as the test case during development.

## Sample Document

The repository includes PwC's Global Annual Review 2025 PDF, used to build and validate the pipeline (including the retrieval-depth experiment below). You can upload this document directly to explore the app, or upload any other PDF.

## DEMO Screenshots



## Tech Stack

- **App:** Streamlit
- **PDF Processing:** pypdf
- **Chunking:** NLTK (sentence-boundary aware, not naive character splitting)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`) — runs locally, no API cost
- **Vector Database:** ChromaDB — persistent local storage
- **LLM:** Llama 3.2, served locally via Ollama — no external API dependency, no cost, no rate limits

## Why these choices

- **Local embeddings + local LLM (Ollama)** instead of a cloud API: avoids external quota/billing dependencies entirely, while still demonstrating the full RAG pattern used in production systems (which typically swap in a hosted LLM API).
- **Sentence-aware chunking**: naive character-based splitting can cut a sentence mid-word, breaking its meaning. Chunking by complete sentences up to a target size keeps each chunk semantically coherent.
- **Retrieval depth (top-k) validated empirically, not guessed**: initially defaulted to top-3 as a standard starting point, then tested k=1/3/5/10 against known-answer questions on the PwC report. The test surfaced a real failure at k=3 — on one question ("how much did PwC invest across its global network"), the correct supporting chunk wasn't in the top-3 results, and rather than saying so, the model hallucinated a plausible but incorrect figure ($1.5B instead of the correct $3.1B). k=5 was the smallest depth that eliminated this failure, with k=10 showing no further improvement. **The app now defaults to top-5** based on this result — see `experiment.ipynb` for the full comparison.
- **Explicit grounding instructions in the prompt**: the model is instructed to answer only from retrieved context and to say so clearly when the context doesn't contain the answer — reducing hallucination, a core motivation for using RAG over a plain LLM call. (Worth noting honestly: this instruction alone didn't fully prevent hallucination when the right chunk was missing at k=3 — grounding wording helps, but retrieving the right context in the first place matters more.)
- **Streamlit over a separate FastAPI + HTML/JS frontend**: the project's substance is the RAG pipeline (extraction, chunking, embeddings, retrieval, grounded generation), not a client-server API layer. Streamlit gives the same upload-and-ask experience in one file, with less surface area to explain and maintain.

## Features

- Upload any PDF and query it conversationally
- Answers are grounded in the actual document — tested explicitly against out-of-scope and missing-context questions
- Structured, readable answer formatting (direct answer + bullet points where relevant)
- Source chunks returned alongside each answer for transparency

## Running Locally

```bash
git clone https://github.com/khushis1103/RAG-Based-Coding-Assistant.git
cd RAG-Based-Coding-Assistant

pip install -r requirements.txt

# Install Ollama from https://ollama.com and pull the model
ollama pull llama3.2

# Run the app
streamlit run app.py
```

## Project Structure

RAG-Based Document Assistant/
├── chroma_db/
├── app.py
├── demo1.png
├── demo2.png
├── experiment.ipynb
├── pwc-global-annual-review-2025.pdf
├── README.md
└── requirements.txt

## Known limitation

- Supports text-based PDFs; scanned PDFs requiring OCR are not supported.
- Local LLM inference can be slower depending on available hardware.

---

Built by Khushi Sahu
