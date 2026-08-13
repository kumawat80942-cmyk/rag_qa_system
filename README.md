# Stacks — RAG-Based Document Question Answering System

A minor project implementing Retrieval-Augmented Generation (RAG) for
question answering over user-supplied documents.

## Pipeline

```
Document --> Chunker --> TF-IDF Retriever --> Top-k passages --> Generator (pluggable LLM) --> Answer
```

1. **Chunking** (`src/chunker.py`) — splits the document into sentences and
   groups them into overlapping windows, so context isn't lost at chunk
   boundaries.
2. **Retrieval** (`src/retriever.py`) — builds a TF-IDF index over the
   chunks with scikit-learn and ranks chunks against a question using
   cosine similarity. No external API is used for retrieval — everything
   runs locally, no Cohere or any other embedding service required.
3. **Generation** (`src/generator.py` + `src/llm_backends/`) — sends the
   top-ranked chunks plus the question to an LLM, instructed to answer
   only from the supplied excerpts and cite which excerpt(s) it used.
    The LLM is **pluggable**: choose Gemini, OpenAI, or a local Ollama
    model without touching retrieval code.

### Pluggable LLM backends

All backends implement the same interface (`src/llm_backends/base.py`),
so switching providers is a one-line change:

```python
from src.generator import AnswerGenerator

# Gemini
gen = AnswerGenerator(provider="gemini", model="gemini-2.5-flash")

# OpenAI
gen = AnswerGenerator(provider="openai", model="gpt-4o-mini")

# Local model via Ollama — no API key, runs on your machine
gen = AnswerGenerator(provider="ollama", model="llama3")
```

In the Streamlit UI, this is a dropdown in the sidebar. In the CLI, use
`--provider` and `--model`.

To add a new provider (e.g. a self-hosted vLLM server), create
`src/llm_backends/your_backend.py` implementing `LLMBackend.generate()`,
then register it in `PROVIDERS` inside `src/llm_backends/__init__.py`.

## Project structure

```
rag_qa_system/
├── app.py                  # Streamlit web UI (provider dropdown in sidebar)
├── cli.py                  # Command-line interface (--provider flag)
├── requirements.txt
├── .env
├── data/
│   └── sample_document.txt
└── src/
    ├── __init__.py
    ├── document_loader.py  # load .txt / .pdf files
    ├── chunker.py           # sentence splitting + chunk building
    ├── retriever.py         # TF-IDF index + cosine similarity retrieval
    ├── generator.py         # builds the grounded prompt, calls a backend
    └── llm_backends/
        ├── base.py           # LLMBackend interface every provider implements
        ├── openai_backend.py
        ├── gemini_backend.py
        ├── ollama_backend.py # local models, no API key needed
        └── __init__.py       # get_llm_backend() factory
```

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Fill in the key(s) in `.env` for whichever provider(s) you plan to use:

   - **Gemini** — get a key from https://aistudio.google.com/app/apikey
   - **OpenAI** — get a key from https://platform.openai.com/api-keys
   - **Ollama** — no key needed. Install from https://ollama.com/download,
     then run `ollama pull llama3` (or any model you like) before use.

## Running it

**Web UI (Streamlit):**

```bash
streamlit run app.py
```

Opens in your browser. Paste or upload a document, click "Process
document", then ask questions.

**Command line:**

```bash
python cli.py --file data/sample_document.txt
```

This loads a document, processes it, and drops you into an interactive
question loop in the terminal.

## Configuration

Key parameters you can tune in `app.py` / `cli.py`:

- `chunk_size` — sentences per chunk (default 3)
- `chunk_overlap` — sentences shared between consecutive chunks (default 1)
- `top_k` — number of passages retrieved per question (default 3)
- `model` — Gemini model used for generation (default `gemini-2.5-flash`)

## Notes for the project report

- Retrieval uses classic TF-IDF + cosine similarity rather than dense
  neural embeddings, so the whole pipeline is explainable without a
  vector database — a reasonable scope for a minor project. Swapping in
  sentence embeddings (e.g. `sentence-transformers`) plus a vector store
  (e.g. FAISS) is a natural "future work" extension and only requires
  changing `src/retriever.py`.
- The generator is instructed (via the system prompt) to answer only
  from retrieved context and to say so when the answer isn't present —
  this reduces hallucination and is the core idea behind RAG.
