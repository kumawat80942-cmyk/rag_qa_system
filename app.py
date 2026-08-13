"""
    streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.chunker import build_chunks
from src.document_loader import load_document
from src.generator import AnswerGenerator
from src.llm_backends.gemini_backend import DEFAULT_MODEL as GEMINI_DEFAULT
from src.llm_backends.ollama_backend import DEFAULT_HOST as OLLAMA_DEFAULT_HOST
from src.llm_backends.ollama_backend import DEFAULT_MODEL as OLLAMA_DEFAULT
from src.llm_backends.openai_backend import DEFAULT_MODEL as OPENAI_DEFAULT
from src.retriever import TfidfRetriever

load_dotenv()

CHUNK_SIZE = 3
CHUNK_OVERLAP = 1
TOP_K = 3

PROVIDER_DEFAULTS = {
    "gemini": GEMINI_DEFAULT,
    "openai": OPENAI_DEFAULT,
    "ollama": OLLAMA_DEFAULT,
}

st.set_page_config(page_title="Stacks — RAG Document Q&A", layout="wide")

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

st.title("RAG Document Q&A")
st.caption(
    "Chunk a document, index it with TF-IDF, retrieve the most relevant "
    "passages for a question, and generate a grounded answer with a "
    "pluggable LLM backend."
)

with st.sidebar:
    st.subheader("LLM backend")
    provider = st.selectbox("Provider", ["gemini", "openai", "ollama"])
    model = st.text_input("Model", value=PROVIDER_DEFAULTS[provider])

    user_api_key = ""
    if provider == "gemini":
        user_api_key = st.text_input(
            "Gemini API Key",
            value=os.environ.get("GEMINI_API_KEY", ""),
            type="password",
            help="Get your key at https://aistudio.google.com/app/apikey",
        )
        st.caption("Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey)")
    elif provider == "openai":
        user_api_key = st.text_input(
            "OpenAI API Key",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
            help="Get your key at https://platform.openai.com/api-keys",
        )
        st.caption("Reads key from input above or `.env` file.")
    else:
        host = st.text_input("Ollama host", value=os.environ.get("OLLAMA_HOST", OLLAMA_DEFAULT_HOST))
        st.caption("Requires a local `ollama serve` with the model pulled.")

left, right = st.columns([1, 1.3])

with left:
    st.subheader("1. Document")

    uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
    pasted_text = st.text_area("...or paste text here", height=220)

    process_clicked = st.button("Process document", type="primary")

    if process_clicked:
        raw_text = ""
        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            raw_text = load_document(tmp_path)
        elif pasted_text.strip():
            raw_text = pasted_text

        if not raw_text.strip():
            st.warning("Paste some text or upload a file first.")
        else:
            chunks = build_chunks(
                raw_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            retriever = TfidfRetriever()
            retriever.fit(chunks)

            st.session_state.chunks = chunks
            st.session_state.retriever = retriever
            st.success(
                f"Indexed {len(chunks)} chunks, "
                f"{retriever.vocabulary_size} vocabulary terms."
            )

    if st.session_state.chunks:
        with st.expander(f"Chunk index ({len(st.session_state.chunks)} chunks)"):
            for c in st.session_state.chunks:
                st.markdown(f"**#{c.index + 1}** — {c.text[:150]}{'…' if len(c.text) > 150 else ''}")

with right:
    st.subheader("2. Ask")

    question = st.text_input("Your question")
    ask_clicked = st.button("Ask", type="primary")

    if ask_clicked and question.strip():
        if st.session_state.retriever is None:
            st.warning("Please upload and process a document first.")
        else:
            with st.spinner("Retrieving passages..."):
                passages = st.session_state.retriever.retrieve(question, top_k=TOP_K)

            if not passages:
                st.warning("No passages matched this question closely enough.")
            else:
                st.markdown("**Retrieved passages**")
                for p in passages:
                    st.markdown(
                        f"`Passage #{p.chunk.index + 1} · {p.score:.0%} match`\n\n"
                        f"{p.chunk.text}"
                    )
                    st.divider()

                with st.spinner(f"Generating answer with {provider}..."):
                    try:
                        backend_kwargs = {}
                        if provider == "ollama":
                            backend_kwargs["host"] = host
                        elif user_api_key.strip():
                            backend_kwargs["api_key"] = user_api_key.strip()
                        generator = AnswerGenerator(
                            provider=provider, model=model, **backend_kwargs
                        )
                        answer = generator.generate_answer(question, passages)
                        st.markdown("**Answer**")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Could not generate an answer: {e}")
