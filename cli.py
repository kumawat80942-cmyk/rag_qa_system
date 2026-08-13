"""
cli.py

Command-line interface for the RAG-based Document Question Answering
System. Works with any configured LLM backend — Gemini, OpenAI, or a
local Ollama model.

Usage:
    python cli.py --file data/sample_document.txt
    python cli.py --file data/sample_document.txt --provider gemini
    python cli.py --file data/sample_document.txt --provider openai --model gpt-4o-mini
    python cli.py --file data/sample_document.txt --provider ollama --model llama3
"""

import argparse

from dotenv import load_dotenv

from src.chunker import build_chunks
from src.document_loader import load_document
from src.generator import AnswerGenerator
from src.retriever import TfidfRetriever


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="RAG document Q&A (CLI)")
    parser.add_argument("--file", required=True, help="Path to a .txt or .pdf document")
    parser.add_argument("--chunk-size", type=int, default=3)
    parser.add_argument("--chunk-overlap", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--provider", choices=["gemini", "openai", "ollama"], default="gemini"
    )
    parser.add_argument("--model", default=None, help="Model name (optional, provider has a default)")
    parser.add_argument("--ollama-host", default=None, help="Only used with --provider ollama")
    args = parser.parse_args()

    print(f"Loading {args.file} ...")
    text = load_document(args.file)

    chunks = build_chunks(text, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    print(f"Split into {len(chunks)} chunks.")

    retriever = TfidfRetriever()
    retriever.fit(chunks)
    print(f"Indexed {retriever.vocabulary_size} vocabulary terms.")

    backend_kwargs = {}
    if args.provider == "ollama" and args.ollama_host:
        backend_kwargs["host"] = args.ollama_host

    generator = AnswerGenerator(provider=args.provider, model=args.model, **backend_kwargs)
    print(f"Using provider: {args.provider} ({args.model or 'default model'})")

    print("\nReady. Type a question, or 'quit' to exit.\n")
    while True:
        question = input("? ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        passages = retriever.retrieve(question, top_k=args.top_k)
        if not passages:
            print("  No relevant passages found for that question.\n")
            continue

        print("\n  Retrieved passages:")
        for p in passages:
            preview = p.chunk.text[:100] + ("…" if len(p.chunk.text) > 100 else "")
            print(f"   [{p.score:.0%}] #{p.chunk.index + 1}: {preview}")

        answer = generator.generate_answer(question, passages)
        print(f"\n  Answer: {answer}\n")


if __name__ == "__main__":
    main()
