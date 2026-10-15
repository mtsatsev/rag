"""Command-line entry point: `rag ingest` and `rag ask "<question>"`."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "src" / "data"
INDEX_DIR = ROOT / ".chroma"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rag", description="Fund documentation assistant"
    )
    parser.add_argument(
        "--index", type=Path, default=INDEX_DIR, help="Chroma index folder"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest_parser = commands.add_parser(
        "ingest", help="Build or update the index from the PDFs"
    )
    ingest_parser.add_argument(
        "--data", type=Path, default=DATA_DIR, help="Folder with the PDFs"
    )

    ask_parser = commands.add_parser("ask", help="Answer a question, printed as JSON")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--model", default="qwen3:4b", help="Ollama model")
    ask_parser.add_argument(
        "--k", type=int, default=5, help="Chunks given to the model"
    )
    ask_parser.add_argument(
        "--min-score", type=float, default=0.35, help="Below this: not found"
    )
    ask_parser.add_argument(
        "--num-ctx", type=int, default=8192, help="Model context size"
    )

    args = parser.parse_args(argv)
    if args.command == "ingest":
        run_ingest(args)
    else:
        run_ask(args)


def run_ingest(args: argparse.Namespace) -> None:
    # Heavy imports are deferred so `rag --help` stays fast.
    from rag.index.embed import BgeM3Embedder
    from rag.index.store import ChunkStore
    from rag.ingest.funds import load_registry
    from rag.ingest.runner import ingest

    store = ChunkStore(args.index, BgeM3Embedder())
    report = ingest(args.data, store, load_registry())
    print(report.model_dump_json(indent=2))
    print(f"Chunks in index: {store.count}")


def run_ask(args: argparse.Namespace) -> None:
    from rag.generate.answer import Answerer
    from rag.generate.llm import OllamaClient
    from rag.index.embed import BgeM3Embedder
    from rag.index.store import ChunkStore
    from rag.ingest.funds import load_registry
    from rag.retrieve.dense import build_dense_retriever

    store = ChunkStore(args.index, BgeM3Embedder())
    answerer = Answerer(
        build_dense_retriever(store, load_registry()),
        OllamaClient(model=args.model, num_ctx=args.num_ctx),
        k=args.k,
        min_score=args.min_score,
    )
    print(answerer.answer(args.question).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
