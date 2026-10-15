# Take Home Assignment

## Preface
The project runs two models locally, so you'll need a machine that can download and run them
* qwen3:4b (4-bit quantised, ~2.5 GB) generates the answers. It runs through `Ollama`, so `Ollama` has to be installed and running, and the model pulled once with `ollama pull qwen3:4b`.
* BAAI/bge-m3 (~2.2 GB) embeds the document chunks and the questions. It's loaded with `sentence-transformers` and downloaded from Hugging Face automatically the first time you run `rag ingest`. I developed and tested everything on a laptop with an NVIDIA RTX 3060 (6 GB VRAM) under WSL2 (8 GB RAM). Both models fit on that GPU together. They can also run on a cpu, just slower.

## Installation 
Clone the repository
```sh
git clone git@github.com:mtsatsev/rag.git
```
Alternatively download it as `.zip` and unpackage it.

The project has all of its dependencies under the `pyproject.toml` and utilized `uv` as package manager. From the root directory execute
```sh
pip install uv
uv sync --all-extras --all-groups     
```

## Usage
From the root directory run
```sh
source .venv/bin/activate
```

To build the index move the documents into the `src/data` folder and run
```sh
rag ingest
```
This should create an index called `.chroma` in the root of this repository. If you want to change the path of either the index or the documents you can use 
```sh
rag --index /path/to/index ingest --data /path/to/data
```
However, keep in mind that if you provided an index path then you must also pass the same path when you ask questions.

To query use
```sh
rag ask "<your query goes here>"
```
If a different index is used besides `.chroma` then use

```sh
rag --index /path/to/index ask "<your query goes here>"
```
Alternatively you can run it from a `ipynb` notebook. In VSCode first connect the kernel to the virtual environment of the repository then use

```py
from rag.generate.answer import Answerer
from rag.generate.llm import OllamaClient
from rag.index.embed import BgeM3Embedder
from rag.index.store import ChunkStore
from rag.ingest.funds import load_registry
from rag.ingest.runner import ingest
from rag.retrieve.dense import build_dense_retriever
```

```py
DATA_DIR = "/path/to/documents"
INDEX_DIR = "/path/to/index"
```

```py
# Build or update the index (unchanged PDFs are skipped)
store = ChunkStore(INDEX_DIR, BgeM3Embedder())
registry = load_registry()
print(ingest(DATA_DIR, store, registry))
```

```py
answerer = Answerer(build_dense_retriever(store, registry), OllamaClient())
answer = answerer.answer("Can I transfer my fund units to another person?")
print(answer.model_dump_json(indent=2))
```
