# RAG Streaming API

FastAPI-based chatbot API using **Retrieval-Augmented Generation (RAG)**, **Qdrant**, an OpenAI-compatible endpoint, and **Server-Sent Events (SSE)** for streaming responses.

The application retrieves relevant chunks from a vector database, uses them as context, and generates grounded answers with the selected LLM.

---

## Run locally

Requirements:

- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- access to a Qdrant instance, such as [Qdrant Cloud](https://cloud.qdrant.io/), where you can create a cluster and obtain the Qdrant URL and API key
- access to the configured embedding and LLM endpoint

After cloning the repository:

```bash
git clone https://github.com/Val2021/rag-streaming-api.git
cd rag-streaming-api
```

Ensure Python 3.14 is available through `uv`:

```bash
uv python install 3.14
```

Install the project dependencies:

```bash
uv sync --frozen
```

`uv` creates the project virtual environment and uses a Python version compatible with the range defined in `pyproject.toml`.

Create the local environment file from the provided example:

```bash
cp .env.example .env
```

Fill in `.env` with the required credentials and service configuration.

Place the source PDF files inside:

```text
data/
```

Create the Qdrant collection:

```bash
uv run python scripts/create_collection.py
```

> The collection creation script recreates the configured collection if it already exists.

### Validate document processing

Before generating embeddings, inspect the chunking pipeline:

```bash
uv run python scripts/ingest.py --dry-run
```

This processes the PDFs and displays sample chunks without generating embeddings or uploading data to Qdrant.

### Validate embedding generation

Before running the complete ingestion, verify that the embedding endpoint is working and that the returned vector dimension matches the Qdrant collection configuration:

```bash
uv run python scripts/test_embedding.py
```

The expected embedding dimension is:

```text
1536
```

### Run the ingestion pipeline

Once document processing and embedding generation have been validated:

```bash
uv run python scripts/ingest.py
```

This executes the complete ingestion flow:

```text
PDF documents
      ↓
Docling
      ↓
HybridChunker
      ↓
filtered chunks
      ↓
embedding generation
      ↓
Qdrant
```

If the embedding endpoint returns an HTTP `429` rate-limit error, the script saves its progress in `.ingest_checkpoint.json`.

Running the same command again:

```bash
uv run python scripts/ingest.py
```

automatically resumes from the saved checkpoint.

### Validate retrieval from Qdrant

After ingestion is complete, verify that the stored vectors can be retrieved correctly:

```bash
uv run python scripts/test_retrieval.py
```

This test generates an embedding for a sample query, searches Qdrant and displays the most relevant chunks and their similarity scores.

### Start the API

After the collection has been populated and retrieval has been validated:

```bash
uv run uvicorn app.main:app --reload
```

Swagger documentation will be available at:

```text
http://127.0.0.1:8000/docs
```

### Validate streaming

With the API running, test the complete RAG flow:

```bash
uv run python scripts/test_stream.py
```

This validates:

```text
query
→ embedding
→ Qdrant retrieval
→ LLM generation
→ SSE streaming
```
---

## Architecture

### Ingestion flow

```text
PDF documents
      ↓
Docling
      ↓
HybridChunker
      ↓
filtered chunks
      ↓
embedding generation
      ↓
Qdrant
```

### Runtime RAG flow

```text
User question
      ↓
query embedding
      ↓
Qdrant similarity search
      ↓
Top-K chunks
      ↓
context construction
      ↓
LLM
      ↓
SSE streaming response
```

Retrieval happens before generation. The LLM receives only the chunks selected by the retriever.

---

## Project structure

```text
.
├── app
│   ├── api
│   │   ├── chat.py
│   │   └── search.py
│   ├── core
│   │   └── settings.py
│   ├── models
│   │   ├── api.py
│   │   └── document.py
│   ├── rag
│   │   ├── embedder.py
│   │   ├── ingestion.py
│   │   ├── prompts.py
│   │   └── retriever.py
│   ├── services
│   │   └── llm_service.py
│   └── main.py
├── data
├── scripts
│   ├── create_collection.py
│   ├── ingest.py
│   ├── test_embedding.py
│   ├── test_retrieval.py
│   └── test_stream.py
├── tests
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## API endpoints

### `GET /health`

Checks whether the FastAPI application is running.

### `GET /chat/models`

Returns the available generation models and the default model.

### `POST /search`

Runs only the retrieval stage.

It embeds the user query, searches Qdrant, and returns the most relevant chunks with their metadata and similarity scores.

### `POST /chat/stream`

Runs the complete RAG flow.

The endpoint:

1. retrieves the most relevant chunks from Qdrant;
2. builds the context;
3. sends the context and query to the selected LLM;
4. streams the generated response using SSE.

The stream contains:

```text
sources
text_delta
...
final
done
```

`text_delta` contains the incremental response and `final` contains the complete concatenated answer.

---

## Technical decisions

### Docling

Docling is used to process the PDF documents while preserving document structure such as headings, sections, and text blocks.

OCR is disabled because the source PDFs contain digitally extractable text.

---

### HybridChunker

The ingestion pipeline uses Docling's `HybridChunker`.

The goal is to combine document structure with a token-size constraint instead of splitting text only by fixed character positions.

The configuration also uses:

```python
merge_peers=True
```

which allows compatible neighboring fragments to be merged when they still fit inside the configured token limit.

---

### Chunk size

The chunking configuration uses:

```python
tokenizer_model = "sentence-transformers/all-MiniLM-L6-v2"
max_tokens = 500
```

The tokenizer is used only to measure chunk size.

It does not generate the embeddings stored in Qdrant.

The actual embeddings are generated using:

```text
cohere/embed-v4.0
```

A maximum of 500 tokens was chosen as a balance between preserving useful local context and avoiding chunks that are too large and introduce retrieval noise.

---

### MiniLM tokenizer

`sentence-transformers/all-MiniLM-L6-v2` is used only as the tokenizer consumed by `HybridChunker`.

Its role is:

```text
text
→ tokenization
→ chunk-size control
```

Embedding generation is handled separately:

```text
text
→ cohere/embed-v4.0
→ vector
```

---

### Qdrant

Qdrant is used as the vector database because the main retrieval operation is semantic vector search.

It provides:

- vector similarity search;
- cosine similarity;
- metadata payloads;
- Top-K retrieval;
- Python client support.

The collection is configured with:

```text
Vector dimension: 1536
Distance: Cosine
```

The similarity score returned by Qdrant is used to rank retrieved chunks.

It represents similarity between the query embedding and the stored chunk embedding, not the quality of the final generated answer.

---

### Top-K retrieval

The default retrieval configuration is:

```text
Top-K = 5
```

This provides multiple relevant candidates to the LLM while keeping the context focused.

The API accepts values between 1 and 10.

---

### Temperature

The generation temperature is:

```text
0.2
```

The application is designed for grounded question answering instead of creative generation.

A low temperature helps produce more consistent answers that remain focused on the retrieved context.

---

### Portuguese prompt

The RAG prompt is written in Portuguese because both the source documents and the expected user queries are primarily in Portuguese.

The selected LLM is multilingual, so an English prompt could also work with Portuguese documents and questions. However, Portuguese was chosen to keep the instructions aligned with the language of the knowledge base and the expected user input, reducing unnecessary language switching and helping maintain more consistent responses.

The prompt instructs the model to:

- answer only using the provided context;
- avoid unsupported information;
- state when the available context is insufficient.

OpenAI documents that its models can understand and generate text in multiple languages and that prompts can be submitted in different languages:
[OpenAI — How can I use the OpenAI API with text in different languages?](https://help.openai.com/en/articles/6742369-how-can-i-use-the-openai-api-with-text-in-different-languages)

---

### OpenAI-compatible client

The application uses `AsyncOpenAI` with a configurable `base_url`.

This allows both embedding and generation models to be accessed through a common OpenAI-compatible interface.

---

### Server-Sent Events

SSE is used because the generation flow is mainly one-way:

```text
server
↓
client
```

This allows the API to send the response progressively while the model is still generating it.

---

## Ingestion and checkpoint recovery

The ingestion pipeline runs as:

```text
PDF
→ Docling
→ HybridChunker
→ embedding batches
→ Qdrant
```

Embeddings are generated in batches of:

```text
32 chunks
```

If the embedding endpoint returns an HTTP `429` rate-limit error, the script stores its progress in:

```text
.ingest_checkpoint.json
```

After every successful batch, the next chunk index is saved.

To continue after an interruption, run again:

```bash
uv run python scripts/ingest.py
```

The script automatically detects the checkpoint and resumes from the saved position.

The checkpoint file is removed after a successful complete ingestion.

---

## Testing

Embedding integration:

```bash
uv run python scripts/test_embedding.py
```

Retrieval:

```bash
uv run python scripts/test_retrieval.py
```

Streaming:

```bash
uv run python scripts/test_stream.py
```

---

## Possible improvements

### Deterministic Qdrant IDs

The current ingestion process generates point IDs with:

```python
uuid4()
```

This guarantees unique IDs, but rerunning the full ingestion against the same collection may create duplicate logical chunks.

A possible improvement would be to generate deterministic IDs, for example:

```text
UUID5(source + chunk_id)
```

or a stable content hash.

This would make the ingestion process idempotent.

---

### Hybrid retrieval

The current retrieval implementation uses dense semantic search.

A future version could combine dense retrieval with sparse lexical retrieval and fuse the rankings using Reciprocal Rank Fusion.

---

### Reranking

A reranking stage could be added after the initial retrieval:

```text
Qdrant candidates
↓
reranker
↓
best context
↓
LLM
```

A Cross-Encoder could be used to improve the ordering of candidates before generation.

---

### Async Qdrant client

The embedding and generation integrations are asynchronous, while the current Qdrant integration uses the synchronous client.

For higher concurrency, the retrieval layer could use Qdrant's asynchronous client.

---

### Dependency separation

Document ingestion requires heavier dependencies such as Docling.

Since the runtime API does not process PDFs during normal requests, ingestion dependencies and API runtime dependencies could be separated.

---

### Authentication and rate limiting

A production version should protect model-backed endpoints with authentication and application-level rate limiting.
