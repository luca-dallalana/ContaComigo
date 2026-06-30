# IRS Assistant (Portugal)

A RAG (Retrieval-Augmented Generation) chatbot that answers Portuguese income tax (IRS) questions accurately, citing the exact article and page of official AT documents. Built as an alternative to the AT portal's chatbot, which frequently gives vague or incorrect answers on specific legal questions.

## Motivation

Navigating Portuguese tax law is hard. The official AT portal has a chatbot, but it consistently fails on specific questions, giving generic answers or simply refusing to answer. Tax professionals are expensive and not always accessible. The CIRS (Código do IRS) alone runs to hundreds of articles.

The goal was to build a system that:
- Answers specific IRS questions correctly, citing the exact legal source
- Is grounded exclusively in official AT documents, no hallucination
- Is cheap to run (Groq free tier covers normal conversational usage)
- Is verifiable: every answer links back to the original PDF at the exact page

## Architecture

```
User question
     │
     ▼
┌─────────────────────────────────────────┐
│              Hybrid Retrieval           │
│                                         │
│  nomic-embed-text (Ollama, local)       │
│         │                               │
│         ▼                               │
│  pgvector HNSW ──┐                      │
│  (semantic)      ├── RRF Fusion ──▶ Top-K chunks
│  BM25 index  ────┘                      │
│  (keyword)                              │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│              Generation                 │
│                                         │
│  llama-3.3-70b-versatile (Groq cloud)   │
│  or llama3.1:8b (Ollama, local)         │
│                                         │
│  Strict context-only prompt             │
│  Temperature: 0.1                       │
└─────────────────────────────────────────┘
     │
     ▼
Answer + source citations + PDF links (page-accurate)
```

**Stack:**
- **Vector DB:** PostgreSQL + pgvector (HNSW index, cosine similarity)
- **Embeddings:** `nomic-embed-text` via Ollama, runs locally, no API cost, 768-dim vectors
- **BM25:** Custom in-database BM25 implementation for keyword recall
- **Fusion:** Reciprocal Rank Fusion (RRF) combining vector and BM25 rankings
- **Generation:** Groq API (`llama-3.3-70b-versatile`) or local Ollama (`llama3.1:8b`)
- **UI:** Streamlit with dark/light mode, clickable source cards linking to AT PDFs
- **Eval:** Custom 35-question harness with retrieval hit + answer quality scoring

## Source Documents

All documents sourced directly from the official AT portal (portaldasfinancas.gov.pt) and kept up to date:

| Document | Year | Chunks |
|---|---|---|
| Código do IRS (CIRS) | 2025 | 576 |
| Estatuto dos Benefícios Fiscais (EBF) | 2025 | 433 |
| Guia de Deduções IRS | 2025 | 20 |
| Principais Prazos IRS | 2025 | 16 |
| Ofício Circulado 20269/2024 | 2024 | 44 |

Documents are automatically downloaded and re-ingested when updated. The downloader always pulls the latest version from AT's canonical PDF URLs.

## Retrieval Design

The core challenge in legal RAG is that questions use natural language while source documents use formal legal terminology. Semantic search alone misses keyword-specific legal terms; BM25 alone misses conceptual matches.

**Hybrid approach:** Vector search (HNSW cosine similarity) + BM25, fused via Reciprocal Rank Fusion:

```
RRF score = (VECTOR_WEIGHT / (rank_v + 60)) + (BM25_WEIGHT / (rank_b + 60))
```

The constant 60 comes from the original RRF paper (Cormack et al., 2009) where it was empirically tuned on IR benchmarks. It acts as a smoothing factor, without it, `1/rank` makes the top result dominate heavily (rank 1 scores 1.0, rank 2 scores 0.5). With `+60`, ranks 1 and 2 score `1/61` and `1/62` respectively, making fusion more democratic: a result ranking 2nd in both methods can outscore one that ranks 1st in only one.

**Tuned parameters:**
- `VECTOR_WEIGHT=0.7`, `BM25_WEIGHT=0.3` — vector dominates for semantic recall
- `TOP_K=12` — each method returns 12 candidates before fusion

## Experiments

### Retrieval Weight Tuning

Tested three configurations against the 35-question eval set:

| Config | Saúde/Habitação questions | Regime Simplificado | Category A |
|---|---|---|---|
| 0.7/0.3, TOP_K=5 | ✓ | ✗ | ✗ |
| 0.5/0.5, TOP_K=8 | ✗ | ✓ | ✓ |
| 0.7/0.3, TOP_K=12 | ✓ | ✓ | ✓ |

Equal weights (0.5/0.5) fixed keyword recall but broke semantic recall for Guia de Deduções questions. The root cause was RRF weight math: with TOP_K=5 and VECTOR_WEIGHT=0.7, a BM25-only result at rank 1 scores `0.3/61 ≈ 0.005`, while a vector result at rank 5 scores `0.7/65 ≈ 0.011`. Vector dominates even for keyword-specific legal terms. Increasing TOP_K to 12 gives both pools enough candidates for RRF to surface the right results regardless of which method ranks a chunk first.

### Model Comparison (8b vs 70b)

| Model | Eval pass rate | Factual accuracy (manual review of all 35 answers) |
|---|---|---|
| llama3.1:8b (local Ollama) | 35/35 | ~60% — frequent hallucination despite strict prompt |
| llama-3.3-70b-versatile (Groq) | 35/35 | ~100% — context-only, well-sourced |

The automated eval metric (retrieval hit + non-empty answer) masks the quality gap entirely, both models score 35/35. Manual review of every answer exposed ~14 factually wrong responses from the 8b model: invented tax rates, wrong income categories (e.g. saying pensions are Category A instead of H), wrong deduction limits, hallucinated values with no basis in the retrieved chunks.

The 70b model reliably follows context-only constraints and every manually verified answer was correct and cited the exact article.

**Key finding:** Pass rate on a recall-based eval is not a good proxy for factual accuracy. Real quality gaps in legal RAG only surface through manual review.

### Prompt Engineering

Iterated from a basic instruction prompt to an 8-rule strict prompt:
- Explicit ban on using prior knowledge, even when the model "knows" the answer
- Prohibition on estimating or inferring any numbers not literally present in the retrieved chunks
- Required fallback phrase when context is insufficient
- Mandatory source citation (document + article) for every claim

The strict prompt meaningfully improved 70b output. For 8b, it helped partially but the model continued hallucinating on specific numerical facts, a model capability issue, not a prompt design issue.

### Infrastructure

Tested AWS EC2 `g5.xlarge` (NVIDIA A10G, 24GB VRAM) for self-hosted 70b inference as an alternative to Groq. Quantized llama-3.3-70b runs on 24GB with Q4 quantization. The EC2 path eliminates rate limits but adds ~$1/hr in compute cost. Groq free tier is sufficient for normal single-user conversational usage and is the production default.

## Evaluation

Custom 35-question harness covering the full range of IRS topics:

- Deadlines and filing procedures
- Income categories (A, B, E, F, G, H)
- Deductions (health, education, housing, dependents, PPR, donations)
- Special regimes (regime simplificado, tributação conjunta)
- Tax rates and thresholds
- Special provisions (incapacidade permanente, mais-valias reinvestimento)
- Tax benefits (Programa Semente, EBF)

A question passes if: retrieval hit (expected source in top-K) AND model produces a non-empty answer without the configured fallback phrase.

**Result: 35/35 with TOP_K=12 and llama-3.3-70b-versatile.**

Run it:
```bash
python eval/run_eval.py --output eval/results.json
# Re-run only failed questions from a previous run:
python eval/run_eval.py --failed-from eval/results.json
```

## Running Locally

### Quickstart

Clone the repo and run the setup script for your OS, it installs all dependencies, configures the database, downloads documents, and starts the app:

```bash
git clone https://github.com/luca-dallalana/ContaComigo.git
cd ContaComigo

# macOS
bash setup_mac.sh

# Linux
bash setup_linux.sh
```

The script will prompt for a PostgreSQL username, password, and your Groq API key. Get a free Groq key at [console.groq.com](https://console.groq.com) → sign up → API Keys → Create API Key.

Open http://localhost:8501.

### Manual setup

If you prefer to set up manually:

```bash
# Start the database
docker compose up -d

# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pull the embedding model
ollama pull nomic-embed-text

# Configure environment
cp .env.example .env
# Fill in POSTGRES_USER, POSTGRES_PASSWORD, and GROQ_API_KEY in .env
# Get a free Groq key at https://console.groq.com → API Keys → Create API Key

# Download documents and build the vector index (~2-3 min)
python scripts/ingest.py
python scripts/insert_curated_chunks.py

# Run the app
streamlit run app/streamlit_app.py
```

### Running with a local model (no Groq account required)

```bash
ollama pull llama3.1:8b
```

In `.env`:
```
INFERENCE_BACKEND=ollama
GENERATION_MODEL=llama3.1:8b
```

The 8b model works for exploring the pipeline but gives factually unreliable answers on specific legal questions. The 70b model via Groq is recommended for real usage.

### CLI

```bash
python cli/chat.py
```

### Re-ingesting updated documents

```bash
python scripts/ingest.py --force
```

The ingest script downloads the latest versions of all documents directly from AT's canonical URLs.

## Project Structure

```
ContaComigo/
├── app/                   # Streamlit UI (dark/light mode, source cards, PDF links)
├── cli/                   # Terminal chat interface
├── eval/
│   ├── questions.yaml     # 35 test cases with expected sources
│   └── run_eval.py        # Eval runner (--failed-from, --delay flags)
├── generation/
│   ├── client_factory.py  # Picks Groq or Ollama based on INFERENCE_BACKEND
│   ├── errors.py          # LLMConnectionError, LLMRateLimitError
│   ├── groq_client.py     # Groq API client (streaming, raises on rate limit)
│   ├── ollama_client.py   # Ollama client
│   └── prompt.py          # Strict context-only system prompt
├── ingestion/
│   ├── downloader.py      # Document sources and AT URL mapping
│   └── parser.py          # PDF chunking by article
├── retrieval/
│   ├── bm25_search.py     # In-database BM25
│   ├── fusion.py          # Reciprocal Rank Fusion
│   └── vector_search.py   # pgvector HNSW search
├── scripts/
│   └── ingest.py          # End-to-end ingestion pipeline
└── .env.example
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_URL` | — | PostgreSQL connection string |
| `INFERENCE_BACKEND` | `groq` | `groq` or `ollama` |
| `GROQ_API_KEY` | — | Required when using Groq |
| `GENERATION_MODEL` | `llama-3.3-70b-versatile` | Model tag |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBED_MODEL` | `nomic-embed-text:latest` | Embedding model |
| `EMBED_DIMENSION` | `768` | Embedding vector dimension |
| `TOP_K` | `12` | Candidates per retrieval method before RRF fusion |
| `VECTOR_WEIGHT` | `0.7` | RRF weight for vector search |
| `BM25_WEIGHT` | `0.3` | RRF weight for BM25 |

## Key Design Decisions

**Why hybrid retrieval?** Legal documents have two retrieval failure modes: semantic search misses exact legal terms (e.g. "regime simplificado" as a keyword), and BM25 misses conceptual questions that don't use the exact document vocabulary. RRF fusion handles both with a single tunable weight parameter.

**Why local embeddings?** Embedding 1,000+ chunks is a one-time cost. Keeping it local means zero API cost, no rate limits during ingestion, and vectors stay in the database permanently regardless of upstream API changes.

**Why Groq free tier?** At 6,000 tokens/minute, the free tier is sufficient for conversational usage. A user asking 5-10 questions in a session stays well within limits. Rate limits only surface during bulk operations like running the full eval in a tight loop.

**Why 70b over 8b?** Tax law is a domain where factual precision matters. A wrong deduction limit or income category classification is worse than no answer. The 70b model reliably suppresses prior knowledge in favour of retrieved context; 8b cannot do this consistently even with strict prompting.

**Why chunk by article?** Each CIRS or EBF article is a discrete legal unit with a specific scope. Chunking by article preserves legal coherence, a chunk about Article 78-C contains exactly that topic, not a fragment that starts mid-rule. Long articles are split into overlapping sub-chunks to stay within embedding context limits.
