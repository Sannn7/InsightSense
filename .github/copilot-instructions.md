# Copilot Instructions — InsightSense

Purpose
- Help AI coding agents be immediately productive in this repo: small RAG-backed Streamlit app.

Big picture
- `src/core/` is the "brain": ingestion, hybrid retrieval (BM25 + FAISS), and cross-encoder re-ranking.
  - `rag_pipeline.py` defines: `ingest_documents()`, `build_vectorstore()`, `hybrid_search()`, `cross_encoder_rank()`.
- `src/core/graph_gen.py` converts text summaries into a simple graph structure with `generate_graph_from_summary(summary)`.
- `src/ui/app.py` is the Streamlit UI that calls the pipeline functions and displays JSON graphs.
- `data/` is gitignored and holds raw PDFs, processed pages, and vectorstores.

Developer workflows
- Create venv, install deps, run UI:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
streamlit run src/ui/app.py
```

- Vectorstores & intermediate data are stored under `data/` (not checked in). If you add a persisted vectorstore path, document it in `README.md`.

Project-specific patterns
- "Hybrid retrieval" pattern: combine a sparse lexical retrieval (BM25) with dense retrieval (FAISS). Implement `hybrid_search(query, vectorstore)` to return candidate dicts like `{id, text, score}`.
- Cross-encoder re-ranking: `cross_encoder_rank(candidates, query)` should accept the candidate list and return the list sorted with an added `score` field.
- Small, explicit function API surface: prefer small functions that the UI directly calls. Use the function names listed above so the UI can remain unchanged.

Files to look at for examples
- `src/core/rag_pipeline.py` — retrieval pipeline skeleton and canonical function names.
- `src/core/graph_gen.py` — graph API and output shape example.
- `src/ui/app.py` — how the UI expects pipeline functions to behave.

When changing behavior
- If you change the return shape of `hybrid_search` or `cross_encoder_rank`, update `src/ui/app.py` accordingly and document the shape in `rag_pipeline.py` docstrings.
- If adding a dependency that requires native wheels (FAISS), document install steps in `README.md` and update `requirements.txt`.

Testing & debugging hints
- There are no tests yet; add unit tests under `tests/` mirroring the function names above.
- For local debugging, Streamlit logs are printed to the terminal that launches `streamlit run`.

Non-goals
- The repository intentionally ships skeletons (no heavy persistence). Implementations should be idempotent and local-by-default (store data under `data/`).

If something is unclear
- Ask which pipeline implementation is preferred (embedding model, FAISS vs other DB, BM25 tokenizer settings) before making large changes.
