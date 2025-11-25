# InsightSense

Lightweight skeleton for an RAG-backed Streamlit app.

Structure
- `data/` — gitignored, local PDFs & vectorstores.
- `src/core/` — pipelines (ingest, hybrid search, cross-encoder, graph generation).
- `src/ui/` — Streamlit dashboard `app.py`.

Quick start
```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
streamlit run src/ui/app.py
```

Data & vectorstores live under `data/`. The repository intentionally stores only code and skeletons.
