<div align="center">

# Career Launchpad 🎓

An AI‑powered placement assistance platform combining a FastAPI backend (retrieval + LLM fallback) and a React frontend chat experience to help students quickly get answers about campus placements, companies, interview process, salary packages, and related career topics.

*Chat with verified dataset answers first; gracefully fall back to a local Llama model (if available) for broader guidance.*

</div>

## ✨ Key Features

- **Hybrid RAG Flow**: Vector similarity search (FAISS + HuggingFace sentence embeddings) over curated Q/A CSV; summary synthesis via LlamaCpp when available.
- **Relevance Filtering**: Keyword + semantic similarity gating ensures only placement / career queries are processed.
- **Two‑tier Answering**:
	1. Verified data answers (summarized) → marked as `(Based on verified placement data)`.
	2. Fallback LLM response → marked as `(Based on general knowledge — not officially verified)`.
- **Local LLM Friendly**: Works even if the Llama model folder is missing (falls back to context only or dataset).
- **React Chat UI**: Dark mode toggle, persistent chat history (localStorage), quick profile panel, inline typing indicator.
- **Zero External API Costs**: Entirely local inference (embeddings + LlamaCpp) once models are downloaded.
- **CORS Enabled**: Frontend can call backend during development without proxying.

## 🧱 Architecture Overview

```
┌────────────────────────────┐        POST /query/           ┌────────────────────────────┐
│        React Frontend      │  ───────────────────────────▶ │        FastAPI Backend      │
│  (ChatBox.jsx + UI State)  │                               │ - Load CSV -> LangChain Docs │
│  - Input / History / Theme │ ◀───────────────────────────  │ - HuggingFace Embeddings     │
└──────────────┬─────────────┘        JSON response          │ - FAISS Similarity Search    │
							 │                                           │ - Relevance Filter           │
							 │                                           │ - LlamaCpp (optional)        │
							 ▼                                           └──────────────┬───────────────┘
				Local Storage (History)                                      CSV Dataset + Model files
```

## 📂 Repository Structure

```
backend/
	app.py              # FastAPI application (RAG + fallback)
	requirements.txt    # Python dependencies
	placement_data.csv  # (ignored by .gitignore if *.csv is excluded) Q/A dataset
	Space.yaml          # Space / container metadata (app_port)
frontend/
	package.json        # React dependencies & scripts
	src/                # Chat UI components
	public/             # Static assets
.gitignore
README.md
```

## 🛠️ Backend (FastAPI) Details

| Concern | Implementation |
|---------|----------------|
| Framework | FastAPI + Uvicorn |
| Data Source | `placement_data.csv` (Questions + Answers) |
| Vector Store | FAISS (in‑memory) |
| Embeddings | `sentence-transformers/paraphrase-MiniLM-L6-v2` via `HuggingFaceEmbeddings` |
| Semantic Relevance | SentenceTransformer `all-MiniLM-L6-v2` + cosine similarity + keyword heuristic |
| LLM (optional) | `llama-cpp-python` pointing to `model/llama-3.2` folder |
| Prompting | Custom query + summarization templates |
| Endpoint | `POST /query/` returns `{ source, answers: [ ... ] }` |

### Query Flow
1. Receive user query.
2. Reject early if not relevant (keywords + semantic threshold > 0.3).
3. Similarity search top K (default 3) from FAISS.
4. If any results pass score filter (`score < THRESHOLD_SCORE` retained), summarize with Llama (if present) else concatenate.
5. If no valid results → fallback Llama model prompt.
6. Return JSON with origin marker: `filter`, `db+llm`, or `llm`.

### Environment & Models
Place the quantized or full Llama model inside `backend/model/llama-3.2/` (matching `LLAMA_MODEL_PATH`). If absent, backend still serves dataset answers.

## 💬 Frontend (React) Details

Component | Purpose
----------|--------
`ChatBox.jsx` | Core chat interaction, message persistence, theming, typing indicator.
`HistoryPanel.jsx` | (Currently standalone) structure for session grouping (not wired into main UI yet).
`App.js` | Shell that renders `ChatBox`.

Features:
- LocalStorage persistence of entire message array (`chatHistory`).
- Profile dropdown stub (static user info & chat count).
- Dark mode toggling via body class.
- Simple optimistic UI while awaiting backend.

## 🚀 Getting Started

### Prerequisites
- Python 3.11+ (check with `python --version`)
- Node.js 18+ & npm (or yarn)
- (Optional) Llama model files in `backend/model/llama-3.2/`

### 1. Backend Setup
```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
# (Optional) place model weights under model/llama-3.2/
uvicorn app:app --reload --port 8000
```

### 2. Frontend Setup
```powershell
cd frontend
npm install
npm start
```
Frontend runs (default) at http://localhost:3000 and calls backend at `http://127.0.0.1:8000/query/`.

### 3. Configuration Tweaks
You can parameterize thresholds & paths by converting constants in `app.py` to environment variables (future enhancement). For now edit directly:
```python
THRESHOLD_SCORE = 60.0
TOP_K_RESULTS = 3
LLAMA_MODEL_PATH = "model/llama-3.2"
```

## 🔌 API Specification

### POST /query/
Request Body:
```json
{ "query": "What is the average package?" }
```
Response (examples):
```json
// Filtered (irrelevant)
{ "source": "filter", "answers": ["**This assistant is focused only on placement, company, college, and career-related queries.**"] }

// Verified data (with summarization)
{ "source": "db+llm", "answers": ["**(Based on verified placement data)**\n\n<answer text>"] }

// Fallback LLM
{ "source": "llm", "answers": ["**(Based on general knowledge — not officially verified)**\n\n<answer text>"] }
```

## 📊 Dataset Handling
- CSV expected at `backend/placement_data.csv` (ensure headers: `Question`, `Answer`).
- If your `.gitignore` ignores `*.csv`, commit the file explicitly or remove the pattern.
- Data is fully loaded into memory; for large datasets consider chunking or streaming.

## 🧪 Testing Ideas (Not yet implemented)
- Unit test `is_relevant_query` with relevant / irrelevant samples.
- Integration test for `/query/` using FastAPI TestClient.
- Frontend component test for ChatBox localStorage persistence.

## 🔐 Security Considerations
- Current CORS allows `*` (broad). Restrict in production.
- No rate limiting — consider adding if deployed publicly.
- LLM fallback not sandboxed; ensure prompts sanitized if extended.

## ⚙️ Performance Notes
- FAISS in‑memory search is O(log n) for index lookup; with small dataset it's effectively instant.
- SentenceTransformer embeddings computed once at startup for reference queries.
- LlamaCpp parameters (`n_threads`, `n_batch`) tune CPU usage; adjust for your hardware.

## 🧭 Roadmap
- [ ] Configurable env vars for model path & thresholds.
- [ ] Docker Compose for backend + frontend.
- [ ] Replace hard-coded profile with auth (JWT / OAuth).
- [ ] Session grouping & HistoryPanel integration.
- [ ] Streaming token responses.
- [ ] Vector store persistence to disk.
- [ ] Admin panel to update Q/A dataset.
- [ ] Add evaluation metrics for answer quality.

## 🐛 Troubleshooting
Issue | Cause | Fix
------|-------|----
`ModuleNotFoundError` | venv not activated | Activate venv before installing.
`RuntimeError: model not found` | Missing Llama weights | Download / place in `model/llama-3.2`.
Empty `answers` | No docs pass threshold | Lower `THRESHOLD_SCORE` (e.g. 80 -> 60 or 50) or enrich dataset.
CORS error | Browser blocked request | Ensure backend running and CORS allows origin.
High latency | Large model or low threads | Reduce `max_tokens`, increase `n_threads`.

## 🤝 Contributing
1. Fork & clone.
2. Create feature branch: `git checkout -b feature/something`.
3. Commit with conventional summary.
4. Open PR and describe change & testing notes.

## 📄 License
Choose and add a LICENSE file (MIT / Apache-2.0 recommended). Not yet included.

## 🙌 Acknowledgements
- HuggingFace Sentence Transformers
- LangChain & FAISS
- Llama.cpp community
- FastAPI project & React ecosystem

---
Feel free to open issues for feature requests or questions. Happy building! 🚀
