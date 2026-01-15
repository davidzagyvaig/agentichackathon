
# ClaimGraph Implementation Walkthrough

## 🚀 Accomplishments

We have successfully built the core of **ClaimGraph**, a system for grounding LLM outputs in verified scientific literature.

### 1. Backend Architecture
- **FastAPI Server**: Robust API handling paper search, claim extraction, and graph generation.
- **Agentic Modules**:
    - `paper_search.py`: Integrates OpenAlex and Semantic Scholar for real paper data.
    - `claim_extractor.py`: **FIXED** to robustly parse loose JSON from GPT-4o, now reliably extracting 3-6 claims per paper.
    - `citation_validator.py`: Verifies claims against Paper text (simulated/cached).
    - `knowledge_graph.py`: Builds a directed graph of Papers -> Claims -> Citations.
- **Deep Verification**: Added logic to include cited papers as nodes, enabling recursive verification.

### 2. Frontend Interface
- **Next.js + Tailwind CSS**: Modern, responsive web application.
- **Chat-First UX**: Implemented a ChatGPT-style interface for natural interaction.
- **Split-Screen Visualization**: Chat on the left, **Interactive Knowledge Graph** (React Flow) on the right.
- **Visual Status**:
    - 🟢 Green: Verified Claims / Safe Papers
    - 🔴 Red: Suspicious Claims / Retracted Papers
    - ⚫ Dark Mode: Premium "Research Lab" aesthetic.

### 3. Key Features
- **"Deep Dive" Verification**:
    - Users can see citations in the graph.
    - Clicking a citation node allows triggering a "Deep Verify" to analyze that cited paper, expanding the graph recursively.
- **Real-time Feedback**: "Thinking" states and progressive graph updates.

## 🛠️ Repository Status

The code has been pushed to: [https://github.com/davidzagyvaig/agentichackathon](https://github.com/davidzagyvaig/agentichackathon)

**Ignored Files (Not Pushed):**
- `.env` (API Keys)
- `venv/`
- `node_modules/`
- `.gemini/`

## 🏃 How to Run

### Backend
```bash
cd backend
# Win
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

python main.py
```
Server runs at `http://localhost:8000`

### Frontend
```bash
cd frontend
npm install
npm run dev
```
App runs at `http://localhost:3000` (or 3001 if 3000 is taken)

## 🔮 Next Steps
1. **Database Integration**: Set up Supabase (as requested) to persist chat history and graph data.
2. **PDF Parsing**: Integrate full-text PDF parsing (e.g., via Unstructured or LlamaParse) for deeper claim verification beyond abstracts.
3. **Multi-hop Reasoning**: Automate the "Deep Verify" to crawl 2-3 layers deep automatically.
