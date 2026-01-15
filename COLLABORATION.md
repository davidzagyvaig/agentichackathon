# 🤝 Hackathon Collaboration Guide

Since "everything is stressful" right now, follow these simple rules to avoid git conflicts and keep moving fast.

## ⚡ Quick Setup for Co-founder
1. **Clone the repo:**
   ```bash
   git clone https://github.com/davidzagyvaig/agentichackathon.git
   cd agentichackathon
   ```
2. **Backend Setup:**
   ```bash
   cd backend
   # Create virtual env
   python -m venv venv
   # Activate (Windows)
   venv\Scripts\activate
   # Activate (Mac)
   source venv/bin/activate
   # Install deps
   pip install fastapi uvicorn openai networkx python-dotenv
   ```
   *Create a `.env` file in `backend/` with `OPENAI_API_KEY=...`*

3. **Frontend Setup:**
   ```bash
   cd frontend
   npm install
   ```

## 🛡️ Anti-Conflict Rules

### Rule 1: Divide and Conquer
To avoid merge conflicts, **work on different files**.
- **David (User):** Focus on Frontend (Next.js) + Graph Visualization.
- **Co-founder:** Focus on Backend (Supabase integration, Database) or PDF Parsing.

### Rule 2: Use Feature Branches
Never push directly to `main` unless it's a tiny fix.
```bash
# David
git checkout -b david/frontend-polish

# Co-founder
git checkout -b cofounder/database-setup
```

### Rule 3: Sync Often
Before you start working, always pull:
```bash
git checkout main
git pull origin main
git checkout your-branch
git merge main
```

## 🚀 Current Status (What works)
- **Backend:** `main.py` is running. `/api/analyze` works. `/api/graph` works.
- **Frontend:** Chat UI is working. Graph Visualization is connected.
- **Deep Link:** Citation verification backbone is ready.

## 📋 To-Do List (Divide these!)
1. **Database (Co-founder?):**
   - Setup Supabase.
   - Create tables for `chats`, `messages`, `graphs`.
   - Update backend to save/load from DB.
2. **Frontend Features (David?):**
   - Improve Graph styling.
   - Add "Export Report" button.
   - Mobile responsiveness.
3. **Advanced Backend:**
   - Add PDF parsing (unstructured.io or llama-parse).
   - Improve "Deep Verify" to be multi-step.
