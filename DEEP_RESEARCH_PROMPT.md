# Deep Research Prompt: ClaimGraph - Scientific Discovery Acceleration System

## Context & Background

We are participating in the **Umrand A1 Science First Science Hackathon**. Our goal is to build a multi-agentic system that accelerates scientific discovery by creating a novel approach to literature review and claim verification.

**Timeline:** 1.5 days remaining
**Tech Stack:** Next.js + React (frontend), Python (backend agents), API-based LLMs
**Hardware Available:** HPC cluster with 2x NVIDIA RTX Pro A6000 GPUs (96GB VRAM each) — **can fine-tune models**
**Team Focus Split:** One founder focuses on the deep research handoff/integration, the rest on the core claim extraction and knowledge graph system.

---

## 🔥 CORE PHILOSOPHY — THIS IS WHAT MAKES US DIFFERENT

> **"We don't summarize. We VERIFY. We don't trust. We PROVE."**

### The Problem With Every Other Tool

Every existing "research AI" (ChatGPT, Perplexity, Edison, Elicit, etc.) does the same thing:
1. Find papers
2. **Summarize** them
3. Present the summary as truth

**This is fundamentally broken.** They are glorified summarizers. They:
- Trust papers at face value
- Don't verify if citations actually support claims
- Don't detect AI-generated garbage papers
- Don't trace claims back to ground truth
- Can't distinguish between solid science and bullshit

### Our Revolutionary Approach: GROUND TRUTH VERIFICATION

We are building **the first AI research system that actually researches like a human scientist:**

| What Others Do | What WE Do |
|----------------|------------|
| Summarize papers | **Verify every claim** |
| Trust citations blindly | **Check if citation actually supports the claim** |
| Present information | **Trace claims to ground truth** |
| Accept papers at face value | **Detect AI-generated/bullshit papers** |
| Static knowledge retrieval | **Dynamic claim provenance graph** |

### The Three Pillars

#### 1. 🎯 GROUND TRUTH, NOT SUMMARIES
- We don't summarize papers — we **extract claims** and **trace their evidence chain**
- Every claim is connected to its proof: citation, experiment, figure, or marked as UNSUPPORTED
- We follow the chain: Claim → Citation → Source Paper → Their Claims → Their Citations → **Until we hit ground truth or find a hole**

#### 2. 🛡️ ANTI-HALLUCINATION LAYER
- We are the **bullshit detector** for scientific literature
- Detect AI-generated papers (specific linguistic patterns, impossible citations)
- Detect papers with:
  - Fake citations (paper doesn't exist)
  - Irrelevant citations (paper exists but doesn't support the claim)
  - Circular citations (A cites B cites A)
  - Retracted sources
- **RED NODES** in our graph = unverified/suspicious claims

#### 3. 🔬 HUMAN-LIKE RESEARCH PROCESS
- A human researcher doesn't just read a paper — they:
  - Question claims
  - Check sources
  - Follow citation chains
  - Cross-reference findings
  - Build mental models of how evidence connects
- **We automate exactly this process** — not summarization, but VERIFICATION

### Why This Is "Shooting For The Stars"

Other tools make research **faster**. We make research **TRUTHFUL**.

- In an era of AI-generated garbage flooding arXiv and journals
- In an era where even peer review is failing
- In an era where LLMs hallucinate citations

**We are building the TRUST LAYER for scientific knowledge.**

This isn't just a hackathon project. This is the foundation for how AI should interact with scientific literature — with verification, not blind trust.

---

## 🔍 COMPETITIVE ANALYSIS — What Existing Tools LACK

The hackathon provides several AI research tools. **NONE of them verify claims:**

| Tool | What It Does | Does It Verify Claims? | Does It Detect Bullshit? |
|------|--------------|------------------------|--------------------------|
| **Denario** | Maker-Hater debate + Semantic Scholar novelty check | ❌ No | ❌ No |
| **AstroAgents** | Multi-agent with Data Analyst → Planner → Researchers → Critic | ❌ No | ❌ No |
| **Sakana AI Scientist** | Full automation: idea → code → experiment → paper | ❌ No | ❌ No |
| **DATAGEN** | Dataset analysis + hypothesis generation | ❌ No | ❌ No |
| **OpenAI Hypothesis** | Simple LangChain hypothesis generator | ❌ No | ❌ No |
| **NCBI Agent** | Biomedical search + RAG | ❌ No | ❌ No |
| **Syxplain** | Symbolic regression + literature validation | Partial (formula validation) | ❌ No |

### The Gap We Fill

All these tools:
- ✅ Generate hypotheses
- ✅ Search literature
- ✅ Summarize papers
- ❌ **DON'T verify if claims in papers are actually supported**
- ❌ **DON'T trace evidence chains to ground truth**
- ❌ **DON'T detect AI-generated/fraudulent papers**
- ❌ **DON'T build trust graphs showing claim provenance**

**We are the missing TRUST LAYER that should come BEFORE hypothesis generation.**

### Why This Matters for the Hackathon

The judges said: **"We prefer radical innovation over feasibility."**

- Existing tools = incremental improvements to summarization
- **Our tool = fundamentally new paradigm (verification over summarization)**

### Reference Papers from Hackathon

- [HypoBench (arXiv:2504.11524)](https://arxiv.org/abs/2504.11524) — Shows current LLMs only recover 38.8% of ground-truth hypotheses. **Verification is needed!**
- [The AI Scientist (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — Full automation but no claim verification
- [Denario (arXiv:2510.26887)](https://arxiv.org/abs/2510.26887) — Novelty checking via Semantic Scholar, but doesn't verify paper claims
- [SR-Scientist (arXiv:2510.11661)](https://arxiv.org/abs/2510.11661) — Code-based scientific discovery, no literature verification

---

## Hackathon Evaluation Criteria (Must Address All)

The judges will evaluate our solution against these criteria:

1. **10x Speed Up:** Does it turn weeks of literature review into hours?
2. **Enable Impossible Science:** Can researchers explore questions too complex for humans alone?
3. **Scale Beyond Traditional Methods:** Does it improve research standards and data quality automatically?
4. **Beyond Static Data Gathering:** Does it enable dynamic modeling, not just document retrieval?
5. **Automate Lower-Level Tasks:** Does it free scientists for critical analysis and decision-making?

---

## Our Solution: ClaimGraph

### Core Concept

We are building a **claim-provenance knowledge graph system** that:

1. **Ingests scientific papers** from multiple sources
2. **Extracts claims** made in those papers (e.g., "X causes Y", "Our findings show Z")
3. **Traces the evidence** supporting each claim (citations, experimental data, figures, other claims, or general knowledge)
4. **Validates claim integrity** by checking if citations exist, are accessible, and actually support the claim
5. **Builds a visual knowledge graph** showing papers, claims, evidence chains, and trust scores
6. **Detects "bullshit" papers** — papers with unsupported claims, circular citations, or AI-generated hallucinations (marked as RED nodes)
7. **Hands off to deep research** — wraps all findings (claims, evidence, graph structure) into a structured prompt for Gemini/Edison AI/Claude for deeper synthesis

### User Flow

```
User enters research query
        ↓
Clarifying Agent asks follow-up questions (if query is vague)
        ↓
Paper Search Agent finds relevant papers (Semantic Scholar, arXiv, PubMed)
        ↓
Papers stored in database
        ↓
Claim Extraction Agent processes each paper:
  - Extracts all claims
  - For each claim, identifies:
    - Type of evidence (citation, figure, experiment, other claim, general knowledge)
    - Source of evidence
    - Confidence score
        ↓
Citation Validation Agent:
  - Fetches cited papers
  - Recursively extracts claims if needed
  - Checks if citation actually supports the claim
  - Flags unsupported/circular/missing citations
        ↓
Knowledge Graph Builder:
  - Creates nodes (Papers, Claims, Evidence)
  - Creates edges (supports, cites, contradicts)
  - Assigns trust scores
  - Marks RED nodes for unverified/suspicious claims
        ↓
Graph displayed to user with initial findings summary
        ↓
User can:
  - Explore graph interactively
  - Continue chatting
  - Click "Deep Research" button
        ↓
Deep Research Handoff:
  - Serializes entire graph + claims + user questions
  - Sends to Gemini 2.5 Pro / Edison AI / Claude
  - Returns comprehensive research synthesis
```

---

## Research Questions I Need Answered

### 1. Paper Source APIs

I need to fetch scientific papers programmatically. Research the following:

- **Semantic Scholar API**
  - Rate limits, authentication requirements
  - Can we get full-text PDFs or just abstracts?
  - How to search by topic/keyword?
  - What metadata is returned (citations, authors, publication date)?

- **arXiv API**
  - How to query papers?
  - Can we download PDFs directly?
  - Rate limits?

- **PubMed / PMC API**
  - Access to biomedical literature
  - Full-text availability?
  - API structure?

- **OpenAlex** (alternative to Semantic Scholar)
  - Is it better for our use case?
  - What are the trade-offs?

**Key Question:** Which API gives us the best combination of: (a) full-text access, (b) citation graph data, (c) easy integration, (d) generous rate limits for a hackathon demo?

---

### 2. PDF Processing & Text Extraction

Papers come as PDFs. How do we extract structured text?

- **PDF parsing libraries:**
  - PyMuPDF (fitz)
  - pdfplumber
  - PyPDF2
  - Marker (ML-based PDF to Markdown)
  - Nougat (Meta's scientific PDF parser)

- **Considerations:**
  - Tables and figures — how to handle?
  - Mathematical equations?
  - Two-column layouts?
  - References section extraction?

**Key Question:** What's the fastest, most accurate approach for extracting clean text from scientific PDFs in 2 days?

---

### 3. Claim Extraction Architecture

How should we prompt an LLM to extract claims and their evidence?

- **Claim Types to Identify:**
  - Causal claims ("X causes Y")
  - Correlational claims ("X is associated with Y")
  - Methodological claims ("Our method achieves X")
  - Comparative claims ("X outperforms Y")
  - Existence claims ("We discovered X")

- **Evidence Types for Each Claim:**
  - `CITATION` — references another paper
  - `FIGURE` — refers to a figure/table in the paper
  - `EXPERIMENT` — describes experimental results
  - `OTHER_CLAIM` — builds on another claim in the same paper
  - `GENERAL_KNOWLEDGE` — commonly accepted fact, no citation needed
  - `UNSUPPORTED` — no evidence provided

**Key Question:** What's the optimal prompt structure for claim extraction? Should we use structured output (JSON mode)? What schema?

---

### 4. Knowledge Graph Storage & Representation

How do we store and query the knowledge graph?

**Options to Research:**

- **Neo4j**
  - Pros: Industry standard, powerful queries (Cypher), good visualization
  - Cons: Needs server setup, might be overkill for hackathon

- **NetworkX (Python in-memory)**
  - Pros: Simple, no setup, easy to serialize to JSON
  - Cons: Not persistent, no built-in visualization

- **Graph databases for JavaScript/TypeScript:**
  - Dgraph
  - ArangoDB
  - Or just use PostgreSQL with JSON columns?

- **Frontend Graph Visualization:**
  - Cytoscape.js
  - D3.js force-directed graph
  - vis.js Network
  - React Flow
  - Sigma.js

**Key Question:** For a 2-day hackathon with Next.js frontend, what's the fastest path to a working, interactive knowledge graph?

**Proposed Schema (validate this):**

```typescript
// Node Types
interface PaperNode {
  id: string;
  type: 'PAPER';
  title: string;
  authors: string[];
  year: number;
  doi?: string;
  abstract: string;
  source: 'arxiv' | 'semantic_scholar' | 'pubmed' | 'upload';
  trustScore: number; // 0-1
  flagged: boolean; // RED node if true
}

interface ClaimNode {
  id: string;
  type: 'CLAIM';
  text: string;
  claimType: 'causal' | 'correlational' | 'methodological' | 'comparative' | 'existence';
  confidence: number;
  paperId: string; // which paper this claim is from
  verified: boolean;
  flagged: boolean;
}

interface EvidenceNode {
  id: string;
  type: 'EVIDENCE';
  evidenceType: 'citation' | 'figure' | 'experiment' | 'other_claim' | 'general_knowledge' | 'unsupported';
  description: string;
  sourceRef?: string; // DOI or figure number
  valid: boolean; // did validation pass?
}

// Edge Types
type Edge = 
  | { type: 'CONTAINS_CLAIM'; from: PaperNode; to: ClaimNode }
  | { type: 'SUPPORTED_BY'; from: ClaimNode; to: EvidenceNode }
  | { type: 'CITES'; from: PaperNode; to: PaperNode }
  | { type: 'CONTRADICTS'; from: ClaimNode; to: ClaimNode }
  | { type: 'BUILDS_ON'; from: ClaimNode; to: ClaimNode };
```

---

### 5. Citation Validation Logic

How do we detect "bullshit" papers? What validation checks should we run?

**Validation Checks to Research:**

1. **Citation Existence Check**
   - Does the cited paper actually exist?
   - Can we fetch it from Semantic Scholar/arXiv?

2. **Citation Relevance Check**
   - Does the cited paper actually discuss what the claim says?
   - Use embedding similarity or LLM verification

3. **Circular Citation Detection**
   - Does paper A cite paper B which cites paper A for the same claim?

4. **Retraction Check**
   - Has the cited paper been retracted?
   - Retraction Watch database?

5. **AI-Generated Content Detection**
   - Signs of LLM-generated papers (specific phrases, patterns)
   - Known problematic publishers

**Key Question:** What validation checks are feasible in 2 days? What's the minimum viable "bullshit detector"?

---

### 6. Multi-Agent Architecture

How should we structure the agents?

**Proposed Agents:**

1. **Clarifying Agent** — Asks follow-up questions if query is vague
2. **Paper Search Agent** — Searches APIs, ranks results, fetches papers
3. **Claim Extraction Agent** — Processes papers, extracts claims + evidence
4. **Validation Agent** — Checks citations, flags suspicious claims
5. **Graph Builder Agent** — Constructs the knowledge graph
6. **Orchestrator Agent** — Coordinates all agents, manages state

**Key Question:** Should we use an agent framework (LangGraph, CrewAI, AutoGen) or build custom? What's fastest for a hackathon?

---

### 7. LLM API Choices

Which LLMs should we use for which tasks?

**Tasks:**
- Clarifying questions → Fast, cheap model (GPT-4o-mini, Claude Haiku)
- Claim extraction → Accurate, structured output (GPT-4o, Claude Sonnet, Gemini Flash)
- Validation/synthesis → High reasoning (Claude Sonnet, GPT-4o)
- Deep research handoff → Best available (Gemini 2.5 Pro, Claude Opus, GPT-4)

**Key Question:** What's the optimal model routing to balance cost, speed, and accuracy?

---

### 8. Deep Research Handoff (Founder's Focus)

How do we structure the handoff to external deep research systems?

**What needs to be serialized:**
- All extracted claims with evidence chains
- Knowledge graph structure (nodes + edges)
- Trust scores and flags
- User's original query and follow-up context
- Specific questions to answer

**Target Systems:**
- Gemini 2.5 Pro with large context window
- Edison AI (if API available)
- Claude with extended thinking
- Perplexity API

**Key Question:** What's the optimal prompt structure for deep research handoff? How do we serialize a knowledge graph for an LLM context window?

---

### 9. Database Schema (Persistent Storage)

Besides the graph, we need persistent storage for:
- User sessions
- Paper cache (avoid re-fetching)
- Extracted claims cache
- Research history

**Options:**
- Supabase (PostgreSQL + Auth + Realtime)
- PlanetScale (MySQL)
- SQLite (local, simple)
- MongoDB (document store)

**Key Question:** For Next.js + fast hackathon development, what's the best database choice?

---

### 10. HPC Cluster Usage — FINE-TUNING OPPORTUNITY

We have 2x NVIDIA RTX Pro A6000 GPUs (96GB VRAM each = 192GB total). This is **significant compute**. Research how to leverage this.

**High-Priority Use Cases:**

#### A. Fine-Tune a Claim Extraction Model
- **Base Model Options:**
  - Llama 3.1 8B or 70B (if quantized)
  - Mistral 7B / Mixtral 8x7B
  - Qwen 2.5 7B/14B/32B
  - Phi-3 or Phi-4

- **Training Data:**
  - Can we generate synthetic claim-extraction data using GPT-4?
  - Are there existing datasets of scientific claims?
  - Few-shot fine-tuning with LoRA/QLoRA

- **Key Question:** Can we fine-tune a claim extraction model in 6-12 hours that outperforms prompting alone?

#### B. Fine-Tune a Bullshit Detector Model
- **Task:** Classify papers as legitimate vs AI-generated/suspicious
- **Training signals:**
  - Known retracted papers (Retraction Watch)
  - Known AI-generated papers (papermills)
  - Synthetic negative examples (generate fake papers with LLM)
- **Key Question:** What features distinguish AI-generated scientific papers? Can we train a classifier?

#### C. Run Local Inference (Cost Savings)
- Run Llama 3.1 70B or Qwen 2.5 72B locally
- Avoid API costs for bulk claim extraction
- 192GB VRAM can run large models without quantization

#### D. Embedding Model for Semantic Search
- Run BGE-large, E5-large, or GTE for paper similarity
- Use for "does this citation actually support this claim?" validation
- Fast local inference for semantic matching

#### E. PDF Processing with Nougat/Marker
- Meta's Nougat model for PDF → structured text
- Requires GPU for fast processing
- Better quality than rule-based PDF extraction

**Research Questions for GPU Usage:**
1. What's the fastest way to set up fine-tuning with LoRA on A6000s?
2. What base model gives best claim extraction with fine-tuning?
3. Is there an existing AI-generated text detector we can fine-tune for scientific papers?
4. What's the ROI: time spent fine-tuning vs. improvement over prompting?

---

### 11. AI-Generated Paper Detection — CRITICAL FEATURE

This is a **key differentiator**. Research specifically:

**Known Signals of AI-Generated Papers:**
- Overuse of certain phrases ("delve", "crucial", "it's important to note")
- Impossible citations (papers that don't exist, wrong DOIs)
- Authors that don't exist (fake researcher profiles)
- Impossible publication timelines
- Statistical anomalies in data
- Repetitive sentence structures
- Lack of specific experimental details

**Detection Approaches:**
1. **Linguistic Analysis**
   - Detect GPT/Claude signature phrases
   - Analyze sentence length distribution
   - Perplexity scoring (low perplexity = potentially AI-generated)

2. **Citation Verification**
   - Check if all cited papers exist in Semantic Scholar
   - Verify DOIs are valid
   - Check if citation context makes sense

3. **Metadata Analysis**
   - Cross-reference author names with ORCID
   - Check publication venue reputation
   - Look for papermill patterns

4. **Existing Tools/Models to Research:**
   - GPTZero for academic papers
   - Originality.AI
   - ZeroGPT
   - Custom fine-tuned models

**Key Question:** What's the minimum viable bullshit detector we can build in 1.5 days? What existing tools can we integrate?

---

## What I Need From This Research

Please provide:

1. **Recommended Tech Stack** — Specific libraries, APIs, databases for our constraints
2. **Database/Graph Schema** — Finalized schema for storing papers, claims, evidence, edges
3. **Agent Architecture** — Which framework to use (or custom), how agents communicate
4. **Claim Extraction Prompt** — Ready-to-use prompt for extracting claims from paper text
5. **Validation Pipeline** — Which checks to implement, in what order
6. **Graph Visualization** — Best library for React/Next.js interactive graph
7. **Deep Research Prompt Template** — How to serialize graph for LLM handoff
8. **MVP Scope** — What's realistic in 2 days, what to cut

---

## Success Criteria

Our demo should show:
1. User enters a research topic
2. System finds and processes 5-10 papers
3. **Claims are extracted with evidence chains — NOT summaries**
4. Knowledge graph is displayed with claims, evidence, and trust indicators
5. **RED nodes clearly visible for suspicious/unsupported claims**
6. **At least one "bullshit" paper detected** (fake citation, AI-generated, unsupported claims)
7. User can click "Deep Research" to get synthesis based on VERIFIED claims
8. Live demo with real papers on a scientific topic

---

## 🎯 PITCH SUMMARY (For Mentors & Judges)

### The Problem
Every AI research tool today (ChatGPT, Perplexity, Elicit, Edison) does the same thing: **find papers and summarize them**. They trust everything at face value. In an era of:
- AI-generated garbage papers flooding arXiv
- Papermills producing fake research
- LLMs hallucinating citations
- Even peer review failing to catch fraud

**Summarization is NOT research. It's just faster reading.**

### Our Solution: Ground Truth Verification
We built **ClaimGraph** — the first AI research system that actually researches like a human scientist:

1. **Extract Claims** — Not summaries. Specific claims the paper makes.
2. **Trace Evidence** — For each claim, what's the proof? Citation? Experiment? Figure? Nothing?
3. **Verify Citations** — Does the cited paper exist? Does it actually support the claim?
4. **Build Trust Graph** — Visual knowledge graph with GREEN (verified) and RED (suspicious) nodes
5. **Catch Bullshit** — Detect AI-generated papers, fake citations, circular references

### Why This Matters
- Other tools make research **faster**
- We make research **TRUTHFUL**
- We are building the **trust layer** for scientific knowledge

### Technical Differentiators
- **192GB GPU cluster** for fine-tuned claim extraction and bullshit detection
- **Knowledge graph** with claim provenance, not flat document retrieval
- **Multi-agent architecture** with specialized agents for each verification step
- **Deep research handoff** — our verified claims feed into Gemini/Claude for synthesis

---

## Project Name Options (Research Name Ideas Too)

- ClaimGraph
- TrustGraph
- ProvenanceAI
- ClaimTrace
- ScienceGraph
- VerifyAI
- GraphScholar
- **GroundTruth** ← Strong name
- **VerifyScience**
- **ClaimCheck**

---

*Use this document as the prompt for deep research. The goal is to get actionable, specific recommendations that we can implement in 1.5 days.*
