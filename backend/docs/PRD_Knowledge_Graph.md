# PRD: ClaimGraph Scientific Knowledge Graph

## Source of Truth for Implementation

**Version**: 1.0  
**Created**: January 2026  
**Status**: Approved for Implementation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Core Concept](#2-core-concept)
3. [Data Model](#3-data-model)
4. [Graph Building Pipeline](#4-graph-building-pipeline)
5. [Storage Architecture](#5-storage-architecture)
6. [Claim Extraction](#6-claim-extraction)
7. [Ground Truth Classification](#7-ground-truth-classification)
8. [Citation Expansion Strategy](#8-citation-expansion-strategy)
9. [MCP Tools Specification](#9-mcp-tools-specification)
10. [Database Schema](#10-database-schema)
11. [API Endpoints](#11-api-endpoints)
12. [Anchor Papers](#12-anchor-papers)
13. [Implementation Phases](#13-implementation-phases)
14. [Appendix](#14-appendix)

---

## 1. Executive Summary

### What We're Building

A **pre-built, persistent Scientific Knowledge Graph** where:
- **Claims** are the only first-class nodes
- **Papers** exist only as metadata on claims
- Every claim traces back to either a **ground truth** or is marked **unsupported**
- The graph enables the Deep Research Agent to ground answers in verified scientific claims

### Key Design Decisions

| Aspect | Decision |
|--------|----------|
| Node type | Claims only (papers as metadata) |
| Edge types | `supports`, `contradicts` |
| Extraction LLM | GPT-4o-mini (batch processing) |
| Pipeline | Batch job with parallel expansion |
| Deduplication | Merge with combined metadata |
| Storage | Supabase (Postgres) + NetworkX cache |
| Vector storage | pgvector in Supabase |
| Embeddings | text-embedding-3-small |
| Confidence scoring | LLM judgment (0.0-1.0) |
| Seed domain | Quantitative Biology (arXiv q-bio) |
| Expansion strategy | Hybrid: anchor papers + citation traversal |

### The Core Innovation

**Provenance Tracing**: Every claim in the graph can be traced backward through supporting claims until it reaches either:
- A **ground truth** (axiomatic fact) → Chain is grounded
- An **unsupported claim** → Chain has a weak foundation
- A **cycle** → Circular reasoning detected

This allows the Deep Research Agent to not just answer questions, but to show the epistemic foundation of those answers.

---

## 2. Core Concept

### The Problem with Current LLM-Based Research

1. LLMs hallucinate citations and claims
2. No way to verify if a claim is actually supported
3. "Trust me" answers without traceable evidence chains
4. No visibility into conflicting evidence

### Our Solution: Claims-Only Knowledge Graph

```
Traditional Approach:
  Query → LLM → "Here's what I think" (unverifiable)

ClaimGraph Approach:
  Query → Search Graph → Traverse Support Chains → "Here's what's proven, and here's the evidence trail"
```

### Query-Time Flow

```
User Query: "Does gut microbiome affect depression?"
    │
    ▼
┌─────────────────────────────────────────┐
│  1. SEMANTIC SEARCH                     │
│     Find relevant claim nodes           │
│     using pgvector similarity           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  2. GRAPH TRAVERSAL                     │
│     Follow support/contradict edges     │
│     Traverse paper metadata (same paper)│
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  3. PROVENANCE CHECK                    │
│     Trace chains to ground truths       │
│     Identify unsupported foundations    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│  4. DEEP RESEARCH SYNTHESIS             │
│     o3-deep-research synthesizes        │
│     with [claim_id] citations           │
└─────────────────────────────────────────┘
    │
    ▼
Response: Grounded answer with full evidence chain
```

---

## 3. Data Model

### 3.1 Claim Node Schema

```typescript
interface ClaimNode {
  // Identity
  id: string;                    // Format: "claim_{hash}" where hash = first 12 chars of SHA256(text)
  
  // Core content
  text: string;                  // The claim itself, rephrased for clarity
  original_text: string;         // Original text from paper (for reference)
  
  // Classification
  type: "empirical" | "ground_truth" | "unsupported";
  confidence: number;            // 0.0-1.0, LLM-judged confidence in the claim
  
  // Source paper metadata (NOT a separate node)
  source_paper: {
    arxiv_id: string;            // e.g., "2301.12345"
    title: string;
    authors: string[];
    year: number;
    venue?: string;              // Journal/conference if known
    section?: string;            // Which section the claim came from
    url: string;                 // Full arXiv URL
  };
  
  // For non-arXiv sources (citation expansion)
  external_source?: {
    url: string;
    retrieval_method: "semantic_scholar" | "web_search" | "manual";
    retrieval_date: string;
    verified: boolean;
  };
  
  // Extraction metadata
  extraction: {
    model: string;               // e.g., "gpt-4o-mini"
    timestamp: string;           // ISO 8601
    prompt_version: string;      // For reproducibility
    context_snippet: string;     // Surrounding text for context
  };
  
  // Vector embedding (stored in pgvector column)
  embedding?: number[];          // 1536-dim for text-embedding-3-small
  
  // Graph metadata (computed)
  support_count: number;         // How many claims support this one
  contradict_count: number;      // How many claims contradict this one
  depth_to_ground_truth: number | null;  // Shortest path to a ground truth, null if ungrounded
}
```

### 3.2 Edge Schema

```typescript
interface ClaimEdge {
  id: string;                    // Format: "edge_{source_id}_{target_id}_{type}"
  source_id: string;             // The claim providing evidence
  target_id: string;             // The claim being supported/contradicted
  
  type: "supports" | "contradicts";
  
  weight: number;                // 0.0-1.0, strength of the relationship
  
  // Justification
  reasoning: string;             // LLM explanation for why this edge exists
  
  // Metadata
  created_at: string;
  model: string;                 // Which LLM created this edge
}
```

### 3.3 Edge Semantics

| Edge Type | Direction | Meaning |
|-----------|-----------|---------|
| `supports` | A → B | Claim A provides evidence that Claim B is true |
| `contradicts` | A → B | Claim A provides evidence that Claim B is false |

**Important**: Edges are directional. "A supports B" does NOT imply "B supports A".

### 3.4 Claim Types

| Type | Definition | Example |
|------|------------|---------|
| `ground_truth` | Axiomatic fact requiring no further justification | "2 + 2 = 4", "DNA encodes genetic information" |
| `empirical` | Claim based on experimental evidence | "Treatment X reduced symptoms by 40% (p<0.01)" |
| `unsupported` | Claim with no supporting evidence in graph | "As everyone knows, X is true" |

---

## 4. Graph Building Pipeline

### 4.1 Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    GRAPH BUILDING PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 1: SEED (Sequential)                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 1. Load anchor papers (10-20 foundational q-bio papers)     ││
│  │ 2. For each anchor paper:                                   ││
│  │    a. Fetch full text from arXiv                            ││
│  │    b. Extract claims (GPT-4o-mini)                          ││
│  │    c. Classify each claim (ground_truth/empirical/unsup)    ││
│  │    d. Generate embeddings                                   ││
│  │    e. Store in Supabase                                     ││
│  │ 3. Identify inter-claim relationships within each paper     ││
│  │ 4. Create edges between claims                              ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  PHASE 2: EXPAND (Parallel)                                     │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 1. Build citation queue from anchor papers                  ││
│  │ 2. Filter: keep only citations with >30% keyword overlap    ││
│  │ 3. Parallel workers (N=5) process citation queue:           ││
│  │    ┌─────────────────────────────────────────────────────┐  ││
│  │    │ Worker:                                             │  ││
│  │    │ a. Check if arxiv → fetch from arXiv                │  ││
│  │    │ b. If not arxiv → check Semantic Scholar            │  ││
│  │    │ c. If critical & not found → web search fallback    │  ││
│  │    │ d. Extract claims                                   │  ││
│  │    │ e. Find edges to existing claims                    │  ││
│  │    │ f. Add new citations to queue (depth < 3)           │  ││
│  │    └─────────────────────────────────────────────────────┘  ││
│  │ 4. Stop when: depth=3 OR 500 claims OR queue empty          ││
│  └─────────────────────────────────────────────────────────────┘│
│                              │                                   │
│                              ▼                                   │
│  PHASE 3: ANALYZE (Sequential)                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 1. Compute depth_to_ground_truth for all claims             ││
│  │ 2. Mark claims with no path as "ungrounded"                 ││
│  │ 3. Detect cycles (circular reasoning)                       ││
│  │ 4. Generate graph statistics                                ││
│  │ 5. Export NetworkX cache                                    ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Parallelization Strategy

```python
# Expansion phase parallel processing
async def expand_citations_parallel(citation_queue: list, max_workers: int = 5):
    """
    Process citation queue with parallel workers.
    
    Each worker:
    1. Takes a citation from queue
    2. Processes it (fetch, extract, edge-find)
    3. Adds new citations to queue if depth < MAX_DEPTH
    
    Queue is thread-safe, workers coordinate via asyncio.
    """
    semaphore = asyncio.Semaphore(max_workers)
    
    async def process_citation(citation):
        async with semaphore:
            # Process single citation
            claims = await extract_claims_from_paper(citation)
            edges = await find_edges_to_existing_claims(claims)
            new_citations = await get_citations_from_paper(citation)
            return claims, edges, new_citations
    
    # Process all citations in parallel batches
    while citation_queue:
        batch = [citation_queue.pop() for _ in range(min(max_workers, len(citation_queue)))]
        results = await asyncio.gather(*[process_citation(c) for c in batch])
        
        for claims, edges, new_citations in results:
            # Store results
            await store_claims(claims)
            await store_edges(edges)
            
            # Add new citations if depth allows
            for nc in new_citations:
                if nc.depth < MAX_DEPTH:
                    citation_queue.append(nc)
```

### 4.3 Stopping Conditions

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Depth limit | 3 hops from anchor | Stop expanding this branch |
| Claim budget | 500 claims | Stop all expansion |
| Queue empty | 0 remaining | Pipeline complete |
| Relevance filter | <30% keyword overlap | Skip this citation |
| Rate limit | API throttled | Exponential backoff |

---

## 5. Storage Architecture

### 5.1 Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                           │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │  Deep Research   │  │   Graph Builder  │  │   MCP Server   │ │
│  │     Agent        │  │     Pipeline     │  │                │ │
│  └────────┬─────────┘  └────────┬─────────┘  └───────┬────────┘ │
│           │                     │                     │          │
└───────────┼─────────────────────┼─────────────────────┼──────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       CACHE LAYER (In-Memory)                    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    NetworkX Graph                            ││
│  │                                                              ││
│  │  - Full graph loaded on startup                              ││
│  │  - All traversal operations happen here                      ││
│  │  - Sub-millisecond query response                            ││
│  │  - Refreshed on write operations                             ││
│  │                                                              ││
│  │  Methods:                                                    ││
│  │  - nx.shortest_path(G, source, target)                       ││
│  │  - nx.ancestors(G, node)                                     ││
│  │  - nx.descendants(G, node)                                   ││
│  │  - G.neighbors(node)                                         ││
│  │  - G[node1][node2]['weight']                                 ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
└───────────────────────────────────────────────────────────────────┘
            │
            │ Write-through / Load on startup
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PERSISTENCE LAYER (Supabase)                  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  PostgreSQL + pgvector                                   │   │
│  │                                                          │   │
│  │  Tables:                                                 │   │
│  │  ├── claims (id, text, type, metadata, embedding)        │   │
│  │  ├── edges (source_id, target_id, type, weight)          │   │
│  │  └── build_runs (metadata about pipeline runs)           │   │
│  │                                                          │   │
│  │  Indexes:                                                │   │
│  │  ├── claims_embedding_idx (ivfflat for similarity)       │   │
│  │  ├── edges_source_idx                                    │   │
│  │  └── edges_target_idx                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Cache Synchronization

```python
class GraphCache:
    """
    In-memory NetworkX graph that syncs with Supabase.
    """
    def __init__(self, supabase_client):
        self.G = nx.DiGraph()
        self.supabase = supabase_client
        self._loaded = False
    
    async def load_from_db(self):
        """Load entire graph from Supabase into NetworkX."""
        # Load all claims as nodes
        claims = await self.supabase.table("claims").select("*").execute()
        for claim in claims.data:
            self.G.add_node(
                claim["id"],
                text=claim["text"],
                type=claim["type"],
                confidence=claim["confidence"],
                metadata=claim["metadata"]
            )
        
        # Load all edges
        edges = await self.supabase.table("edges").select("*").execute()
        for edge in edges.data:
            self.G.add_edge(
                edge["source_id"],
                edge["target_id"],
                type=edge["type"],
                weight=edge["weight"],
                reasoning=edge["reasoning"]
            )
        
        self._loaded = True
    
    async def add_claim(self, claim: ClaimNode):
        """Write-through: add to both cache and DB."""
        # Add to DB first
        await self.supabase.table("claims").insert(claim.dict()).execute()
        
        # Then update cache
        self.G.add_node(claim.id, **claim.dict())
    
    async def add_edge(self, edge: ClaimEdge):
        """Write-through: add to both cache and DB."""
        await self.supabase.table("edges").insert(edge.dict()).execute()
        self.G.add_edge(edge.source_id, edge.target_id, **edge.dict())
```

---

## 6. Claim Extraction

### 6.1 Extraction Pipeline

```
Paper Full Text
      │
      ▼
┌─────────────────────────────────────────┐
│  STEP 1: Section Splitting              │
│  - Abstract                             │
│  - Introduction                         │
│  - Methods                              │
│  - Results                              │
│  - Discussion                           │
│  - Conclusion                           │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  STEP 2: Claim Identification           │
│  (GPT-4o-mini per section)              │
│                                         │
│  Input: Section text                    │
│  Output: List of claim candidates       │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  STEP 3: Claim Rephrasing               │
│  (GPT-4o-mini)                          │
│                                         │
│  - Compress verbose statements          │
│  - Standardize terminology              │
│  - Remove hedging language              │
│  - Keep original for reference          │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  STEP 4: Claim Classification           │
│  (GPT-4o-mini)                          │
│                                         │
│  Classify as:                           │
│  - ground_truth                         │
│  - empirical                            │
│  - unsupported                          │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  STEP 5: Edge Detection                 │
│  (GPT-4o-mini)                          │
│                                         │
│  For each claim pair in paper:          │
│  - Does A support B?                    │
│  - Does A contradict B?                 │
│  - No relationship?                     │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│  STEP 6: Embedding Generation           │
│  (text-embedding-3-small)               │
│                                         │
│  Generate 1536-dim vector for each      │
│  claim's rephrased text                 │
└─────────────────────────────────────────┘
```

### 6.2 Extraction Prompt (GPT-4o-mini)

```markdown
# SYSTEM PROMPT: Scientific Claim Extractor

You are extracting scientific claims from academic papers. Your job is to identify distinct, verifiable claims.

## What IS a claim:
- A specific assertion that can be true or false
- A statement of fact, finding, or conclusion
- A quantitative result (e.g., "X increased by 40%")
- A causal relationship (e.g., "A causes B")
- A methodological assertion (e.g., "Method X is more accurate than Y")

## What is NOT a claim:
- Background context or definitions
- Descriptions of what the paper will do
- Acknowledgments or administrative text
- Vague statements without specific content
- Questions or hypotheses (unless stated as conclusions)

## Output Format

For each claim found, output JSON:

```json
{
  "claims": [
    {
      "original_text": "The exact text from the paper",
      "rephrased": "Clear, concise version of the claim",
      "section": "Which section this came from",
      "importance": "high" | "medium" | "low",
      "claim_type": "empirical" | "methodological" | "causal" | "comparative"
    }
  ]
}
```

## Guidelines:
- Extract 5-15 claims per paper (focus on important ones)
- Rephrase to remove hedging ("may", "might", "could")
- Keep quantitative details (numbers, p-values)
- If a claim references another paper, note the citation
```

### 6.3 Classification Prompt (GPT-4o-mini)

```markdown
# SYSTEM PROMPT: Claim Classifier

You are classifying scientific claims into three categories.

## Categories:

### ground_truth
A claim that requires no further justification because it is:
- A mathematical fact (e.g., "2 + 2 = 4")
- A definition (e.g., "DNA is deoxyribonucleic acid")
- An established scientific law (e.g., "Energy is conserved")
- Basic logic (e.g., "If A implies B and B implies C, then A implies C")

**BE VERY STRICT**: Only mark as ground_truth if NO reasonable scientist would dispute it.

### empirical
A claim based on evidence that could theoretically be tested:
- Experimental results
- Observational findings
- Statistical conclusions
- Comparative assessments

Most scientific claims are empirical.

### unsupported
A claim stated without evidence or justification:
- "As is well known..."
- "It is obvious that..."
- Appeals to authority without citation
- Assumptions stated as facts

## Output Format

```json
{
  "classification": "ground_truth" | "empirical" | "unsupported",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of classification"
}
```

## IMPORTANT:
- When in doubt, classify as "empirical" (not ground_truth)
- "Unsupported" is NOT a judgment of truth, just that no evidence is provided in context
- A claim can be true AND unsupported (if no justification is given)
```

### 6.4 Edge Detection Prompt (GPT-4o-mini)

```markdown
# SYSTEM PROMPT: Claim Relationship Detector

You are identifying relationships between scientific claims.

## Relationship Types:

### supports
Claim A provides evidence that Claim B is true.
- A is a specific instance that validates B's general statement
- A's data directly confirms B's assertion
- A's conclusion logically implies B

### contradicts
Claim A provides evidence that Claim B is false.
- A's data conflicts with B's assertion
- A's conclusion is incompatible with B
- A explicitly disputes B

### none
No clear evidential relationship between A and B.
- They discuss different topics
- They're both true but unrelated
- Insufficient information to determine relationship

## Output Format

For each pair of claims (A, B), output:

```json
{
  "source_claim_id": "A",
  "target_claim_id": "B",
  "relationship": "supports" | "contradicts" | "none",
  "weight": 0.0-1.0,
  "reasoning": "Why this relationship exists"
}
```

## Weight Guidelines:
- 0.9-1.0: Direct, explicit support/contradiction
- 0.7-0.9: Strong implied support/contradiction
- 0.5-0.7: Moderate, indirect relationship
- 0.3-0.5: Weak, tenuous relationship
- Below 0.3: Probably "none"

## IMPORTANT:
- Relationships are DIRECTIONAL: "A supports B" ≠ "B supports A"
- Only mark relationships where there's actual evidential connection
- When in doubt, mark as "none"
```

---

## 7. Ground Truth Classification

### 7.1 Strict Criteria

A claim is classified as `ground_truth` ONLY if it meets ALL of these criteria:

| Criterion | Test | Example Pass | Example Fail |
|-----------|------|--------------|--------------|
| **Universally accepted** | No reasonable scientist disputes it | "Cells contain DNA" | "Gut bacteria affect mood" |
| **Self-evident or definitional** | True by definition or basic logic | "A square has 4 sides" | "This method is best" |
| **Domain-independent** | True regardless of field | "1 + 1 = 2" | "Standard treatment is X" |
| **Timeless** | Won't become false with new evidence | "Water is H2O" | "Current evidence suggests..." |

### 7.2 Examples

**Ground Truths (ACCEPT)**:
- "Proteins are composed of amino acids"
- "The mitochondria produce ATP"
- "Statistical significance is typically p < 0.05"
- "DNA replication is semi-conservative"

**NOT Ground Truths (REJECT)**:
- "As we all know, X is important" → `unsupported`
- "It is widely accepted that..." → `empirical` (needs citation)
- "The standard approach is..." → `empirical`
- "Obviously, X implies Y" → `unsupported`

### 7.3 Classification Flow

```
Claim: "Gut bacteria produce neurotransmitters"
    │
    ▼
Is it a mathematical/logical fact? → NO
    │
    ▼
Is it a basic definition? → NO
    │
    ▼
Would any scientist dispute it? → YES (some might want evidence)
    │
    ▼
Classification: EMPIRICAL (not ground_truth)
```

---

## 8. Citation Expansion Strategy

### 8.1 Hybrid Approach: Anchor + Citation Traversal

```
PHASE 1: Manual Anchor Selection
┌─────────────────────────────────────────────────────────────────┐
│  Select 10-20 foundational papers in Quantitative Biology       │
│                                                                 │
│  Criteria:                                                      │
│  - High citation count (>100)                                   │
│  - Broad influence in the field                                 │
│  - Mix of topics within q-bio                                   │
│  - Available on arXiv                                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
PHASE 2: Citation Extraction
┌─────────────────────────────────────────────────────────────────┐
│  For each anchor paper:                                         │
│  1. Parse reference section                                     │
│  2. Extract all citations                                       │
│  3. Resolve to arXiv IDs where possible                         │
│  4. Use Semantic Scholar API for metadata                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
PHASE 3: Relevance Filtering
┌─────────────────────────────────────────────────────────────────┐
│  For each citation:                                             │
│  1. Get title + abstract                                        │
│  2. Extract keywords                                            │
│  3. Compare to anchor keyword set                               │
│  4. Keep if overlap > 30%                                       │
│  5. Skip if clearly different domain                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
PHASE 4: Parallel Processing
┌─────────────────────────────────────────────────────────────────┐
│  Queue: [filtered citations]                                    │
│  Workers: 5 parallel                                            │
│                                                                 │
│  Each worker:                                                   │
│  1. Pop citation from queue                                     │
│  2. Determine source:                                           │
│     - arXiv → fetch PDF, extract text                           │
│     - Semantic Scholar → get abstract only                      │
│     - Neither → web search (if critical)                        │
│  3. Extract claims                                              │
│  4. Find edges to existing claims                               │
│  5. Get this paper's citations                                  │
│  6. Add to queue (if depth < 3)                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Non-arXiv Citation Handling

```
Citation found that's not on arXiv
                │
                ▼
┌─────────────────────────────────────────┐
│  STEP 1: Check Semantic Scholar         │
│  - Get metadata (title, authors, year)  │
│  - Get abstract if available            │
│  - Get citation count                   │
└─────────────────────────────────────────┘
                │
                ▼
        Is abstract available?
           /           \
         YES            NO
          │              │
          ▼              ▼
┌──────────────┐  ┌──────────────────────┐
│ Extract      │  │ STEP 2: Web Search   │
│ claims from  │  │ - Search paper title │
│ abstract     │  │ - Find PDF or text   │
│ (limited)    │  │ - Extract if found   │
└──────────────┘  └──────────────────────┘
                           │
                           ▼
                    Found full text?
                      /        \
                    YES         NO
                     │           │
                     ▼           ▼
              ┌──────────┐  ┌──────────────────┐
              │ Extract  │  │ Mark as          │
              │ claims   │  │ "external_       │
              │ normally │  │ unverified"      │
              └──────────┘  └──────────────────┘
```

### 8.3 Critical Citation Threshold

A citation is "critical" and requires web search fallback if:
- It's cited by >= 3 claims in our graph
- It's the only support for a key claim
- It contradicts a well-supported claim

Non-critical citations can be marked as "external_unverified" without web search.

---

## 9. MCP Tools Specification

### 9.1 Overview

The Deep Research Agent interacts with the Knowledge Graph through MCP tools. These tools provide:
- **Search**: Find relevant claims
- **Traversal**: Navigate relationships
- **Analysis**: Assess claim validity

### 9.2 Tool: `search_claims`

```typescript
{
  name: "search_claims",
  description: "Semantic and keyword search for claims in the knowledge graph",
  parameters: {
    query: string,              // Natural language search query
    search_type: "semantic" | "keyword" | "hybrid",
    limit: number,              // Max results (default: 10)
    filters?: {
      type?: "empirical" | "ground_truth" | "unsupported",
      min_confidence?: number,
      source_paper_year?: { min?: number, max?: number },
      has_support?: boolean     // Only claims with supporting evidence
    }
  },
  returns: {
    claims: Array<{
      id: string,
      text: string,
      type: string,
      confidence: number,
      source_paper: object,
      similarity_score: number,  // For semantic search
      support_count: number,
      contradict_count: number
    }>
  }
}
```

**Implementation**:
```python
async def search_claims(query: str, search_type: str, limit: int, filters: dict):
    if search_type == "semantic":
        # Generate embedding for query
        embedding = await get_embedding(query)
        
        # pgvector similarity search
        results = await supabase.rpc(
            "search_claims_by_embedding",
            {"query_embedding": embedding, "match_count": limit}
        ).execute()
    
    elif search_type == "keyword":
        # Full-text search on claims table
        results = await supabase.table("claims") \
            .select("*") \
            .textSearch("text", query) \
            .limit(limit) \
            .execute()
    
    else:  # hybrid
        # Combine both, re-rank
        semantic = await search_claims(query, "semantic", limit * 2, filters)
        keyword = await search_claims(query, "keyword", limit * 2, filters)
        results = hybrid_rerank(semantic, keyword, limit)
    
    return apply_filters(results, filters)
```

### 9.3 Tool: `get_claim_support`

```typescript
{
  name: "get_claim_support",
  description: "Get all claims that support or contradict a specific claim",
  parameters: {
    claim_id: string,
    relationship_type: "supports" | "contradicts" | "both",
    direction: "incoming" | "outgoing" | "both",
    include_transitive: boolean,  // Follow chains up to depth N
    max_depth?: number            // Default: 1
  },
  returns: {
    claim: ClaimNode,            // The queried claim
    supporting_claims: Array<{
      claim: ClaimNode,
      edge: ClaimEdge,
      depth: number              // How many hops away
    }>,
    contradicting_claims: Array<{
      claim: ClaimNode,
      edge: ClaimEdge,
      depth: number
    }>
  }
}
```

**Implementation**:
```python
async def get_claim_support(claim_id: str, relationship_type: str, 
                           direction: str, include_transitive: bool, max_depth: int):
    # Use NetworkX cache for fast traversal
    G = graph_cache.G
    
    if direction in ["incoming", "both"]:
        # Claims that support THIS claim
        predecessors = list(G.predecessors(claim_id))
        supporting = [
            (pred, G[pred][claim_id])
            for pred in predecessors
            if G[pred][claim_id]["type"] == "supports"
        ]
    
    if include_transitive:
        # BFS to find all ancestors up to max_depth
        supporting = nx.bfs_edges(G.reverse(), claim_id, depth_limit=max_depth)
    
    return format_support_response(claim_id, supporting, contradicting)
```

### 9.4 Tool: `trace_provenance`

```typescript
{
  name: "trace_provenance",
  description: "Trace a claim back to its foundational ground truths or identify if ungrounded",
  parameters: {
    claim_id: string,
    max_depth: number            // Default: 10
  },
  returns: {
    claim: ClaimNode,
    is_grounded: boolean,
    ground_truths_found: Array<ClaimNode>,
    ungrounded_foundations: Array<ClaimNode>,  // Unsupported claims at chain end
    cycles_detected: Array<string[]>,          // Circular reasoning paths
    provenance_chains: Array<{
      path: string[],            // Claim IDs from target to foundation
      chain_confidence: number   // Product of edge weights
    }>
  }
}
```

**Implementation**:
```python
async def trace_provenance(claim_id: str, max_depth: int):
    G = graph_cache.G
    
    # Find all paths to ground truths
    ground_truths = [n for n in G.nodes if G.nodes[n]["type"] == "ground_truth"]
    
    chains = []
    for gt in ground_truths:
        try:
            paths = nx.all_simple_paths(G, claim_id, gt, cutoff=max_depth)
            for path in paths:
                confidence = compute_chain_confidence(G, path)
                chains.append({"path": path, "chain_confidence": confidence})
        except nx.NetworkXNoPath:
            continue
    
    # Detect cycles
    try:
        cycles = list(nx.simple_cycles(G.subgraph(nx.ancestors(G, claim_id) | {claim_id})))
    except:
        cycles = []
    
    # Find ungrounded endpoints
    ancestors = nx.ancestors(G, claim_id)
    ungrounded = [
        n for n in ancestors 
        if G.in_degree(n) == 0 and G.nodes[n]["type"] != "ground_truth"
    ]
    
    return {
        "is_grounded": len(chains) > 0,
        "ground_truths_found": [G.nodes[c["path"][-1]] for c in chains],
        "ungrounded_foundations": [G.nodes[n] for n in ungrounded],
        "cycles_detected": cycles,
        "provenance_chains": chains
    }
```

### 9.5 Tool: `get_related_claims`

```typescript
{
  name: "get_related_claims",
  description: "Get claims related by metadata (same paper, similar topic, etc.)",
  parameters: {
    claim_id: string,
    relation_type: "same_paper" | "same_author" | "similar_topic" | "citing_same",
    limit: number
  },
  returns: {
    source_claim: ClaimNode,
    related_claims: Array<{
      claim: ClaimNode,
      relation: string,
      relevance_score: number
    }>
  }
}
```

### 9.6 Tool: `find_contradictions`

```typescript
{
  name: "find_contradictions",
  description: "Find claims that contradict a given claim or find all contradictions in a topic",
  parameters: {
    claim_id?: string,           // Specific claim, or...
    topic_query?: string,        // Find contradictions within a topic
    min_weight: number           // Minimum contradiction weight (default: 0.5)
  },
  returns: {
    contradictions: Array<{
      claim_a: ClaimNode,
      claim_b: ClaimNode,
      edge: ClaimEdge,
      resolution_hints: string[] // Suggestions for why they might conflict
    }>
  }
}
```

### 9.7 Tool: `get_graph_statistics`

```typescript
{
  name: "get_graph_statistics",
  description: "Get overview statistics about the knowledge graph",
  parameters: {},
  returns: {
    total_claims: number,
    claims_by_type: {
      empirical: number,
      ground_truth: number,
      unsupported: number
    },
    total_edges: number,
    edges_by_type: {
      supports: number,
      contradicts: number
    },
    avg_support_depth: number,
    grounded_percentage: number,
    papers_indexed: number,
    last_updated: string
  }
}
```

---

## 10. Database Schema

### 10.1 Supabase Tables

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Claims table
CREATE TABLE claims (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    original_text TEXT,
    type TEXT NOT NULL CHECK (type IN ('empirical', 'ground_truth', 'unsupported')),
    confidence FLOAT NOT NULL DEFAULT 0.5 CHECK (confidence >= 0 AND confidence <= 1),
    
    -- Source paper (JSONB for flexibility)
    source_paper JSONB NOT NULL,
    -- {
    --   "arxiv_id": "2301.12345",
    --   "title": "Paper Title",
    --   "authors": ["Author 1", "Author 2"],
    --   "year": 2023,
    --   "venue": "Nature",
    --   "section": "Results",
    --   "url": "https://arxiv.org/abs/2301.12345"
    -- }
    
    -- External source (for non-arXiv)
    external_source JSONB,
    -- {
    --   "url": "https://...",
    --   "retrieval_method": "semantic_scholar",
    --   "retrieval_date": "2025-01-15",
    --   "verified": true
    -- }
    
    -- Extraction metadata
    extraction JSONB NOT NULL,
    -- {
    --   "model": "gpt-4o-mini",
    --   "timestamp": "2025-01-15T10:30:00Z",
    --   "prompt_version": "1.0",
    --   "context_snippet": "..."
    -- }
    
    -- Vector embedding (1536 dimensions for text-embedding-3-small)
    embedding vector(1536),
    
    -- Computed fields (updated by triggers or application)
    support_count INTEGER DEFAULT 0,
    contradict_count INTEGER DEFAULT 0,
    depth_to_ground_truth INTEGER,  -- NULL if ungrounded
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Edges table
CREATE TABLE edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    type TEXT NOT NULL CHECK (type IN ('supports', 'contradicts')),
    weight FLOAT NOT NULL DEFAULT 0.5 CHECK (weight >= 0 AND weight <= 1),
    reasoning TEXT,
    model TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Prevent duplicate edges
    UNIQUE(source_id, target_id, type)
);

-- Build runs table (for tracking pipeline executions)
CREATE TABLE build_runs (
    id SERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status TEXT CHECK (status IN ('running', 'completed', 'failed')),
    anchor_papers JSONB,
    claims_extracted INTEGER DEFAULT 0,
    edges_created INTEGER DEFAULT 0,
    errors JSONB,
    config JSONB
);

-- Indexes
CREATE INDEX claims_type_idx ON claims(type);
CREATE INDEX claims_confidence_idx ON claims(confidence);
CREATE INDEX claims_source_paper_idx ON claims USING GIN (source_paper);
CREATE INDEX edges_source_idx ON edges(source_id);
CREATE INDEX edges_target_idx ON edges(target_id);
CREATE INDEX edges_type_idx ON edges(type);

-- pgvector index for similarity search (IVFFlat for speed)
CREATE INDEX claims_embedding_idx ON claims 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### 10.2 Supabase Functions

```sql
-- Semantic search function
CREATE OR REPLACE FUNCTION search_claims_by_embedding(
    query_embedding vector(1536),
    match_count INT DEFAULT 10,
    filter_type TEXT DEFAULT NULL,
    min_confidence FLOAT DEFAULT 0.0
)
RETURNS TABLE (
    id TEXT,
    text TEXT,
    type TEXT,
    confidence FLOAT,
    source_paper JSONB,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.text,
        c.type,
        c.confidence,
        c.source_paper,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM claims c
    WHERE 
        (filter_type IS NULL OR c.type = filter_type)
        AND c.confidence >= min_confidence
        AND c.embedding IS NOT NULL
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Get claim with support/contradict counts
CREATE OR REPLACE FUNCTION get_claim_with_counts(claim_id_param TEXT)
RETURNS TABLE (
    id TEXT,
    text TEXT,
    type TEXT,
    confidence FLOAT,
    source_paper JSONB,
    support_count BIGINT,
    contradict_count BIGINT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.text,
        c.type,
        c.confidence,
        c.source_paper,
        (SELECT COUNT(*) FROM edges e WHERE e.target_id = c.id AND e.type = 'supports'),
        (SELECT COUNT(*) FROM edges e WHERE e.target_id = c.id AND e.type = 'contradicts')
    FROM claims c
    WHERE c.id = claim_id_param;
END;
$$;
```

### 10.3 Row Level Security (Optional)

```sql
-- Enable RLS
ALTER TABLE claims ENABLE ROW LEVEL SECURITY;
ALTER TABLE edges ENABLE ROW LEVEL SECURITY;

-- For now, allow all access (can restrict later)
CREATE POLICY "Allow all access to claims" ON claims FOR ALL USING (true);
CREATE POLICY "Allow all access to edges" ON edges FOR ALL USING (true);
```

---

## 11. API Endpoints

### 11.1 Graph Building Endpoints

```
POST /api/graph/build
  Start a new graph building pipeline
  
  Request:
  {
    "anchor_papers": ["arxiv_id_1", "arxiv_id_2", ...],  // Optional, uses defaults if not provided
    "max_depth": 3,
    "max_claims": 500,
    "parallel_workers": 5
  }
  
  Response:
  {
    "build_id": "build_123",
    "status": "started",
    "estimated_time_minutes": 30
  }

GET /api/graph/build/{build_id}
  Check build status
  
  Response:
  {
    "build_id": "build_123",
    "status": "running" | "completed" | "failed",
    "progress": {
      "papers_processed": 15,
      "claims_extracted": 127,
      "edges_created": 89,
      "current_depth": 2
    }
  }

POST /api/graph/refresh-cache
  Reload NetworkX cache from Supabase
  
  Response:
  {
    "success": true,
    "nodes_loaded": 450,
    "edges_loaded": 312
  }
```

### 11.2 Query Endpoints

```
POST /api/graph/search
  Search for claims
  
  Request:
  {
    "query": "gut microbiome depression",
    "search_type": "hybrid",
    "limit": 10,
    "filters": {
      "type": "empirical",
      "min_confidence": 0.5
    }
  }
  
  Response:
  {
    "claims": [
      {
        "id": "claim_abc123",
        "text": "Gut microbiome diversity correlates with depression severity",
        "type": "empirical",
        "confidence": 0.78,
        "source_paper": { ... },
        "similarity_score": 0.92
      }
    ]
  }

GET /api/graph/claim/{claim_id}
  Get full claim details
  
  Response:
  {
    "claim": { ... },
    "support_count": 5,
    "contradict_count": 1,
    "is_grounded": true,
    "depth_to_ground_truth": 3
  }

GET /api/graph/claim/{claim_id}/support
  Get supporting/contradicting claims
  
  Query params: ?type=supports&direction=incoming&depth=2
  
  Response:
  {
    "claim": { ... },
    "supporting_claims": [ ... ],
    "contradicting_claims": [ ... ]
  }

GET /api/graph/claim/{claim_id}/provenance
  Trace claim to ground truths
  
  Response:
  {
    "claim": { ... },
    "is_grounded": true,
    "provenance_chains": [
      {
        "path": ["claim_abc", "claim_def", "claim_ghi"],
        "chain_confidence": 0.72
      }
    ],
    "ground_truths_found": [ ... ],
    "ungrounded_foundations": [ ... ]
  }

GET /api/graph/stats
  Get graph statistics
  
  Response:
  {
    "total_claims": 450,
    "claims_by_type": { "empirical": 380, "ground_truth": 45, "unsupported": 25 },
    "total_edges": 312,
    "grounded_percentage": 0.84,
    "papers_indexed": 47
  }
```

---

## 12. Anchor Papers

### 12.1 Quantitative Biology Seed Papers

These foundational papers will seed the graph. Selection criteria:
- Confirmed availability on arXiv
- High citation count / influence
- Foundational to quantitative biology / computational biology
- Mix of subfields (genomics, microbiome, structural, ML for biology)

**Confirmed Anchor Papers**:

| # | arXiv ID | Title | Year | Subfield | Why Selected |
|---|----------|-------|------|----------|--------------|
| 1 | `1706.01787` | "Global metabolic interaction network of the human gut microbiota for context-specific community-scale analysis" | 2017 | **Microbiome** | NJS16 network: 570 species, 4,400+ interactions. Foundational for gut microbiome systems biology. Published in Nature Communications. |
| 2 | `1710.05086` | "Deep generative models for single-cell RNA sequencing" (scVI) | 2017 | **Genomics/ML** | Introduced scVI variational autoencoder for scRNA-seq. Highly influential, published in Nature Methods. Widely adopted. |
| 3 | `1704.01212` | "Neural Message Passing for Quantum Chemistry" | 2017 | **Cheminformatics** | Introduced MPNN framework. Foundational for GNNs in molecular property prediction. Basis for many subsequent methods. |
| 4 | `1605.08368` | "Inferring Biological Networks by Sparse Identification of Nonlinear Dynamics" | 2016 | **Systems Biology** | Implicit-SINDy for data-driven biological network inference. Applied to enzyme kinetics, bacterial regulation. |
| 5 | `1607.06358` | "Bayesian uncertainty analysis for complex systems biology models" | 2016 | **Systems Biology** | Statistical emulation + history matching for high-dimensional parameter spaces. Best practices for uncertainty quantification. |
| 6 | `1802.04944` | "Edge Attention-based Multi-Relational Graph Convolutional Networks" (EAGCN) | 2018 | **Drug Discovery** | GNN with edge attention for molecular property prediction. Multiple relationship types (bond, aromaticity, ring). |
| 7 | `1906.11081` | "Molecular Property Prediction: A Multilevel Quantum Interactions Modeling Perspective" (MGCN) | 2019 | **Cheminformatics** | Multilevel GCN capturing quantum-scale interactions. Handles equilibrium and off-equilibrium molecules. |
| 8 | `2104.10082` | "Modeling biological networks: from single gene systems to large microbial communities" | 2021 | **Microbiome/Gene Reg** | Spans gene regulation to gut microbiota population models. Statistical properties of abundance distributions. |
| 9 | `2207.13921` | "HelixFold-Single: MSA-free Protein Structure Prediction by Using Protein Language Models" | 2022 | **Structural Biology** | Single-sequence protein structure prediction. Combines PLM + AlphaFold2 modules. Faster inference. |
| 10 | `1905.02269` | "A joint model of unpaired data from scRNA-seq and spatial transcriptomics" (gimVI) | 2019 | **Genomics** | Integrates spatial transcriptomics with scRNA-seq. Imputes missing genes in spatial data. |
| 11 | `2509.07911` | "Gut-Brain Axis as a Closed-Loop Molecular Communication Network" | 2025 | **Neuroscience/Microbiome** | Models gut-brain axis with 6 coupled nonlinear DDEs. Information theory analysis. Very recent. |
| 12 | `2412.19945` | "Modeling and Analysis of SCFA-Driven Vagus Nerve Signaling in the Gut-Brain Axis" | 2024 | **Neuroscience/Microbiome** | SCFA → vagal afferents → action potentials. Molecular communication framework. |

### 12.2 Anchor Paper JSON Configuration

```json
{
  "anchor_papers": [
    {
      "arxiv_id": "1706.01787",
      "priority": 1,
      "subfield": "microbiome",
      "notes": "NJS16 gut microbiome metabolic network - foundational"
    },
    {
      "arxiv_id": "1710.05086",
      "priority": 1,
      "subfield": "genomics",
      "notes": "scVI - VAE for single-cell RNA-seq"
    },
    {
      "arxiv_id": "1704.01212",
      "priority": 1,
      "subfield": "cheminformatics",
      "notes": "MPNN - foundational GNN for molecules"
    },
    {
      "arxiv_id": "1605.08368",
      "priority": 2,
      "subfield": "systems_biology",
      "notes": "SINDy for biological network inference"
    },
    {
      "arxiv_id": "1607.06358",
      "priority": 2,
      "subfield": "systems_biology",
      "notes": "Bayesian uncertainty in systems biology"
    },
    {
      "arxiv_id": "1802.04944",
      "priority": 2,
      "subfield": "drug_discovery",
      "notes": "EAGCN - edge attention GNN for molecules"
    },
    {
      "arxiv_id": "1906.11081",
      "priority": 2,
      "subfield": "cheminformatics",
      "notes": "MGCN - multilevel quantum interactions"
    },
    {
      "arxiv_id": "2104.10082",
      "priority": 1,
      "subfield": "microbiome",
      "notes": "Gene to microbiome community modeling"
    },
    {
      "arxiv_id": "2207.13921",
      "priority": 2,
      "subfield": "structural_biology",
      "notes": "HelixFold-Single - MSA-free structure prediction"
    },
    {
      "arxiv_id": "1905.02269",
      "priority": 3,
      "subfield": "genomics",
      "notes": "gimVI - spatial + scRNA-seq integration"
    },
    {
      "arxiv_id": "2509.07911",
      "priority": 3,
      "subfield": "gut_brain",
      "notes": "Gut-brain axis molecular communication model"
    },
    {
      "arxiv_id": "2412.19945",
      "priority": 3,
      "subfield": "gut_brain",
      "notes": "SCFA vagus nerve signaling model"
    }
  ],
  "processing_order": "priority ascending (1 first)",
  "total_papers": 12
}
```

### 12.3 Subfield Distribution

| Subfield | Papers | Coverage |
|----------|--------|----------|
| Microbiome / Gut-Brain | 4 | 33% |
| Genomics (scRNA-seq) | 2 | 17% |
| Cheminformatics / Drug Discovery | 3 | 25% |
| Systems Biology | 2 | 17% |
| Structural Biology | 1 | 8% |

This distribution provides good coverage across quantitative biology while having a strong microbiome focus (relevant to the gut-brain demo domain in the mock graph).

### 12.4 Keyword Set for Relevance Filtering

Citations will be filtered based on keyword overlap with this set (derived from anchor paper domains):

```python
ANCHOR_KEYWORDS = {
    # Core biology
    "gene", "protein", "cell", "genome", "transcriptome", "proteome",
    "dna", "rna", "mrna", "expression", "regulation", "pathway",
    "metabolic", "metabolism", "enzyme", "amino acid",
    
    # Microbiome specific (strong focus given anchor papers)
    "microbiome", "microbiota", "gut", "intestinal", "bacteria", "microbial",
    "scfa", "short-chain fatty acid", "butyrate", "propionate", "acetate",
    "vagus", "gut-brain", "enteric", "colonocyte", "dysbiosis",
    
    # Single-cell genomics
    "single-cell", "scrna-seq", "rna-seq", "transcriptomics", "spatial",
    "variational", "autoencoder", "vae", "latent", "embedding",
    
    # Structural biology
    "protein structure", "folding", "conformation", "alphafold",
    "language model", "plm", "esm", "sequence",
    
    # Cheminformatics / Drug discovery
    "molecular", "molecule", "drug", "compound", "property prediction",
    "graph neural", "gnn", "message passing", "mpnn",
    "binding", "affinity", "toxicity", "solubility",
    
    # Systems biology / Quantitative methods
    "network", "dynamics", "dynamical", "differential equation",
    "stochastic", "bayesian", "inference", "parameter",
    "model", "simulation", "computational",
    
    # Machine learning general
    "neural network", "deep learning", "machine learning", "prediction",
    "classification", "clustering", "attention", "transformer"
}

def compute_keyword_overlap(paper_keywords: set, anchor_keywords: set = ANCHOR_KEYWORDS) -> float:
    """
    Compute Jaccard similarity with anchor keywords.
    
    Args:
        paper_keywords: Set of keywords from cited paper (title + abstract)
        anchor_keywords: Reference keyword set
        
    Returns:
        Overlap score between 0.0 and 1.0
    """
    # Normalize to lowercase
    paper_kw = {k.lower() for k in paper_keywords}
    anchor_kw = {k.lower() for k in anchor_keywords}
    
    intersection = paper_kw & anchor_kw
    union = paper_kw | anchor_kw
    
    return len(intersection) / len(union) if union else 0.0


def extract_keywords_from_text(text: str) -> set[str]:
    """
    Extract keywords from paper title + abstract for filtering.
    """
    import re
    
    # Simple tokenization
    words = re.findall(r'\b[a-zA-Z][a-zA-Z-]+\b', text.lower())
    
    # Filter common stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
                 'those', 'it', 'its', 'we', 'our', 'their', 'they', 'which', 'what'}
    
    # Also look for bigrams (important for terms like "neural network")
    bigrams = set()
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if bigram.lower() in {k.lower() for k in ANCHOR_KEYWORDS}:
            bigrams.add(bigram)
    
    keywords = {w for w in words if w not in stopwords and len(w) > 2}
    keywords.update(bigrams)
    
    return keywords


# Relevance threshold for citation inclusion
RELEVANCE_THRESHOLD = 0.15  # At least 15% keyword overlap to include citation
```

---

## 13. Implementation Phases

### Phase 1: Infrastructure (Day 1, Hours 1-4)

| Task | Owner | Deliverable |
|------|-------|-------------|
| Set up Supabase tables | Backend | SQL schema deployed |
| Enable pgvector extension | Backend | Vector search working |
| Create NetworkX cache class | Backend | `GraphCache` class |
| Set up arXiv API client | Backend | `ArxivClient` class |

**Checkpoint**: Can store and retrieve claims from Supabase.

### Phase 2: Extraction Pipeline (Day 1, Hours 5-10)

| Task | Owner | Deliverable |
|------|-------|-------------|
| Claim extraction prompts | Backend | Tested prompts for GPT-4o-mini |
| Classification prompts | Backend | Tested classification logic |
| Edge detection prompts | Backend | Tested relationship detection |
| Embedding generation | Backend | Integration with text-embedding-3-small |
| Single paper extraction | Backend | `extract_claims_from_paper()` |

**Checkpoint**: Can extract claims from a single arXiv paper.

### Phase 3: Graph Building (Day 2, Hours 1-4)

| Task | Owner | Deliverable |
|------|-------|-------------|
| Anchor paper list | Both | 10-15 confirmed arXiv IDs |
| Sequential seed processing | Backend | Process all anchor papers |
| Parallel expansion workers | Backend | `expand_citations_parallel()` |
| Stopping conditions | Backend | Depth/budget limits working |

**Checkpoint**: Can build graph from anchor papers + 1 depth of citations.

### Phase 4: MCP Integration (Day 2, Hours 5-8)

| Task | Owner | Deliverable |
|------|-------|-------------|
| `search_claims` tool | Backend | Working semantic search |
| `get_claim_support` tool | Backend | Working traversal |
| `trace_provenance` tool | Backend | Working provenance chains |
| MCP server setup | Backend | Tools exposed via MCP |

**Checkpoint**: Deep Research Agent can query the graph via MCP.

### Phase 5: Integration & Polish (Day 2, Hours 9-10)

| Task | Owner | Deliverable |
|------|-------|-------------|
| End-to-end test | Both | Query → Graph → Deep Research → Response |
| Demo preparation | Both | Sample queries with good results |
| Documentation | Both | README updates |

**Checkpoint**: Demo-ready system.

---

## 14. Appendix

### A. File Structure

```
backend/
├── graph/
│   ├── __init__.py
│   ├── knowledge_graph.py      # Legacy (keep for now)
│   ├── graph_cache.py          # NEW: NetworkX cache layer
│   ├── claim_extractor.py      # NEW: GPT-4o-mini extraction
│   ├── graph_builder.py        # NEW: Build pipeline orchestration
│   └── provenance.py           # NEW: Provenance tracing logic
│
├── tools/                      # NEW: MCP tools
│   ├── __init__.py
│   ├── search_claims.py
│   ├── get_claim_support.py
│   ├── trace_provenance.py
│   ├── get_related_claims.py
│   └── find_contradictions.py
│
├── clients/                    # NEW: External API clients
│   ├── __init__.py
│   ├── arxiv_client.py
│   ├── semantic_scholar.py
│   └── embeddings.py
│
├── data/
│   ├── mock_graph.json         # Keep for reference
│   └── anchor_papers.json      # NEW: List of seed papers
│
├── docs/
│   ├── PRD_Scientific_Discovery_Agent.md
│   └── PRD_Knowledge_Graph.md  # THIS DOCUMENT
│
└── main.py                     # Add new endpoints
```

### B. Environment Variables

```bash
# .env additions
OPENAI_API_KEY=sk-...                    # For GPT-4o-mini and embeddings
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
SEMANTIC_SCHOLAR_API_KEY=...             # Optional, for better rate limits

# Build configuration
GRAPH_MAX_DEPTH=3
GRAPH_MAX_CLAIMS=500
GRAPH_PARALLEL_WORKERS=5
```

### C. Key Algorithms

**Chain Confidence Calculation**:
```python
def compute_chain_confidence(G: nx.DiGraph, path: list[str]) -> float:
    """
    Compute confidence of a provenance chain as product of edge weights.
    
    A chain [A, B, C] means A←B←C (C supports B supports A).
    Confidence = weight(B→A) * weight(C→B)
    """
    confidence = 1.0
    for i in range(len(path) - 1):
        source = path[i + 1]
        target = path[i]
        edge_data = G.get_edge_data(source, target)
        if edge_data:
            confidence *= edge_data.get("weight", 0.5)
        else:
            confidence *= 0.1  # Penalty for missing edge
    return confidence
```

**Deduplication**:
```python
def merge_duplicate_claims(claim1: ClaimNode, claim2: ClaimNode) -> ClaimNode:
    """
    Merge two claims that express the same assertion.
    
    - Keep the higher confidence
    - Combine source paper metadata
    - Keep earliest extraction timestamp
    """
    merged = ClaimNode(
        id=claim1.id,  # Keep first ID
        text=claim1.text,  # Assume same text (that's why they're duplicates)
        type=claim1.type if claim1.confidence > claim2.confidence else claim2.type,
        confidence=max(claim1.confidence, claim2.confidence),
        source_papers=[claim1.source_paper, claim2.source_paper],  # Combine
        extraction=claim1.extraction if claim1.extraction["timestamp"] < claim2.extraction["timestamp"] else claim2.extraction
    )
    return merged
```

### D. Error Handling

| Error | Handling |
|-------|----------|
| arXiv rate limit | Exponential backoff (1s, 2s, 4s, 8s), max 3 retries |
| GPT-4o-mini rate limit | Queue with semaphore, 5 concurrent max |
| Embedding API failure | Retry 3x, then skip embedding (mark as needs_embedding) |
| Invalid arXiv ID | Log warning, mark as external_unverified |
| Cycle detected in graph | Log warning, continue (cycles are valid data) |
| Claim extraction returns empty | Retry with different prompt, then skip paper |

---

*This PRD is the source of truth for Knowledge Graph implementation.*  
*Last Updated: January 2026*
