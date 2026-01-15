# 📋 PRD: ClaimGraph Scientific Discovery Agent
## Deep Research Over Knowledge Graphs

---

## 1. 📌 Executive Summary

**ClaimGraph** is a human-in-the-loop scientific discovery agent that navigates a Scientific Knowledge Graph (KG) to perform deep research. The system uses a **Planner-Worker cognitive architecture** where GPT-4.1 orchestrates high-level reasoning while o3-deep-research executes precise graph traversal operations.

**Core Value Proposition:** Transform passive graph querying into active scientific investigation—surfacing contradictions, tracing axiom provenance, and synthesizing findings through an interactive, visual interface.

| Attribute | Value |
|-----------|-------|
| Timeline | 2 days (hackathon sprint) |
| Team | 2 full-stack developers |
| Deployment | Local development environment |
| Target User | Single-user demo (researcher persona) |
| Hero Feature | **Debate Simulator** |

---

## 2. 🎯 Problem Statement

### The Gap
Scientific knowledge graphs contain rich relational data (supports, contradicts, cites), but:
1. **LLMs hallucinate** connections that don't exist in the graph
2. **Graph traversal is opaque**—users can't see the agent's reasoning path
3. **Contradictions are buried**—conflicting papers aren't surfaced proactively
4. **No synthesis**—raw graph data isn't transformed into insight

### The Opportunity
Build an agent that treats the KG as a "reasoning substrate"—grounding every claim in traversable edges while presenting findings through an interactive, visual interface that makes the research process legible.

---

## 3. ✅ Goals & Success Metrics

### Primary Goals (Must Ship)
| Goal | Success Metric |
|------|----------------|
| Grounded graph traversal | 0 hallucinated edges in demo |
| Visual reasoning trace | Every agent step renders in Mermaid |
| Debate Simulator works | Demo shows 2 "authors" arguing a contradiction |
| Human-in-the-loop flow | User can pause, redirect, and approve agent actions |

### Stretch Goals (If Time Permits)
| Goal | Success Metric |
|------|----------------|
| Axiom Buster | Trace claim to oldest ancestor |
| Cross-Pollinator | Find cross-domain structural match |
| Session persistence | Resume investigation after refresh |

---

## 4. 👤 User Personas & Stories

### Primary Persona: Dr. Maya Chen
- **Role:** Computational biologist
- **Pain:** Spends hours manually tracing citation chains
- **Goal:** Quickly validate whether a popular claim has solid foundational evidence

### User Stories

| ID | Story | Priority |
|----|-------|----------|
| US-01 | As Maya, I want to enter a research question and see the agent formulate an investigation plan | P0 |
| US-02 | As Maya, I want to see the agent's traversal path visualized as it explores | P0 |
| US-03 | As Maya, I want to pause the agent and inject my own hypothesis | P1 |
| US-04 | As Maya, I want to trigger "Debate Mode" when the agent finds a contradiction | P0 |
| US-05 | As Maya, I want to export the final reasoning graph as JSON/image | P2 |
| US-06 | As Maya, I want the agent to explain why it pruned certain paths | P1 |

---

## 5. 🏗️ System Architecture

### 5.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         REACT UI                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Query Input │  │ Control Bar │  │ Mermaid Graph Renderer  │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│  ┌─────────────────────────────┐  ┌───────────────────────────┐│
│  │   Investigation Timeline   │  │   Debate Simulator View   ││
│  └─────────────────────────────┘  └───────────────────────────┘│
└───────────────────────────┬─────────────────────────────────────┘
                            │ WebSocket / REST
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MCP SERVER (Local)                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    PLANNER (GPT-4.1)                        ││
│  │  • Maintains Investigation State                            ││
│  │  • Scores & prunes candidate paths                          ││
│  │  • Delegates traversal tasks to Worker                      ││
│  └─────────────────────────────┬───────────────────────────────┘│
│                                │ Internal call                  │
│  ┌─────────────────────────────▼───────────────────────────────┐│
│  │                    WORKER (o3-deep-research)                ││
│  │  • Executes graph queries via tools                         ││
│  │  • Returns structured JSON results                          ││
│  │  • Reports dead-ends gracefully                             ││
│  └─────────────────────────────┬───────────────────────────────┘│
│                                │                                │
│  ┌─────────────────────────────▼───────────────────────────────┐│
│  │                    GRAPH TOOLS                              ││
│  │  search_nodes | traverse_edge | get_metadata | find_paths   ││
│  └─────────────────────────────┬───────────────────────────────┘│
└────────────────────────────────┼────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   KNOWLEDGE GRAPH DB   │
                    │      (Given/Mock)      │
                    └────────────────────────┘
```

### 5.2 Component Responsibilities

| Component | Technology | Responsibility |
|-----------|------------|----------------|
| **React UI** | React + Vite | User input, visualization, human-in-the-loop controls |
| **MCP Server** | Node.js / TypeScript | Tool execution, Planner-Worker orchestration |
| **Planner** | GPT-4.1 API | High-level reasoning, state management, path scoring |
| **Worker** | o3-deep-research API | Graph traversal execution, structured result generation |
| **Graph DB** | Mock JSON / Neo4j | Stores nodes (claims/papers) and edges (supports/contradicts/cites) |

---

## 6. 🔧 MCP Server Specification

### 6.1 Tool Definitions

The MCP server exposes the following tools to the LLM agents:

#### Tool: `search_nodes`
```typescript
{
  name: "search_nodes",
  description: "Semantic search for claim/paper nodes in the knowledge graph",
  parameters: {
    query: string,           // Natural language search query
    limit?: number,          // Max results (default: 10)
    node_type?: "claim" | "paper" | "all"
  },
  returns: {
    nodes: Array<{
      id: string,
      label: string,
      type: "claim" | "paper",
      metadata: {
        year?: number,
        citations?: number,
        authors?: string[],
        venue?: string
      },
      relevance_score: number
    }>
  }
}
```

#### Tool: `traverse_edge`
```typescript
{
  name: "traverse_edge",
  description: "Follow edges from a node in the knowledge graph",
  parameters: {
    node_id: string,
    edge_type: "supports" | "contradicts" | "cites" | "all",
    direction: "outgoing" | "incoming" | "both",
    limit?: number
  },
  returns: {
    source_node: string,
    edges: Array<{
      id: string,
      type: "supports" | "contradicts" | "cites",
      target_node: {
        id: string,
        label: string,
        type: "claim" | "paper",
        metadata: object
      },
      weight?: number,
      evidence_snippet?: string
    }>
  }
}
```

#### Tool: `get_metadata`
```typescript
{
  name: "get_metadata",
  description: "Retrieve full metadata for a specific node",
  parameters: {
    node_id: string
  },
  returns: {
    id: string,
    label: string,
    type: "claim" | "paper",
    abstract?: string,
    full_text_available: boolean,
    metadata: {
      year: number,
      citations: number,
      authors: string[],
      venue: string,
      doi?: string
    },
    edge_counts: {
      supports_outgoing: number,
      supports_incoming: number,
      contradicts_outgoing: number,
      contradicts_incoming: number,
      cites_outgoing: number,
      cites_incoming: number
    }
  }
}
```

#### Tool: `find_paths`
```typescript
{
  name: "find_paths",
  description: "Find shortest paths between two nodes",
  parameters: {
    source_id: string,
    target_id: string,
    max_hops?: number,        // Default: 5
    edge_types?: string[]     // Filter by edge types
  },
  returns: {
    paths: Array<{
      nodes: string[],
      edges: Array<{
        from: string,
        to: string,
        type: string
      }>,
      length: number
    }>
  }
}
```

#### Tool: `get_subgraph`
```typescript
{
  name: "get_subgraph",
  description: "Extract a subgraph around a focal node",
  parameters: {
    center_node_id: string,
    depth: number,            // How many hops from center
    edge_types?: string[]
  },
  returns: {
    nodes: Array<NodeObject>,
    edges: Array<EdgeObject>
  }
}
```

### 6.2 MCP Message Protocol

#### Request Format (UI → MCP)
```typescript
interface MCPRequest {
  id: string;                          // UUID for request tracking
  type: "query" | "control" | "debate";
  payload: {
    query?: string;                    // For type: "query"
    action?: "pause" | "resume" | "redirect" | "trigger_debate";
    context?: {
      selected_node_id?: string;
      user_hypothesis?: string;
    }
  };
  timestamp: string;
}
```

#### Response Format (MCP → UI)
```typescript
interface MCPResponse {
  id: string;                          // Matches request ID
  type: "update" | "complete" | "error" | "debate_turn";
  payload: {
    investigation_state: InvestigationState;
    graph_delta?: GraphDelta;          // Incremental graph updates
    message?: string;                  // Human-readable status
    debate?: DebateTurn;              // For debate mode
  };
  timestamp: string;
}
```

#### Streaming Updates (WebSocket)
```typescript
interface StreamEvent {
  event: "thought" | "action" | "result" | "prune" | "debate";
  data: {
    step_id: string;
    content: string;
    graph_update?: GraphDelta;
    confidence?: number;
  };
}
```

---

## 7. 🤖 Agent Architecture

### 7.1 Planner-Worker Loop

```
┌──────────────────────────────────────────────────────────────┐
│                    PLANNER (GPT-4.1)                         │
│                                                              │
│  1. Parse user query                                         │
│  2. Initialize Investigation State                           │
│  3. LOOP:                                                    │
│     a. Score frontier nodes                                  │
│     b. Select highest-priority node                          │
│     c. Formulate task for Worker                             │
│     d. Dispatch to Worker ──────────────────────┐            │
│     e. Receive Worker result ◄──────────────────┤            │
│     f. Update Investigation State               │            │
│     g. Check termination conditions             │            │
│        - Research saturation?                   │            │
│        - User interrupt?                        │            │
│        - Max depth reached?                     │            │
│     h. Prune low-value branches                 │            │
│     i. Emit graph update to UI                  │            │
│  4. Synthesize findings                         │            │
│  5. Return final report + graph                 │            │
└─────────────────────────────────────────────────┼────────────┘
                                                  │
                    ┌─────────────────────────────▼────────────┐
                    │              WORKER (o3)                 │
                    │                                          │
                    │  1. Receive task from Planner            │
                    │  2. Select appropriate tool(s)           │
                    │  3. Execute graph query                  │
                    │  4. Format structured response           │
                    │  5. Return result OR failure notice      │
                    │                                          │
                    └──────────────────────────────────────────┘
```

### 7.2 Worker Failure Handling

The Worker must handle failures gracefully and return structured notices:

```typescript
interface WorkerResult {
  status: "ok" | "dead_end" | "error" | "rate_limited";
  task_id: string;
  node_id?: string;
  
  // On success
  data?: {
    nodes_found: NodeObject[];
    edges_found: EdgeObject[];
    insights: string[];
  };
  
  // On failure
  reason?: string;
  suggestions?: string[];  // Alternative paths to try
  
  // Always included
  tokens_used: number;
  execution_time_ms: number;
}
```

**Dead-End Response Example:**
```json
{
  "status": "dead_end",
  "task_id": "task_042",
  "node_id": "paper_1987_smith",
  "reason": "No outgoing 'cites' edges found. This appears to be a foundational paper.",
  "suggestions": [
    "Try incoming 'cites' edges to find papers that cite this one",
    "Check 'contradicts' edges for opposing viewpoints"
  ],
  "tokens_used": 145,
  "execution_time_ms": 892
}
```

### 7.3 Investigation State Schema

The Planner maintains a JSON state object throughout the session:

```typescript
interface InvestigationState {
  // Session metadata
  session_id: string;
  created_at: string;
  updated_at: string;
  status: "active" | "paused" | "completed" | "error";
  
  // Original query
  query: {
    original: string;
    clarified?: string;
    extracted_entities: string[];
    research_goal: string;
  };
  
  // Graph exploration state
  graph: {
    visited_nodes: Set<string>;       // IDs of fully explored nodes
    frontier: FrontierNode[];         // Nodes awaiting exploration
    pruned_nodes: Set<string>;        // IDs of pruned nodes
    edges_traversed: number;
    
    // The accumulated subgraph for visualization
    nodes: Map<string, NodeObject>;
    edges: Map<string, EdgeObject>;
  };
  
  // Reasoning trace
  reasoning: {
    steps: ReasoningStep[];
    current_hypothesis: string;
    confidence: number;               // 0-1 scale
    contradictions_found: Contradiction[];
  };
  
  // Findings
  findings: {
    key_claims: Claim[];
    evidence_chains: EvidenceChain[];
    open_questions: string[];
  };
  
  // Control
  control: {
    max_depth: number;
    max_nodes: number;
    saturation_threshold: number;     // Stop when new info rate drops
    user_paused: boolean;
    user_redirections: Redirection[];
  };
}

interface FrontierNode {
  node_id: string;
  priority_score: number;             // Higher = explore sooner
  reason: string;                     // Why this node is interesting
  depth: number;                      // Hops from origin
  parent_node_id?: string;
}

interface ReasoningStep {
  step_id: string;
  timestamp: string;
  type: "search" | "traverse" | "analyze" | "prune" | "synthesize";
  description: string;
  node_ids_involved: string[];
  outcome: string;
  confidence_delta: number;
}

interface Contradiction {
  id: string;
  node_a: string;
  node_b: string;
  edge_id: string;
  summary: string;
  debate_available: boolean;
}
```

### 7.4 Research Saturation Heuristics

The Planner uses these signals to determine when to stop exploring:

| Signal | Threshold | Action |
|--------|-----------|--------|
| **Novelty Rate** | < 10% new nodes in last 5 traversals | Suggest stopping |
| **Contradiction Density** | > 3 contradictions unresolved | Prompt user for direction |
| **Depth Limit** | > 7 hops from origin | Hard stop on branch |
| **Node Budget** | > 50 nodes visited | Hard stop, synthesize |
| **Confidence Plateau** | Δ confidence < 0.01 for 3 steps | Suggest stopping |
| **User Signal** | Pause button pressed | Immediate pause |

---

## 8. 💬 Prompt Engineering

### 8.1 Planner System Prompt

```markdown
# SYSTEM PROMPT: Scientific Discovery Planner

You are the **Planner** in a scientific discovery system. Your role is to orchestrate deep research over a Scientific Knowledge Graph by coordinating with a Worker agent who executes graph queries.

## Your Capabilities
- Formulate investigation plans from user queries
- Prioritize which nodes/edges to explore next
- Delegate specific graph operations to the Worker
- Synthesize findings into coherent insights
- Detect and surface contradictions in the literature

## Your Constraints
- You CANNOT directly query the graph—only the Worker can
- You MUST ground all claims in graph data returned by the Worker
- You MUST maintain the Investigation State accurately
- You MUST emit graph updates after each significant step

## Investigation State
You maintain a JSON state object (schema provided) that tracks:
- Visited and frontier nodes
- Reasoning steps taken
- Findings and contradictions
- User control signals

## Communication Protocol

### To dispatch a task to the Worker:
```json
{
  "task_id": "unique_id",
  "action": "traverse_edge | search_nodes | get_metadata | find_paths",
  "parameters": { ... },
  "rationale": "Why this action advances the investigation"
}
```

### Worker will respond with:
```json
{
  "status": "ok | dead_end | error",
  "data": { ... } | null,
  "reason": "..." // if status != ok
}
```

## Decision Framework

### Prioritizing Frontier Nodes
Score each frontier node by:
1. **Relevance** (0-1): How related to the research goal?
2. **Novelty** (0-1): How much new information might it provide?
3. **Connectivity** (0-1): How many unexplored edges does it have?
4. **Recency** (0-1): Prefer newer publications

Priority = (0.4 × Relevance) + (0.3 × Novelty) + (0.2 × Connectivity) + (0.1 × Recency)

### Pruning Criteria
Prune a branch if:
- Relevance score < 0.3
- Node is > 7 hops from origin
- Node has already been visited
- Node type doesn't match research goal

### Termination Conditions
Stop exploration when:
- User requests stop
- Novelty rate < 10% over last 5 steps
- Node budget exhausted (50 nodes)
- High-confidence answer found (confidence > 0.85)

## Output Format

After each step, emit:
```json
{
  "step_type": "thought | action | result | synthesis",
  "content": "Human-readable description",
  "graph_update": {
    "add_nodes": [...],
    "add_edges": [...],
    "highlight_node": "node_id" | null
  },
  "state_update": { ... partial state update ... }
}
```

## Special Modes

### Debate Mode
When you detect a CONTRADICTS edge between two papers:
1. Pause normal exploration
2. Announce: "Found contradiction between [Paper A] and [Paper B]"
3. If user triggers debate, switch to Debate Simulator protocol
4. After debate, integrate insights and resume

Remember: Your goal is not just to retrieve—it's to DISCOVER. Surface surprising connections, challenge assumptions, and help the user see the scientific landscape clearly.
```

### 8.2 Worker System Prompt

```markdown
# SYSTEM PROMPT: Knowledge Graph Explorer (Worker)

You are a **Worker** agent specialized in executing precise queries against a Scientific Knowledge Graph. You operate under the direction of a Planner agent.

## Your Role
- Execute graph operations using the provided tools
- Return structured, accurate results
- Report failures gracefully with actionable suggestions
- NEVER hallucinate or invent data

## Critical Constraints

### Grounding Rule
You MUST ONLY report information that comes directly from tool responses. If a tool returns no results, report that honestly—do not fabricate data.

### Output Format
ALL responses must be valid JSON matching this schema:

```json
{
  "status": "ok" | "dead_end" | "error",
  "task_id": "echo the task_id from the request",
  "node_id": "the focal node if applicable",
  
  // If status == "ok":
  "data": {
    "nodes_found": [...],
    "edges_found": [...],
    "insights": ["Observation 1", "Observation 2"]
  },
  
  // If status == "dead_end":
  "reason": "Clear explanation of why this is a dead end",
  "suggestions": ["Alternative approach 1", "Alternative approach 2"],
  
  // If status == "error":
  "error_type": "tool_error | parse_error | rate_limit",
  "error_message": "Detailed error description",
  
  // Always:
  "tokens_used": 0,
  "execution_time_ms": 0
}
```

## Tool Usage Guidelines

### search_nodes
Use for: Finding starting points, discovering related papers
Best practices:
- Be specific with queries
- Request appropriate limits (5-10 for exploration, 20+ for broad survey)

### traverse_edge
Use for: Following citation chains, finding support/contradiction
Best practices:
- Specify edge_type when you have a hypothesis
- Use direction wisely: "outgoing" for what this paper claims, "incoming" for who cites/challenges it

### get_metadata
Use for: Deep dive on a specific paper before deciding to explore further
Best practices:
- Check edge_counts to assess node importance
- Use citation count as a proxy for influence

### find_paths
Use for: Understanding how two claims/papers relate
Best practices:
- Keep max_hops low (3-4) for interpretable paths
- Filter edge_types if you're looking for specific relationship chains

## Dead End Handling

A dead end is NOT a failure. Report it with:
1. Clear reason (no edges? wrong domain? too old?)
2. At least 2 alternative suggestions
3. Any partial insights gained

Example:
```json
{
  "status": "dead_end",
  "task_id": "task_015",
  "node_id": "claim_mitochondria_energy",
  "reason": "This claim node has no outgoing 'contradicts' edges—it appears to be well-established consensus.",
  "suggestions": [
    "Search for recent papers that cite this claim to find modern challenges",
    "Explore 'supports' edges to understand the evidence base"
  ]
}
```

## Quality Signals

For each node/edge you return, assess:
- **Relevance**: How related to the Planner's stated goal?
- **Recency**: Publication year (newer often better for active research)
- **Impact**: Citation count as proxy
- **Controversy**: Presence of 'contradicts' edges

Include brief insights like:
- "High-impact paper (500+ citations) from 2019"
- "Controversial: 3 contradicting papers found"
- "Foundational: cited by 50+ papers in this subgraph"

Remember: You are the Planner's eyes into the graph. Be precise, be honest, and always provide enough context for good decisions.
```

### 8.3 Debate Simulator Prompts

#### Debate Moderator Prompt (Planner assumes this role)
```markdown
# DEBATE MODERATOR PROTOCOL

You are moderating a scientific debate between two papers with contradictory claims.

## Setup
- **Paper A**: {{paper_a_title}} ({{paper_a_year}})
  - Claim: {{paper_a_claim}}
  
- **Paper B**: {{paper_b_title}} ({{paper_b_year}})
  - Claim: {{paper_b_claim}}

## Your Role
1. Frame the debate with a clear thesis statement
2. Alternate between the two perspectives (3 rounds each)
3. Identify the crux of the disagreement
4. Synthesize: What would resolve this?

## Round Structure
- **Round 1**: Each paper states its core claim with primary evidence
- **Round 2**: Each paper responds to the other's argument
- **Round 3**: Each paper addresses: "What evidence would change your mind?"

## Output per Round
```json
{
  "round": 1,
  "speaker": "Paper A" | "Paper B",
  "argument": "The actual argument text",
  "evidence_cited": ["node_id_1", "node_id_2"],
  "rhetorical_move": "claim | rebuttal | concession | challenge"
}
```

## Final Synthesis
After 6 turns, provide:
```json
{
  "crux": "The fundamental point of disagreement",
  "resolution_path": "What evidence or experiment would resolve this?",
  "implications": "What does this mean for the user's research question?",
  "winner": null | "Paper A" | "Paper B",
  "confidence": 0.0-1.0
}
```

Keep arguments grounded in the actual paper abstracts/content from the graph. No hallucinated claims.
```

#### Debater Persona Prompt (Worker assumes this role during debate)
```markdown
# DEBATER PERSONA: {{paper_title}}

You are arguing the position of "{{paper_title}}" ({{year}}) by {{authors}}.

## Your Claim
{{main_claim}}

## Your Evidence Base
{{abstract}}

## Debate Rules
1. Argue ONLY from the evidence in this paper and papers it cites
2. When challenged, you may request data from cited papers via tools
3. Be rigorous but fair—concede valid points
4. Avoid ad hominem—focus on methods and evidence

## Response Format
```json
{
  "argument": "Your argument text (2-4 sentences)",
  "evidence": "Quote or paraphrase from the paper",
  "confidence": 0.0-1.0,
  "concessions": ["Any points you concede"],
  "challenges": ["Questions for the opponent"]
}
```

Argue persuasively but honestly. The goal is illumination, not victory.
```

---

## 9. 🖥️ UI Specification

### 9.1 Screen Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  🧠 ClaimGraph                                   [Settings] [Export]    │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     QUERY INPUT BAR                                │ │
│  │  ┌──────────────────────────────────────────┐  [🔍 Investigate]   │ │
│  │  │ What is the evidence that X causes Y?    │                      │ │
│  │  └──────────────────────────────────────────┘                      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
├──────────────────────────────────┬──────────────────────────────────────┤
│         REASONING PANEL          │          GRAPH PANEL                 │
│                                  │                                      │
│  ┌────────────────────────────┐  │  ┌──────────────────────────────┐   │
│  │ Step 1: Searching for...  │  │  │                              │   │
│  │ ✓ Found 5 relevant papers │  │  │      [MERMAID GRAPH]         │   │
│  ├────────────────────────────┤  │  │                              │   │
│  │ Step 2: Traversing cites  │  │  │   ┌───┐     ┌───┐           │   │
│  │ ⏳ Exploring Paper A...   │  │  │   │ A │────▶│ B │           │   │
│  ├────────────────────────────┤  │  │   └───┘     └───┘           │   │
│  │                            │  │  │      │       ▲              │   │
│  │  [▶ Resume] [⏸ Pause]     │  │  │      │  contradicts         │   │
│  │                            │  │  │      ▼       │              │   │
│  │  💡 Hypothesis:            │  │  │   ┌───┐     ┌───┐           │   │
│  │  "X appears supported by  │  │  │   │ C │─ ─ ─│ D │           │   │
│  │   3 papers from 2018-22"  │  │  │   └───┘     └───┘           │   │
│  │                            │  │  │                              │   │
│  │  ⚠️ Contradiction Found!   │  │  │                              │   │
│  │  [🎭 Start Debate]         │  │  │                              │   │
│  └────────────────────────────┘  │  └──────────────────────────────┘   │
│                                  │                                      │
│  ┌────────────────────────────┐  │  ┌──────────────────────────────┐   │
│  │      FINDINGS PANEL        │  │  │      LEGEND                   │  │
│  │                            │  │  │  ── supports                  │  │
│  │  📍 Key Claims:            │  │  │  ─ ─ contradicts              │  │
│  │  • Claim 1 (conf: 0.85)   │  │  │  ──▶ cites                    │  │
│  │  • Claim 2 (conf: 0.72)   │  │  │                               │  │
│  │                            │  │  │  🟢 High confidence           │  │
│  │  ❓ Open Questions:        │  │  │  🟡 Medium confidence         │  │
│  │  • Why does Z conflict?   │  │  │  🔴 Contested                  │  │
│  └────────────────────────────┘  │  └──────────────────────────────┘   │
├──────────────────────────────────┴──────────────────────────────────────┤
│                           DEBATE PANEL (Expandable)                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  🎭 DEBATE: Paper A vs Paper B                                    │  │
│  │                                                                   │  │
│  │  ┌─────────────────────┐        ┌─────────────────────┐          │  │
│  │  │     PAPER A         │   VS   │     PAPER B         │          │  │
│  │  │                     │        │                     │          │  │
│  │  │ "Our study shows    │        │ "We found no        │          │  │
│  │  │  that X causes Y    │        │  significant link   │          │  │
│  │  │  with p < 0.01"     │        │  between X and Y"   │          │  │
│  │  └─────────────────────┘        └─────────────────────┘          │  │
│  │                                                                   │  │
│  │  Round 2/3                                    [Next Round ▶]     │  │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Component Hierarchy

```
App
├── Header
│   ├── Logo ("ClaimGraph")
│   ├── SettingsButton
│   └── ExportButton
│
├── QueryBar
│   ├── TextInput
│   ├── InvestigateButton
│   └── SuggestionDropdown (optional)
│
├── MainLayout (split view)
│   ├── LeftPanel
│   │   ├── ReasoningTimeline
│   │   │   ├── StepCard (repeating)
│   │   │   │   ├── StepIcon
│   │   │   │   ├── StepDescription
│   │   │   │   └── StepStatus
│   │   │   └── ControlBar
│   │   │       ├── PauseButton
│   │   │       ├── ResumeButton
│   │   │       └── RedirectInput
│   │   │
│   │   ├── HypothesisCard
│   │   │   ├── HypothesisText
│   │   │   └── ConfidenceBar
│   │   │
│   │   ├── ContradictionAlert
│   │   │   ├── AlertMessage
│   │   │   └── StartDebateButton
│   │   │
│   │   └── FindingsPanel
│   │       ├── KeyClaimsList
│   │       └── OpenQuestionsList
│   │
│   └── RightPanel
│       ├── MermaidGraph
│       │   ├── GraphRenderer
│       │   ├── ZoomControls
│       │   └── NodeTooltip
│       │
│       └── Legend
│           ├── EdgeTypeLegend
│           └── ConfidenceLegend
│
└── DebatePanel (expandable/collapsible)
    ├── DebateHeader
    │   ├── PaperATitle
    │   ├── VSBadge
    │   └── PaperBTitle
    │
    ├── DebateArena
    │   ├── ArgumentCard (Paper A)
    │   └── ArgumentCard (Paper B)
    │
    └── DebateControls
        ├── RoundIndicator
        └── NextRoundButton
```

### 9.3 Mermaid Graph Rendering

#### Graph Update Schema (from Agent → UI)
```typescript
interface GraphDelta {
  add_nodes?: MermaidNode[];
  add_edges?: MermaidEdge[];
  remove_nodes?: string[];
  update_nodes?: Partial<MermaidNode>[];
  highlight?: string;              // Node ID to pulse/highlight
  center_on?: string;              // Node ID to center view on
}

interface MermaidNode {
  id: string;
  label: string;
  type: "claim" | "paper";
  confidence?: number;             // 0-1, affects color
  status?: "visited" | "frontier" | "pruned" | "highlighted";
}

interface MermaidEdge {
  source: string;
  target: string;
  type: "supports" | "contradicts" | "cites";
  label?: string;
}
```

#### Mermaid Template Generator
```typescript
function generateMermaidCode(state: InvestigationState): string {
  const lines: string[] = ["graph TD"];
  
  // Add nodes with styling
  for (const [id, node] of state.graph.nodes) {
    const label = sanitize(node.label);
    const style = getNodeStyle(node);
    lines.push(`    ${id}["${label}"]`);
    lines.push(`    style ${id} ${style}`);
  }
  
  // Add edges with styling
  for (const [id, edge] of state.graph.edges) {
    const arrow = getEdgeArrow(edge.type);
    const label = edge.type;
    lines.push(`    ${edge.source} ${arrow}|${label}| ${edge.target}`);
  }
  
  return lines.join("\n");
}

function getNodeStyle(node: NodeObject): string {
  const colors = {
    high_conf: "fill:#22c55e,stroke:#16a34a",      // Green
    mid_conf: "fill:#eab308,stroke:#ca8a04",       // Yellow
    low_conf: "fill:#ef4444,stroke:#dc2626",       // Red
    frontier: "fill:#3b82f6,stroke:#2563eb",       // Blue
    pruned: "fill:#6b7280,stroke:#4b5563"          // Gray
  };
  
  if (node.status === "pruned") return colors.pruned;
  if (node.status === "frontier") return colors.frontier;
  
  const conf = node.confidence ?? 0.5;
  if (conf > 0.7) return colors.high_conf;
  if (conf > 0.4) return colors.mid_conf;
  return colors.low_conf;
}

function getEdgeArrow(type: string): string {
  switch (type) {
    case "supports": return "-->"; 
    case "contradicts": return "-.->";  // Dashed
    case "cites": return "--->";
    default: return "-->";
  }
}
```

### 9.4 Key UI Interactions

| Interaction | Trigger | Effect |
|-------------|---------|--------|
| Start Investigation | Click "Investigate" | Clear graph, send query to MCP, show loading |
| Pause | Click "Pause" | Send pause signal, disable step animations |
| Resume | Click "Resume" | Send resume signal, continue from frontier |
| Node Click | Click graph node | Show metadata tooltip, add to context |
| Start Debate | Click "Start Debate" | Expand debate panel, pause main investigation |
| Next Round | Click "Next Round" | Trigger next debate turn |
| Export | Click "Export" | Download graph as JSON + PNG |
| Redirect | Submit redirect input | Add user hypothesis to state, reprioritize |

---

## 10. ⚡ Feature Specifications

### 10.1 Core Flow: Investigate Query

**Trigger:** User submits research question

**Flow:**
1. UI sends `{ type: "query", payload: { query: "..." } }` to MCP
2. Planner parses query, extracts entities
3. Planner sends clarification if needed (optional for demo)
4. Planner initializes Investigation State
5. Worker executes `search_nodes` for initial papers
6. Planner scores results, adds top-5 to frontier
7. **LOOP:**
   - Planner selects highest-priority frontier node
   - Planner dispatches traversal task to Worker
   - Worker executes, returns results or dead-end
   - Planner updates state, emits graph delta
   - UI renders incremental Mermaid update
   - Check termination conditions
8. Planner synthesizes findings
9. UI displays final summary + complete graph

**Mermaid rendering:** Update graph after each step, highlight active node, animate new edges appearing.

### 10.2 Hero Feature: Debate Simulator

**Trigger:** Agent finds `CONTRADICTS` edge OR user clicks "Start Debate" on any two papers

**Flow:**
1. Planner detects contradiction: Paper A ←contradicts→ Paper B
2. Planner emits `{ type: "contradiction_found", papers: [A, B] }`
3. UI shows alert: "⚠️ Contradiction Found! [Start Debate]"
4. User clicks "Start Debate"
5. UI sends `{ type: "debate", payload: { paper_a: "id", paper_b: "id" } }`
6. Planner fetches full metadata for both papers
7. Planner enters Moderator mode, Worker enters Debater mode
8. **DEBATE LOOP (3 rounds × 2 speakers = 6 turns):**
   - Moderator prompts Speaker A (round N)
   - Worker-as-A generates argument using paper content
   - UI renders A's argument in left card
   - User clicks "Next Round"
   - Moderator prompts Speaker B (round N)
   - Worker-as-B generates counter-argument
   - UI renders B's argument in right card
9. After 6 turns, Moderator generates synthesis
10. UI displays final verdict card
11. Planner integrates insights into Investigation State
12. User can resume main investigation

**UI Animation:**
- Cards slide in from sides
- Typing animation for arguments
- Edge between papers pulses during debate
- Synthesis appears as expanding card

### 10.3 Stretch: Axiom Buster

**Trigger:** User clicks "Trace Origins" on any claim/paper node

**Flow:**
1. UI sends `{ type: "axiom_bust", payload: { node_id: "..." } }`
2. Planner sets mode to "axiom_trace"
3. Worker recursively follows `cites` edges backward
4. At each step, select oldest cited paper
5. Continue until no more `cites` edges (found root)
6. Worker fetches root paper metadata
7. Planner checks:
   - Age of root paper
   - Citation count
   - Any `contradicts` edges in the chain
8. Planner generates assessment:
   - "Strong foundation" OR "Weak/outdated origin"
9. UI renders citation chain as vertical timeline
10. Highlight root paper with assessment badge

### 10.4 Stretch: Cross-Pollinator

**Trigger:** User clicks "Find Analogies" on a problem node

**Flow:**
1. Planner extracts structural pattern around node (2-hop subgraph)
2. Worker searches for similar patterns in other domains
3. Return candidate analogies with similarity scores
4. UI displays side-by-side subgraph comparison

---

## 11. 📊 Data Schemas

### 11.1 Investigation State (Full Example)

```json
{
  "session_id": "sess_2024_001",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:35:42Z",
  "status": "active",
  
  "query": {
    "original": "What is the evidence that gut microbiome affects depression?",
    "clarified": "Find evidence chains linking gut microbiome composition to clinical depression, including contradictory findings",
    "extracted_entities": ["gut microbiome", "depression", "mental health"],
    "research_goal": "evidence_evaluation"
  },
  
  "graph": {
    "visited_nodes": ["paper_2019_smith", "paper_2020_chen", "claim_microbiome_mood"],
    "frontier": [
      {
        "node_id": "paper_2021_johnson",
        "priority_score": 0.87,
        "reason": "High-impact paper (300 citations) that supports the microbiome-mood link",
        "depth": 2,
        "parent_node_id": "paper_2020_chen"
      },
      {
        "node_id": "paper_2022_lee",
        "priority_score": 0.65,
        "reason": "Recent paper that contradicts earlier findings",
        "depth": 3,
        "parent_node_id": "paper_2021_johnson"
      }
    ],
    "pruned_nodes": ["paper_2010_old", "claim_unrelated"],
    "edges_traversed": 12,
    "nodes": {
      "paper_2019_smith": {
        "id": "paper_2019_smith",
        "label": "Smith et al. (2019): Gut-Brain Axis Review",
        "type": "paper",
        "confidence": 0.85,
        "metadata": {
          "year": 2019,
          "citations": 450,
          "venue": "Nature Reviews"
        }
      }
    },
    "edges": {
      "edge_001": {
        "id": "edge_001",
        "source": "paper_2019_smith",
        "target": "claim_microbiome_mood",
        "type": "supports"
      }
    }
  },
  
  "reasoning": {
    "steps": [
      {
        "step_id": "step_001",
        "timestamp": "2024-01-15T10:31:00Z",
        "type": "search",
        "description": "Searched for papers on gut microbiome and depression",
        "node_ids_involved": [],
        "outcome": "Found 8 relevant papers, selected top 5 for exploration",
        "confidence_delta": 0.0
      },
      {
        "step_id": "step_002",
        "timestamp": "2024-01-15T10:32:15Z",
        "type": "traverse",
        "description": "Explored citations from Smith 2019",
        "node_ids_involved": ["paper_2019_smith"],
        "outcome": "Found 3 supporting papers and 1 contradicting paper",
        "confidence_delta": 0.15
      }
    ],
    "current_hypothesis": "The gut microbiome appears to influence mood through the vagus nerve, but effect sizes vary significantly across studies",
    "confidence": 0.72,
    "contradictions_found": [
      {
        "id": "contra_001",
        "node_a": "paper_2020_chen",
        "node_b": "paper_2022_lee",
        "edge_id": "edge_005",
        "summary": "Chen claims probiotic supplementation improves depression scores; Lee found no significant effect in larger RCT",
        "debate_available": true
      }
    ]
  },
  
  "findings": {
    "key_claims": [
      {
        "id": "claim_001",
        "text": "Gut microbiome composition correlates with depression severity",
        "confidence": 0.78,
        "supporting_papers": ["paper_2019_smith", "paper_2020_chen"],
        "contradicting_papers": ["paper_2022_lee"]
      }
    ],
    "evidence_chains": [
      {
        "id": "chain_001",
        "description": "Microbiome → SCFA production → Vagus nerve → Brain inflammation → Depression",
        "nodes": ["claim_scfa", "paper_2019_smith", "claim_inflammation"],
        "confidence": 0.65
      }
    ],
    "open_questions": [
      "What specific bacterial strains are most associated with mood changes?",
      "Is the relationship causal or correlational?"
    ]
  },
  
  "control": {
    "max_depth": 7,
    "max_nodes": 50,
    "saturation_threshold": 0.1,
    "user_paused": false,
    "user_redirections": []
  }
}
```

### 11.2 Graph Output (Mermaid-Ready)

```json
{
  "format": "mermaid",
  "direction": "TD",
  "nodes": [
    {
      "id": "paper_2019_smith",
      "label": "Smith 2019",
      "shape": "rectangle",
      "style": {
        "fill": "#22c55e",
        "stroke": "#16a34a",
        "stroke_width": 2
      }
    },
    {
      "id": "paper_2022_lee",
      "label": "Lee 2022",
      "shape": "rectangle",
      "style": {
        "fill": "#ef4444",
        "stroke": "#dc2626",
        "stroke_width": 2
      }
    },
    {
      "id": "claim_microbiome_mood",
      "label": "Microbiome affects mood",
      "shape": "rounded",
      "style": {
        "fill": "#eab308",
        "stroke": "#ca8a04",
        "stroke_width": 2
      }
    }
  ],
  "edges": [
    {
      "source": "paper_2019_smith",
      "target": "claim_microbiome_mood",
      "type": "supports",
      "style": "solid",
      "label": "supports"
    },
    {
      "source": "paper_2022_lee",
      "target": "claim_microbiome_mood",
      "type": "contradicts",
      "style": "dashed",
      "label": "contradicts"
    }
  ],
  "mermaid_code": "graph TD\n    paper_2019_smith[\"Smith 2019\"]\n    paper_2022_lee[\"Lee 2022\"]\n    claim_microbiome_mood(\"Microbiome affects mood\")\n    paper_2019_smith -->|supports| claim_microbiome_mood\n    paper_2022_lee -.->|contradicts| claim_microbiome_mood\n    style paper_2019_smith fill:#22c55e,stroke:#16a34a\n    style paper_2022_lee fill:#ef4444,stroke:#dc2626\n    style claim_microbiome_mood fill:#eab308,stroke:#ca8a04"
}
```

### 11.3 Debate Turn Schema

```json
{
  "debate_id": "debate_001",
  "round": 2,
  "total_rounds": 3,
  "current_speaker": "paper_b",
  "papers": {
    "paper_a": {
      "id": "paper_2020_chen",
      "title": "Probiotic Supplementation Reduces Depression: A 12-Week RCT",
      "authors": ["Chen, L.", "Wang, M."],
      "year": 2020,
      "claim": "Daily probiotic supplementation significantly reduced PHQ-9 depression scores compared to placebo"
    },
    "paper_b": {
      "id": "paper_2022_lee",
      "title": "No Effect of Probiotics on Depression: A Large Multi-Center Trial",
      "authors": ["Lee, K.", "Park, S."],
      "year": 2022,
      "claim": "In our larger, multi-center trial, we found no significant difference between probiotic and placebo groups"
    }
  },
  "turns": [
    {
      "round": 1,
      "speaker": "paper_a",
      "argument": "Our 12-week randomized controlled trial with 200 participants demonstrated a statistically significant reduction in PHQ-9 scores (mean difference: 3.2, p < 0.01). Participants receiving Lactobacillus rhamnosus showed 40% greater improvement than placebo.",
      "evidence_cited": ["paper_2020_chen"],
      "rhetorical_move": "claim",
      "confidence": 0.85
    },
    {
      "round": 1,
      "speaker": "paper_b",
      "argument": "While Chen's study shows promising results, our multi-center trial with 1,200 participants across 8 sites found no significant effect (mean difference: 0.8, p = 0.34). The smaller sample size and single-site design of the Chen study may have inflated effect sizes.",
      "evidence_cited": ["paper_2022_lee"],
      "rhetorical_move": "rebuttal",
      "confidence": 0.82
    },
    {
      "round": 2,
      "speaker": "paper_a",
      "argument": "Our study specifically selected participants with mild-to-moderate depression (PHQ-9: 10-19), whereas Lee's study included a broader range. The effect may be most pronounced in this specific population. Additionally, we used a higher CFU count (20 billion vs 10 billion).",
      "evidence_cited": ["paper_2020_chen"],
      "rhetorical_move": "rebuttal",
      "confidence": 0.75
    }
  ],
  "synthesis": null
}
```

---

## 12. 🛠️ Technical Stack

### 12.1 Technology Choices

| Layer | Technology | Rationale |
|-------|------------|-----------|
| **UI Framework** | React 18 + Vite | Fast dev, hot reload, modern |
| **Styling** | Tailwind CSS | Rapid prototyping, consistent design |
| **Graph Rendering** | Mermaid.js | Simple, declarative, good enough for demo |
| **State Management** | Zustand | Minimal boilerplate, easy to understand |
| **MCP Server** | Node.js + TypeScript | Type safety, familiar for most devs |
| **WebSocket** | ws (Node) + native (browser) | Real-time streaming updates |
| **LLM Client** | OpenAI SDK | GPT-4.1 + o3 access |
| **Mock Graph DB** | In-memory JSON | Fast for demo, easy to seed |

### 12.2 Project Structure

> **Note:** This section shows the current repo structure plus planned additions (marked with `# TODO`).
> When adding new files, follow this layout to avoid merge conflicts.

```
agentichackathon/
│
├── backend/                              # Python FastAPI backend
│   ├── agents/                           # 🤖 Agentic modules
│   │   ├── __init__.py
│   │   ├── paper_search.py               # OpenAlex/Semantic Scholar integration
│   │   ├── claim_extractor.py            # GPT-4o claim extraction
│   │   ├── citation_validator.py         # Claim verification logic
│   │   ├── planner.py                    # TODO: GPT-4.1 planner agent
│   │   ├── worker.py                     # TODO: o3 worker agent
│   │   └── debater.py                    # TODO: Debate simulator agent
│   │
│   ├── graph/                            # 📊 Knowledge graph logic
│   │   ├── __init__.py
│   │   └── knowledge_graph.py            # NetworkX graph builder
│   │
│   ├── models/                           # 📦 Pydantic schemas
│   │   ├── __init__.py
│   │   └── schemas.py                    # API request/response models
│   │
│   ├── tools/                            # TODO: MCP tool implementations
│   │   ├── search_nodes.py               # TODO: Semantic search tool
│   │   ├── traverse_edge.py              # TODO: Edge traversal tool
│   │   ├── get_metadata.py               # TODO: Node metadata tool
│   │   └── find_paths.py                 # TODO: Path finding tool
│   │
│   ├── prompts/                          # TODO: System prompts
│   │   ├── planner.md                    # TODO: Planner system prompt
│   │   ├── worker.md                     # TODO: Worker system prompt
│   │   └── debater.md                    # TODO: Debate mode prompt
│   │
│   ├── docs/
│   │   └── PRD_Scientific_Discovery_Agent.md  # This document
│   │
│   ├── main.py                           # FastAPI entry point
│   └── requirements.txt                  # Python dependencies
│
├── frontend/                             # Next.js frontend
│   ├── app/                              # App Router pages
│   │   ├── globals.css                   # Global styles (Tailwind)
│   │   ├── layout.tsx                    # Root layout
│   │   └── page.tsx                      # Main chat + graph UI
│   │
│   ├── components/                       # TODO: React components
│   │   ├── QueryBar.tsx                  # TODO: Research query input
│   │   ├── ReasoningTimeline.tsx         # TODO: Agent step display
│   │   ├── GraphViewer.tsx               # TODO: React Flow graph
│   │   ├── DebatePanel.tsx               # TODO: Debate simulator UI
│   │   └── FindingsPanel.tsx             # TODO: Key findings display
│   │
│   ├── hooks/                            # TODO: Custom React hooks
│   │   ├── useInvestigation.ts           # TODO: Investigation state
│   │   └── useWebSocket.ts               # TODO: Real-time updates
│   │
│   ├── public/                           # Static assets
│   ├── package.json
│   └── tsconfig.json
│
├── papers_from_hunren/                   # Hackathon reference materials
│   ├── welcome_text.txt
│   └── resources_text.txt
│
├── COLLABORATION.md                      # Team collaboration guide
├── WALKTHROUGH.md                        # Implementation walkthrough
└── README.md                             # TODO: Project readme
```

**File Ownership Guide (to avoid conflicts):**
- `backend/agents/` — Agent logic (coordinate before editing)
- `backend/graph/` — Graph operations (single owner recommended)
- `frontend/app/page.tsx` — Main UI (coordinate before editing)
- `frontend/components/` — Add new components freely
- `backend/tools/` — Add new MCP tools freely

### 12.3 Mock Graph Data

Seed the demo with ~50-100 nodes representing a controversy in a scientific field:

```json
{
  "nodes": [
    {
      "id": "paper_2019_smith",
      "type": "paper",
      "label": "Smith et al. (2019): Gut-Brain Axis Review",
      "metadata": {
        "year": 2019,
        "citations": 450,
        "authors": ["Smith, J.", "Brown, A."],
        "venue": "Nature Reviews Neuroscience",
        "abstract": "This review examines the bidirectional communication between the gut microbiome and the central nervous system..."
      }
    },
    {
      "id": "claim_microbiome_depression",
      "type": "claim",
      "label": "Gut microbiome composition influences depression risk",
      "metadata": {
        "first_proposed": 2015,
        "consensus_level": "emerging"
      }
    }
  ],
  "edges": [
    {
      "id": "edge_001",
      "source": "paper_2019_smith",
      "target": "claim_microbiome_depression",
      "type": "supports",
      "metadata": {
        "evidence_snippet": "Our meta-analysis of 12 studies found consistent associations between reduced microbial diversity and depression severity"
      }
    },
    {
      "id": "edge_002",
      "source": "paper_2022_lee",
      "target": "paper_2020_chen",
      "type": "contradicts",
      "metadata": {
        "evidence_snippet": "We were unable to replicate the findings of Chen et al. despite using similar methodology"
      }
    }
  ]
}
```

---

## 13. 🏃 Implementation Plan (2-Day Sprint)

### Day 1: Foundation (Hours 1-10)

| Hour | Dev 1 (Backend) | Dev 2 (Frontend) |
|------|-----------------|------------------|
| 1-2 | Set up MCP server, WebSocket | Set up React + Vite + Tailwind |
| 3-4 | Implement mock graph DB + basic tools | Build QueryBar + basic layout |
| 5-6 | Implement `search_nodes`, `traverse_edge` | Build MermaidGraph component |
| 7-8 | Wire up GPT-4.1 planner (basic loop) | Build ReasoningTimeline component |
| 9-10 | Test planner-worker handoff | WebSocket integration, streaming |

**Day 1 Checkpoint:** User can submit a query, see agent searching and traversing, watch graph build in real-time.

### Day 2: Polish + Hero Feature (Hours 11-20)

| Hour | Dev 1 (Backend) | Dev 2 (Frontend) |
|------|-----------------|------------------|
| 11-12 | Implement full Planner state machine | Build ContradictionAlert component |
| 13-14 | Implement Debate Simulator backend | Build DebatePanel UI |
| 15-16 | Refine prompts, handle edge cases | Polish graph styling, animations |
| 17-18 | Integration testing, bug fixes | Export feature, final styling |
| 19-20 | Demo prep, seed interesting data | Demo prep, screenshot/recording |

**Day 2 Checkpoint:** Full demo working—query → explore → find contradiction → debate → synthesis.

### Parallel Tasks

- Both devs seed mock graph data together (15 min)
- Both devs test end-to-end flow together (30 min)
- Both devs practice demo (30 min)

---

## 14. ⚠️ Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **LLM rate limits** | Medium | High | Implement backoff, cache responses, use mock in dev |
| **o3 latency** | High | Medium | Show loading states, stream partial results |
| **Mermaid rendering bugs** | Medium | Medium | Test with various graph sizes, have fallback text view |
| **Planner gets stuck** | Medium | High | Implement timeout + manual override, good error handling |
| **Not enough time** | High | High | Prioritize ruthlessly: Query → Graph → Debate, cut everything else |
| **Graph too complex** | Low | Medium | Limit to 50 nodes in demo, implement collapsing |

### Fallback Plan

If running behind schedule:
1. **Cut:** Axiom Buster, Cross-Pollinator
2. **Simplify:** Static graph instead of streaming updates
3. **Fake it:** Pre-compute one compelling demo path

---

## 15. 🔮 Future Considerations (Post-Hackathon)

- **Multi-user:** Authentication, separate sessions
- **Persistence:** Save/resume investigations
- **Real graph DB:** Neo4j or similar for production
- **Advanced viz:** React Force Graph for interactive exploration
- **Citations export:** BibTeX export of relevant papers
- **Collaborative:** Multiple users investigating same question
- **Fine-tuning:** Custom models trained on scientific reasoning

---

## 📝 Appendix A: Demo Script

### Setup (1 min)
- Show empty interface
- "This is ClaimGraph, a scientific discovery agent"

### Act 1: The Question (2 min)
- Type: "What's the evidence that gut microbiome affects depression?"
- Watch agent formulate plan
- See initial nodes appear

### Act 2: The Discovery (3 min)
- Agent traverses graph
- Nodes and edges appear in real-time
- Show reasoning timeline growing
- "Notice how it's following citation chains..."

### Act 3: The Contradiction (2 min)
- Agent finds contradicting papers
- Alert appears: "⚠️ Contradiction Found!"
- "Let's see what these papers have to say..."

### Act 4: The Debate (3 min)
- Click "Start Debate"
- Watch back-and-forth arguments
- "The agent is now simulating what a debate between these authors would look like..."
- Show synthesis: "The key disagreement is about sample size..."

### Closing (1 min)
- Show final graph with confidence colors
- "In 5 minutes, we've done what would take a researcher hours"
- Export graph for further analysis

---

## 🔐 Appendix B: Environment Variables

```bash
# .env.local (MCP Server)
OPENAI_API_KEY=sk-...
GPT4_MODEL=gpt-4.1
O3_MODEL=o3-deep-research
WEBSOCKET_PORT=8080

# .env.local (UI)
VITE_WS_URL=ws://localhost:8080
```

---

*Document Version: 1.0*  
*Last Updated: January 2025*  
*Authors: ClaimGraph Team*
