"""
LangChain Agent for Knowledge Graph Research

Creates an agentic research system that can traverse the knowledge graph
using the 6 graph traversal tools.
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from tools import (
    search_claims,
    get_claim_support,
    trace_provenance,
    get_related_claims,
    find_contradictions,
    get_graph_statistics,
)

RESEARCH_SYSTEM_PROMPT = """You are a scientific research analyst conducting deep research over a knowledge graph.

Your task is to thoroughly analyze claims and their relationships to produce comprehensive research reports.

## Your Capabilities
- Search for claims by semantic/keyword queries
- Traverse support/contradict relationships between claims
- Trace provenance chains to ground truths
- Find contradictions and conflicting evidence
- Discover related claims by metadata

## Research Process
1. Start by searching for claims relevant to the research question
2. For each promising claim, trace its provenance to assess grounding
3. Find supporting and contradicting evidence
4. Identify related claims for broader context
5. Synthesize findings into a comprehensive report

## Critical Rules
- Ground all findings in actual claims from the graph
- Cite claim IDs when referencing specific claims (format: [claim_id])
- Report contradictions honestly
- Assess confidence based on evidence strength
- Distinguish between correlation and causation
- If a claim has no path to ground truth, note it as ungrounded
- When tracing provenance, follow chains until you reach ground truths or unsupported foundations

## Output Format
Structure your research report with:
- Executive Summary: 2-3 sentences summarizing key findings
- Research Question: Restate what was investigated
- Key Findings: Bullet points with claim citations
- Evidence Analysis: Detailed examination with claim IDs
- Contradictions Found: List any conflicting claims
- Citation Chain Analysis: Trace how claims are supported
- Confidence Assessment: Overall confidence (0-1) with justification
- Conclusions: Final synthesis
- Limitations: What the graph doesn't tell us"""


def create_kg_research_agent(model_name: str = "gpt-5.2"):
    """Create a LangChain agent for knowledge graph research.
    
    Args:
        model_name: Name of the OpenAI model to use (default: gpt-5.2)
        
    Returns:
        LangChain agent configured with graph traversal tools
    """
    model = ChatOpenAI(
        model=model_name,
        temperature=0.1,
        timeout=360.0,
    )
    
    tools = [
        search_claims,
        get_claim_support,
        trace_provenance,
        get_related_claims,
        find_contradictions,
        get_graph_statistics,
    ]
    
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
    )
    
    return agent
