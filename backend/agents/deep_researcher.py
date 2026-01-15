"""
Deep Research Agent
Uses LangChain agent with graph traversal tools to conduct comprehensive research
over a scientific knowledge graph.
"""

import os
import json
import asyncio
import time
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.messages import HumanMessage

from agents.kg_research_agent import create_kg_research_agent

load_dotenv()


class ResearchCitation(BaseModel):
    """A citation from the research report"""
    node_id: str
    node_type: str
    label: str
    context: str = ""


class ContradictionFound(BaseModel):
    """A contradiction identified in the research"""
    paper_a_id: str
    paper_a_label: str
    paper_b_id: str
    paper_b_label: str
    summary: str
    edge_id: Optional[str] = None


class DeepResearchResult(BaseModel):
    """Result from deep research execution"""
    report: str
    research_goal: str
    citations: list[ResearchCitation] = []
    contradictions: list[ContradictionFound] = []
    confidence: float = 0.0
    reasoning_summary: str = ""
    status: str = "completed"
    tokens_used: int = 0
    execution_time_ms: int = 0


DEEP_RESEARCH_INSTRUCTIONS = """
You are a scientific research analyst conducting deep research over a curated knowledge graph.
Your task is to thoroughly analyze the provided graph data and produce a comprehensive research report.

## Your Capabilities
- Analyze relationships between papers and claims
- Identify supporting and contradicting evidence
- Trace citation chains to assess claim validity
- Synthesize findings into actionable insights

## Critical Rules
1. **Grounding**: ONLY report information present in the provided graph. Never hallucinate.
2. **Citations**: Always cite sources using [node_id] format (e.g., [paper_2019_smith])
3. **Contradictions**: Actively search for and report contradicting evidence
4. **Confidence**: Assess confidence based on evidence strength and consensus

## Output Format
Structure your report with these sections:

### Executive Summary
2-3 sentences summarizing the key findings

### Research Question
Restate what we're investigating

### Key Findings
Bullet points of main discoveries, each with citations

### Evidence Analysis
Detailed examination of supporting evidence with node citations

### Contradictions Found
List any conflicting claims or papers, explaining the nature of disagreement

### Citation Chain Analysis
Trace how claims are supported through the graph

### Confidence Assessment
Overall confidence level (0-1) with justification

### Conclusions
Final synthesis and recommendations

### Limitations
What the graph data doesn't tell us

Remember: Scientific rigor is paramount. Distinguish between correlation and causation.
Report uncertainty honestly. Cite everything.
"""


async def execute_research(
    enriched_prompt: str,
    graph_context: str = "",
    max_tool_calls: int = 50
) -> DeepResearchResult:
    """
    Execute deep research using LangChain agent with graph traversal tools.
    
    Args:
        enriched_prompt: The enriched research prompt from the planner
        graph_context: Not used (kept for API compatibility) - agent uses tools instead
        max_tool_calls: Maximum number of tool calls (controls cost/latency)
        
    Returns:
        DeepResearchResult with the research report and extracted information
    """
    start_time = time.time()
    
    try:
        agent = create_kg_research_agent()
        
        # Invoke agent with the research prompt
        # The agent will automatically use tools to traverse the graph
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=enriched_prompt)]
        })
        
        # Extract the final message content as the report
        # The result is a state dict with messages list
        messages = result.get("messages", [])
        report_text = ""
        
        # Find the last AI message (final answer)
        for msg in reversed(messages):
            # Check if it's an AIMessage or has content
            if hasattr(msg, 'content') and msg.content:
                content = msg.content
                # Handle both string and list content
                if isinstance(content, str):
                    report_text = content
                elif isinstance(content, list):
                    # Extract text from content blocks
                    text_parts = [item.get('text', '') if isinstance(item, dict) else str(item) 
                                 for item in content if item]
                    report_text = '\n'.join(text_parts)
                else:
                    report_text = str(content)
                break
        
        if not report_text:
            # Fallback: concatenate all message contents
            all_content = []
            for msg in messages:
                if hasattr(msg, 'content') and msg.content:
                    all_content.append(str(msg.content))
            report_text = '\n\n'.join(all_content) if all_content else "Research completed but no report generated."
        
        citations = extract_citations_from_report(report_text)
        contradictions = extract_contradictions_from_report(report_text)
        confidence = extract_confidence_from_report(report_text)
        
        return DeepResearchResult(
            report=report_text,
            research_goal=enriched_prompt[:200],
            citations=citations,
            contradictions=contradictions,
            confidence=confidence,
            reasoning_summary="",  # LangChain doesn't provide reasoning summary
            status="completed",
            execution_time_ms=int((time.time() - start_time) * 1000)
        )
        
    except Exception as e:
        print(f"Error in execute_research: {e}")
        import traceback
        traceback.print_exc()
        return DeepResearchResult(
            report=f"Research error: {str(e)}",
            research_goal=enriched_prompt[:200],
            status="error",
            execution_time_ms=int((time.time() - start_time) * 1000)
        )


async def execute_research_sync(
    enriched_prompt: str,
    graph_context: str = "",
    max_tool_calls: int = 30
) -> DeepResearchResult:
    """
    Execute deep research synchronously.
    Alias for execute_research (LangChain agent is always synchronous).
    """
    return await execute_research(enriched_prompt, graph_context, max_tool_calls)


def extract_citations_from_report(report: str) -> list[ResearchCitation]:
    """
    Extract all [node_id] citations from the research report.
    """
    import re
    citations = []
    seen = set()
    
    pattern = r'\[([^\]]+)\]'
    matches = re.findall(pattern, report)
    
    for node_id in matches:
        # Match claim IDs (format: claim_xxxxx) or any ID in brackets
        if node_id.startswith('claim_') or node_id.startswith('paper_') or 'claim_' in node_id.lower():
            if node_id not in seen:
                seen.add(node_id)
                node_type = "paper" if node_id.startswith('paper_') else "claim"
                citations.append(ResearchCitation(
                    node_id=node_id,
                    node_type=node_type,
                    label=node_id.replace('_', ' ').title()
                ))
    
    return citations


def extract_contradictions_from_report(report: str) -> list[ContradictionFound]:
    """
    Extract contradiction information from the report.
    Looks for patterns like "contradicts" near paper citations.
    """
    import re
    contradictions = []
    
    contradiction_section = ""
    if "### Contradictions Found" in report:
        start = report.index("### Contradictions Found")
        end = report.find("###", start + 1)
        if end == -1:
            end = len(report)
        contradiction_section = report[start:end]
    
    if contradiction_section:
        paper_pattern = r'\[paper_[^\]]+\]'
        papers = re.findall(paper_pattern, contradiction_section)
        
        for i in range(0, len(papers) - 1, 2):
            if i + 1 < len(papers):
                paper_a = papers[i].strip('[]')
                paper_b = papers[i + 1].strip('[]')
                contradictions.append(ContradictionFound(
                    paper_a_id=paper_a,
                    paper_a_label=paper_a.replace('_', ' ').title(),
                    paper_b_id=paper_b,
                    paper_b_label=paper_b.replace('_', ' ').title(),
                    summary=f"Contradicting findings between {paper_a} and {paper_b}"
                ))
    
    return contradictions


def extract_confidence_from_report(report: str) -> float:
    """
    Extract the confidence assessment from the report.
    """
    import re
    
    patterns = [
        r'confidence[:\s]+(\d+\.?\d*)',
        r'confidence level[:\s]+(\d+\.?\d*)',
        r'overall confidence[:\s]+(\d+\.?\d*)',
        r'\((\d+\.?\d*)\s*confidence\)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, report.lower())
        if match:
            try:
                conf = float(match.group(1))
                if conf > 1:
                    conf = conf / 100
                return min(max(conf, 0), 1)
            except ValueError:
                continue
    
    return 0.7


if __name__ == "__main__":
    import asyncio
    
    async def test():
        test_prompt = """
        I want to understand the relationship between gut microbiome and depression.
        Specifically:
        1. What is the evidence supporting this link?
        2. Are there contradicting studies?
        3. What are the proposed mechanisms (vagus nerve, inflammation, etc.)?
        4. What is the overall confidence in this relationship?
        
        Please provide a comprehensive analysis grounded in the knowledge graph data.
        Use the available tools to search for claims, trace provenance, and find contradictions.
        """
        
        print("Executing deep research with LangChain agent (this may take a while)...")
        result = await execute_research_sync(test_prompt, max_tool_calls=20)
        
        print(f"\nStatus: {result.status}")
        print(f"Execution time: {result.execution_time_ms}ms")
        print(f"Confidence: {result.confidence}")
        print(f"Citations found: {len(result.citations)}")
        print(f"Contradictions found: {len(result.contradictions)}")
        print(f"\nReport preview:\n{result.report[:1000]}...")
    
    asyncio.run(test())
