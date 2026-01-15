"""
Deep Research Agent
Uses o3-deep-research via the Responses API to conduct comprehensive research
over a scientific knowledge graph.
"""

import os
import json
import asyncio
from typing import Optional
from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=3600.0
)
RESEARCH_MODEL = "o3-deep-research"


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
    graph_context: str,
    max_tool_calls: int = 50
) -> DeepResearchResult:
    """
    Execute deep research using o3-deep-research model.
    
    Args:
        enriched_prompt: The enriched research prompt from the planner
        graph_context: Formatted knowledge graph context
        max_tool_calls: Maximum number of tool calls (controls cost/latency)
        
    Returns:
        DeepResearchResult with the research report and extracted information
    """
    import time
    start_time = time.time()
    
    full_input = f"""
{enriched_prompt}

{graph_context}

Please conduct thorough research on the above question using the provided knowledge graph data.
Ground all findings in the graph nodes and edges. Cite using [node_id] format.
"""
    
    try:
        response = await client.responses.create(
            model=RESEARCH_MODEL,
            input=full_input,
            instructions=DEEP_RESEARCH_INSTRUCTIONS,
            background=True,
            reasoning={"summary": "auto"},
            max_tool_calls=max_tool_calls,
        )
        
        while response.status not in ["completed", "failed", "cancelled"]:
            await asyncio.sleep(5)
            response = await client.responses.retrieve(response.id)
        
        if response.status != "completed":
            return DeepResearchResult(
                report=f"Research failed with status: {response.status}",
                research_goal=enriched_prompt[:200],
                status=response.status,
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
        
        report_text = response.output_text or ""
        
        citations = extract_citations_from_report(report_text)
        contradictions = extract_contradictions_from_report(report_text)
        confidence = extract_confidence_from_report(report_text)
        
        reasoning_summary = ""
        if hasattr(response, 'reasoning') and response.reasoning:
            reasoning_summary = getattr(response.reasoning, 'summary', '')
        
        return DeepResearchResult(
            report=report_text,
            research_goal=enriched_prompt[:200],
            citations=citations,
            contradictions=contradictions,
            confidence=confidence,
            reasoning_summary=reasoning_summary,
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
    graph_context: str,
    max_tool_calls: int = 30
) -> DeepResearchResult:
    """
    Execute deep research synchronously (no background mode).
    Use this for faster responses when background mode isn't needed.
    """
    import time
    start_time = time.time()
    
    full_input = f"""
{enriched_prompt}

{graph_context}

Please conduct thorough research on the above question using the provided knowledge graph data.
Ground all findings in the graph nodes and edges. Cite using [node_id] format.
"""
    
    try:
        response = await client.responses.create(
            model=RESEARCH_MODEL,
            input=full_input,
            instructions=DEEP_RESEARCH_INSTRUCTIONS,
            reasoning={"summary": "auto"},
            max_tool_calls=max_tool_calls,
        )
        
        report_text = response.output_text or ""
        
        citations = extract_citations_from_report(report_text)
        contradictions = extract_contradictions_from_report(report_text)
        confidence = extract_confidence_from_report(report_text)
        
        reasoning_summary = ""
        if hasattr(response, 'reasoning') and response.reasoning:
            reasoning_summary = getattr(response.reasoning, 'summary', '')
        
        return DeepResearchResult(
            report=report_text,
            research_goal=enriched_prompt[:200],
            citations=citations,
            contradictions=contradictions,
            confidence=confidence,
            reasoning_summary=reasoning_summary,
            status="completed",
            execution_time_ms=int((time.time() - start_time) * 1000)
        )
        
    except Exception as e:
        print(f"Error in execute_research_sync: {e}")
        return DeepResearchResult(
            report=f"Research error: {str(e)}",
            research_goal=enriched_prompt[:200],
            status="error",
            execution_time_ms=int((time.time() - start_time) * 1000)
        )


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
        if node_id.startswith('paper_') or node_id.startswith('claim_'):
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
        from planner import create_graph_context_prompt
        
        with open("../data/mock_graph.json", "r") as f:
            graph_data = json.load(f)
        
        graph_context = create_graph_context_prompt(graph_data)
        
        test_prompt = """
        I want to understand the relationship between gut microbiome and depression.
        Specifically:
        1. What is the evidence supporting this link?
        2. Are there contradicting studies?
        3. What are the proposed mechanisms (vagus nerve, inflammation, etc.)?
        4. What is the overall confidence in this relationship?
        
        Please provide a comprehensive analysis grounded in the knowledge graph data.
        """
        
        print("Executing deep research (this may take a while)...")
        result = await execute_research_sync(test_prompt, graph_context, max_tool_calls=20)
        
        print(f"\nStatus: {result.status}")
        print(f"Execution time: {result.execution_time_ms}ms")
        print(f"Confidence: {result.confidence}")
        print(f"Citations found: {len(result.citations)}")
        print(f"Contradictions found: {len(result.contradictions)}")
        print(f"\nReport preview:\n{result.report[:1000]}...")
    
    asyncio.run(test())
