"""
Deep Research Planner Agent
Uses GPT-4.1 via the Responses API to clarify user queries and enrich prompts
for the o3-deep-research model.
"""

import os
import json
from typing import Optional
from openai import AsyncOpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PLANNER_MODEL = "gpt-4.1"


class ClarificationQuestion(BaseModel):
    """A clarification question for the user"""
    question: str
    options: list[str] = []
    required: bool = False


class ClarificationResponse(BaseModel):
    """Response from clarification step"""
    needs_clarification: bool
    questions: list[ClarificationQuestion] = []
    reasoning: str = ""


class EnrichedPrompt(BaseModel):
    """Enriched prompt ready for deep research"""
    enriched_prompt: str
    research_goal: str
    key_dimensions: list[str] = []
    expected_format: str = ""


CLARIFICATION_INSTRUCTIONS = """
You are talking to a user who is asking for a scientific research task to be conducted. 
Your job is to gather more information from the user to successfully complete the task.

GUIDELINES:
- Be concise while gathering all necessary information
- Make sure to gather all the information needed to carry out the research task in a concise, well-structured manner
- Use bullet points or numbered lists if appropriate for clarity
- Don't ask for unnecessary information, or information that the user has already provided
- Focus on scientific research queries about claims, evidence, and relationships between papers

For scientific research queries, you may want to clarify:
1. The specific aspect they want to investigate (causation vs correlation, mechanisms, contradictions)
2. Time range of papers to consider (recent only, or historical context)
3. What outcome they're looking for (verification, synthesis, finding contradictions)
4. Any specific domains or subfields to focus on

IMPORTANT: 
- Do NOT conduct any research yourself, just gather information
- If the query is already specific enough, respond with {"needs_clarification": false}
- Only ask 2-3 questions maximum
- Provide multiple-choice options where possible to make answering easier

Output your response as JSON with this structure:
{
  "needs_clarification": true/false,
  "questions": [
    {
      "question": "The question text",
      "options": ["Option 1", "Option 2", "Option 3"],
      "required": true/false
    }
  ],
  "reasoning": "Brief explanation of why these questions are needed"
}
"""


ENRICHMENT_INSTRUCTIONS = """
You will be given a research task by a user. Your job is to produce a detailed set of
instructions for a deep research agent that will complete the task. Do NOT complete the
task yourself, just provide instructions on how to complete it.

The research will be conducted over a Scientific Knowledge Graph containing:
- Paper nodes (scientific publications with abstracts, authors, years)
- Claim nodes (specific assertions made in papers)
- Edges: "supports", "contradicts", "cites" relationships

GUIDELINES:

1. **Maximize Specificity and Detail**
- Include all known user preferences and explicitly list key attributes or dimensions to consider
- It is of utmost importance that all details from the user are included in the instructions

2. **Fill in Unstated But Necessary Dimensions as Open-Ended**
- If certain attributes are essential for a meaningful output but the user has not provided them,
  explicitly state that they are open-ended or default to no specific constraint

3. **Avoid Unwarranted Assumptions**
- If the user has not provided a particular detail, do not invent one
- Instead, state the lack of specification and guide the researcher to treat it as flexible

4. **Use the First Person**
- Phrase the request from the perspective of the user

5. **Structure the Output**
- Request a structured report with clear sections
- Include: Executive Summary, Key Findings, Evidence Analysis, Contradictions Found, Conclusions

6. **Scientific Rigor**
- Emphasize grounding all claims in the graph data
- Request citations using [node_id] format
- Ask for confidence levels where appropriate
- Request identification of any contradicting evidence

7. **Sources**
- Prefer primary sources (original papers) over secondary reviews
- Request that all claims be traceable to specific nodes in the graph

Output your enriched prompt as JSON with this structure:
{
  "enriched_prompt": "The full, detailed prompt for the researcher",
  "research_goal": "One-sentence summary of what we're trying to find",
  "key_dimensions": ["dimension1", "dimension2", ...],
  "expected_format": "Description of expected output format"
}
"""


async def clarify_query(query: str) -> ClarificationResponse:
    """
    Use GPT-4.1 to determine if clarification is needed and generate questions.
    
    Args:
        query: The user's research query
        
    Returns:
        ClarificationResponse with questions if needed
    """
    try:
        response = await client.responses.create(
            model=PLANNER_MODEL,
            input=f"User research query: {query}",
            instructions=CLARIFICATION_INSTRUCTIONS,
        )
        
        output_text = response.output_text
        
        try:
            parsed = json.loads(output_text)
            return ClarificationResponse(
                needs_clarification=parsed.get("needs_clarification", False),
                questions=[
                    ClarificationQuestion(
                        question=q.get("question", ""),
                        options=q.get("options", []),
                        required=q.get("required", False)
                    )
                    for q in parsed.get("questions", [])
                ],
                reasoning=parsed.get("reasoning", "")
            )
        except json.JSONDecodeError:
            return ClarificationResponse(
                needs_clarification=False,
                reasoning="Query appears sufficiently detailed for research."
            )
            
    except Exception as e:
        print(f"Error in clarify_query: {e}")
        return ClarificationResponse(
            needs_clarification=False,
            reasoning=f"Proceeding without clarification due to error: {str(e)}"
        )


async def enrich_prompt(
    query: str,
    clarifications: dict = None,
    graph_summary: dict = None
) -> EnrichedPrompt:
    """
    Use GPT-4.1 to transform user query into detailed research instructions.
    
    Args:
        query: The user's research query
        clarifications: Optional dict of clarification answers
        graph_summary: Optional summary of available graph data
        
    Returns:
        EnrichedPrompt with detailed instructions for deep research
    """
    context_parts = [f"User research query: {query}"]
    
    if clarifications:
        context_parts.append(f"\nUser clarifications: {json.dumps(clarifications, indent=2)}")
    
    if graph_summary:
        context_parts.append(f"\nAvailable graph data summary: {json.dumps(graph_summary, indent=2)}")
    
    input_text = "\n".join(context_parts)
    
    try:
        response = await client.responses.create(
            model=PLANNER_MODEL,
            input=input_text,
            instructions=ENRICHMENT_INSTRUCTIONS,
        )
        
        output_text = response.output_text
        
        try:
            parsed = json.loads(output_text)
            return EnrichedPrompt(
                enriched_prompt=parsed.get("enriched_prompt", query),
                research_goal=parsed.get("research_goal", "Investigate the research question"),
                key_dimensions=parsed.get("key_dimensions", []),
                expected_format=parsed.get("expected_format", "Structured report")
            )
        except json.JSONDecodeError:
            return EnrichedPrompt(
                enriched_prompt=output_text if output_text else query,
                research_goal="Investigate the research question",
                key_dimensions=[],
                expected_format="Structured report"
            )
            
    except Exception as e:
        print(f"Error in enrich_prompt: {e}")
        return EnrichedPrompt(
            enriched_prompt=query,
            research_goal="Investigate the research question",
            key_dimensions=[],
            expected_format="Structured report"
        )


def create_graph_context_prompt(graph_data: dict) -> str:
    """
    Format the knowledge graph data as context for the deep research prompt.
    
    Args:
        graph_data: The mock graph JSON data
        
    Returns:
        Formatted string to include in the research prompt
    """
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    
    papers = [n for n in nodes if n.get("type") == "paper"]
    claims = [n for n in nodes if n.get("type") == "claim"]
    
    supports_edges = [e for e in edges if e.get("type") == "supports"]
    contradicts_edges = [e for e in edges if e.get("type") == "contradicts"]
    cites_edges = [e for e in edges if e.get("type") == "cites"]
    
    context = f"""
## Scientific Knowledge Graph Context

You have access to a curated scientific knowledge graph. You MUST ground ALL claims in this data.
Do NOT hallucinate or invent information not present in the graph.

### Graph Statistics
- Papers: {len(papers)}
- Claims: {len(claims)}
- Support relationships: {len(supports_edges)}
- Contradiction relationships: {len(contradicts_edges)}
- Citation relationships: {len(cites_edges)}

### Papers (Nodes)
{json.dumps(papers, indent=2)}

### Claims (Nodes)
{json.dumps(claims, indent=2)}

### Relationships (Edges)
{json.dumps(edges, indent=2)}

### Citation Format
When citing evidence, use the format: [node_id]
Example: "The gut microbiome influences mood through the vagus nerve [paper_2018_wilson]"

### Instructions
1. ONLY report findings that can be traced to nodes in this graph
2. When you find contradictions, cite BOTH papers with their node IDs
3. Assess confidence based on the number of supporting vs contradicting edges
4. If the graph doesn't contain information about something, say "No data available in graph"
"""
    return context


if __name__ == "__main__":
    import asyncio
    
    async def test():
        print("Testing clarify_query...")
        result = await clarify_query("Tell me about gut bacteria and depression")
        print(f"Needs clarification: {result.needs_clarification}")
        if result.questions:
            for q in result.questions:
                print(f"  Q: {q.question}")
                print(f"     Options: {q.options}")
        
        print("\nTesting enrich_prompt...")
        enriched = await enrich_prompt(
            "What is the evidence that gut microbiome affects depression?",
            clarifications={"focus": "causation", "time_range": "2018-2024"}
        )
        print(f"Research goal: {enriched.research_goal}")
        print(f"Key dimensions: {enriched.key_dimensions}")
        print(f"Enriched prompt preview: {enriched.enriched_prompt[:500]}...")
    
    asyncio.run(test())
