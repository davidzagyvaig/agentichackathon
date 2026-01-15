"""
Claim Extractor for Knowledge Graph

Uses GPT-4o with Responses API to extract, classify, and detect relationships
between scientific claims directly from PDF files.

Features:
- JOINT EXTRACTION: Claims + Classification + Relationships in ONE API call
- STRUCTURED OUTPUTS: SDK's responses.parse() handles Pydantic conversion automatically
- PDF SUPPORT: Direct PDF URL processing (no download needed)

Implements PRD Section 6: Claim Extraction.
"""

import asyncio
import base64
from typing import Optional
from datetime import datetime
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field
import os

from models.kg_schemas import (
    ClaimNode,
    ClaimEdge,
    SourcePaperMetadata,
    ExtractionMetadata,
    generate_claim_id,
)


# ============================================================================
# PROMPT TEMPLATES (from PRD Section 6)
# ============================================================================

JOINT_EXTRACTION_SYSTEM_PROMPT = """# SYSTEM PROMPT: Scientific Claim & Relationship Extractor

You are performing JOINT EXTRACTION of scientific claims and their relationships from academic papers.
This is a SINGLE-PASS extraction - extract claims, classify them, AND identify relationships all at once.

## TASK 1: Claim Extraction

### What IS a claim:
- A specific assertion that can be true or false
- A statement of fact, finding, or conclusion
- A quantitative result (e.g., "X increased by 40%")
- A causal relationship (e.g., "A causes B")
- A methodological assertion (e.g., "Method X is more accurate than Y")

### What is NOT a claim:
- Background context or definitions
- Descriptions of what the paper will do
- Acknowledgments or administrative text
- Vague statements without specific content

## TASK 2: Claim Classification

For each claim, classify into one of three types:

### ground_truth
A claim requiring no further justification:
- Mathematical facts
- Definitions
- Established scientific laws
**BE STRICT**: Only use if NO scientist would dispute it.

### empirical
A claim based on testable evidence (MOST claims are this):
- Experimental results
- Observational findings
- Statistical conclusions

### unsupported
A claim stated without evidence:
- "As is well known..."
- Appeals to authority without citation

## TASK 3: Relationship Detection

Identify relationships BETWEEN claims you extracted:

### supports
Claim A provides evidence that Claim B is true:
- A's data confirms B's assertion
- A is a specific instance validating B's general statement

### contradicts
Claim A provides evidence that Claim B is false:
- A's data conflicts with B
- A's conclusion is incompatible with B

## OUTPUT FORMAT

Return a SINGLE JSON object with this structure:
{
  "claims": [
    {
      "id": 0,
      "original_text": "Exact quote from paper",
      "rephrased": "Clear, concise claim",
      "section": "abstract|introduction|methods|results|discussion|conclusion",
      "classification": "empirical|ground_truth|unsupported",
      "confidence": 0.0-1.0,
      "importance": "high|medium|low"
    }
  ],
  "relationships": [
    {
      "source_id": 0,
      "target_id": 1,
      "type": "supports|contradicts",
      "weight": 0.0-1.0,
      "reasoning": "Brief explanation of why this relationship exists"
    }
  ]
}

## GUIDELINES:
- Extract 10-20 important claims per paper
- Rephrase to remove hedging ("may", "might", "could")
- Keep quantitative details (numbers, p-values, effect sizes)
- Relationships are DIRECTIONAL: source provides evidence for/against target
- Only include relationships with weight >= 0.5
- Focus on claims that could be verified or disputed by other research"""


# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================

class ExtractedClaim(BaseModel):
    """A single extracted claim from a scientific paper."""
    id: int = Field(description="Sequential ID for this claim (0, 1, 2, ...)")
    original_text: str = Field(description="Exact quote from paper")
    rephrased: str = Field(description="Clear, concise claim without hedging")
    section: str = Field(description="Section where claim was found: abstract, introduction, methods, results, discussion, or conclusion")
    classification: str = Field(description="Claim type: empirical, ground_truth, or unsupported")
    confidence: float = Field(description="Confidence in classification (0.0-1.0)")
    importance: str = Field(description="Importance level: high, medium, or low")


class ExtractedRelationship(BaseModel):
    """A relationship between two claims."""
    source_id: int = Field(description="ID of the source claim (provides evidence)")
    target_id: int = Field(description="ID of the target claim (receives evidence)")
    type: str = Field(description="Relationship type: supports or contradicts")
    weight: float = Field(description="Relationship strength (0.0-1.0)")
    reasoning: str = Field(description="Brief explanation of why this relationship exists")


class JointExtractionResult(BaseModel):
    """Complete result of joint extraction from a paper."""
    claims: list[ExtractedClaim] = Field(description="List of extracted claims")
    relationships: list[ExtractedRelationship] = Field(description="List of relationships between claims")


# Simplified schema for abstract-only extraction (no relationships)
class AbstractExtractionResult(BaseModel):
    """Result of extraction from abstract only."""
    claims: list[ExtractedClaim] = Field(description="List of extracted claims")


class ClaimExtractor:
    """
    Extracts scientific claims from paper PDFs using OpenAI Responses API.
    
    Features:
    - JOINT EXTRACTION: Claims + Classification + Relationships in ONE API call
    - STRUCTURED OUTPUTS: SDK's responses.parse() handles Pydantic conversion automatically
    - PDF SUPPORT: Direct PDF URL processing (no download/encoding needed)
    
    Primary method: extract_claims_from_pdf() - JOINT extraction with structured output.
    Fallback: extract_claims_from_abstract() - For when PDF is unavailable.
    """
    
    MODEL = "gpt-4o-2024-08-06"  # Model with structured output support
    FALLBACK_MODEL = "gpt-4o-mini"  # Use mini for abstract fallback
    PROMPT_VERSION = "5.0"  # Responses API with responses.parse() and file_url
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the claim extractor.
        
        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.sync_client = OpenAI(api_key=self.api_key)
    
    async def extract_claims_from_pdf(
        self,
        pdf_url: str,
        paper_metadata: dict,
        max_claims: int = 20
    ) -> tuple[list[ClaimNode], list[ClaimEdge]]:
        """
        JOINT EXTRACTION: Extract claims AND relationships from PDF URL in a single call.
        
        Uses the Responses API with:
        - PDF file_url (no download/encoding needed)
        - responses.parse() with text_format=Pydantic for automatic parsing
        
        Args:
            pdf_url: URL to the PDF file (e.g., https://arxiv.org/pdf/2301.00001.pdf)
            paper_metadata: Paper metadata (arxiv_id, title, authors, etc.)
            max_claims: Maximum claims to extract
            
        Returns:
            Tuple of (claims, edges) - both extracted in one API call with guaranteed schema.
        """
        # Run in thread pool since responses API is sync
        loop = asyncio.get_event_loop()
        claims, edges = await loop.run_in_executor(
            None,
            self._extract_with_responses_api,
            pdf_url,
            paper_metadata,
            max_claims
        )
        
        return claims, edges
    
    async def extract_claims_from_pdf_bytes(
        self,
        pdf_bytes: bytes,
        paper_metadata: dict,
        max_claims: int = 20
    ) -> tuple[list[ClaimNode], list[ClaimEdge]]:
        """
        Fallback: Extract claims from PDF bytes when URL is not available.
        
        Uses base64 encoding with file_data parameter.
        
        Args:
            pdf_bytes: Raw PDF file bytes
            paper_metadata: Paper metadata (arxiv_id, title, authors, etc.)
            max_claims: Maximum claims to extract
            
        Returns:
            Tuple of (claims, edges).
        """
        loop = asyncio.get_event_loop()
        claims, edges = await loop.run_in_executor(
            None,
            self._extract_with_base64,
            pdf_bytes,
            paper_metadata,
            max_claims
        )
        
        return claims, edges
    
    def _extract_with_responses_api(
        self,
        pdf_url: str,
        paper_metadata: dict,
        max_claims: int
    ) -> tuple[list[ClaimNode], list[ClaimEdge]]:
        """
        JOINT EXTRACTION using responses.parse() with file_url.
        
        Uses the simplified SDK approach:
        - file_url: pass PDF URL directly (no download needed)
        - text_format: Pydantic model (SDK handles schema conversion)
        - output_parsed: Already parsed as Pydantic object
        """
        try:
            # Build the user prompt
            user_prompt = f"""Perform JOINT EXTRACTION on this scientific paper.

Paper: "{paper_metadata.get('title', 'Unknown')}"
Authors: {', '.join(paper_metadata.get('authors', [])[:5])}
Year: {paper_metadata.get('year', 'Unknown')}
arXiv ID: {paper_metadata.get('arxiv_id', 'Unknown')}

## YOUR TASK (Single Pass):

1. **Extract up to {max_claims} important claims** from all sections
2. **Classify each claim** as ground_truth, empirical, or unsupported  
3. **Detect relationships** between claims (supports/contradicts)

Focus on:
- Quantitative findings (numbers, p-values, effect sizes)
- Causal relationships ("X causes Y")
- Comparative statements ("X outperforms Y")
- Methodological assertions

Important:
- Assign sequential IDs to claims (0, 1, 2, ...)
- Use these IDs when defining relationships
- Only include relationships with weight >= 0.5"""

            # Call responses.parse() with file_url and text_format=Pydantic
            response = self.sync_client.responses.parse(
                model=self.MODEL,
                input=[
                    {"role": "system", "content": JOINT_EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_file", "file_url": pdf_url},
                            {"type": "input_text", "text": user_prompt},
                        ]
                    }
                ],
                text_format=JointExtractionResult,  # SDK handles Pydantic conversion
            )
            
            # response.output_parsed is already a JointExtractionResult Pydantic object
            result = response.output_parsed
            
            return self._convert_pydantic_to_nodes_and_edges(result, paper_metadata)
            
        except Exception as e:
            print(f"   Error in Responses API extraction: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def _extract_with_base64(
        self,
        pdf_bytes: bytes,
        paper_metadata: dict,
        max_claims: int
    ) -> tuple[list[ClaimNode], list[ClaimEdge]]:
        """
        Fallback: Use base64 file_data when URL is not available.
        """
        try:
            # Encode PDF as base64
            pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
            
            # Build prompt
            user_prompt = f"""Perform JOINT EXTRACTION on this scientific paper.

Paper: "{paper_metadata.get('title', 'Unknown')}"
Authors: {', '.join(paper_metadata.get('authors', [])[:5])}
Year: {paper_metadata.get('year', 'Unknown')}
arXiv ID: {paper_metadata.get('arxiv_id', 'Unknown')}

Extract up to {max_claims} important claims, classify each, and detect relationships."""

            # Call responses.parse() with base64 file_data
            response = self.sync_client.responses.parse(
                model=self.MODEL,
                input=[
                    {"role": "system", "content": JOINT_EXTRACTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "filename": f"{paper_metadata.get('arxiv_id', 'paper')}.pdf",
                                "file_data": f"data:application/pdf;base64,{pdf_base64}"
                            },
                            {"type": "input_text", "text": user_prompt},
                        ]
                    }
                ],
                text_format=JointExtractionResult,
            )
            
            result = response.output_parsed
            
            return self._convert_pydantic_to_nodes_and_edges(result, paper_metadata)
            
        except Exception as e:
            print(f"   Error in base64 extraction: {e}")
            import traceback
            traceback.print_exc()
            return [], []
    
    def _convert_pydantic_to_nodes_and_edges(
        self,
        result: JointExtractionResult,
        paper_metadata: dict
    ) -> tuple[list[ClaimNode], list[ClaimEdge]]:
        """
        Convert Pydantic JointExtractionResult to ClaimNode and ClaimEdge objects.
        """
        claim_nodes = []
        id_to_claim = {}
        
        for claim_data in result.claims:
            rephrased = claim_data.rephrased or claim_data.original_text
            if not rephrased:
                continue
            
            classification = claim_data.classification
            if classification not in ['empirical', 'ground_truth', 'unsupported']:
                classification = 'empirical'
            
            source_paper = SourcePaperMetadata(
                arxiv_id=paper_metadata.get('arxiv_id'),
                title=paper_metadata.get('title', 'Unknown'),
                authors=paper_metadata.get('authors', []),
                year=paper_metadata.get('year'),
                venue=paper_metadata.get('venue') or paper_metadata.get('journal_ref'),
                section=claim_data.section or 'unknown',
                url=paper_metadata.get('url'),
                doi=paper_metadata.get('doi'),
            )
            
            claim_id = generate_claim_id(rephrased)
            
            claim_node = ClaimNode(
                id=claim_id,
                text=rephrased,
                original_text=claim_data.original_text,
                type=classification,
                confidence=float(claim_data.confidence),
                source_paper=source_paper,
                extraction=ExtractionMetadata(
                    model=self.MODEL,
                    timestamp=datetime.utcnow().isoformat(),
                    prompt_version=self.PROMPT_VERSION,
                    context_snippet=(claim_data.original_text or '')[:500],
                ),
            )
            claim_nodes.append(claim_node)
            
            # Map local ID to actual claim ID
            id_to_claim[claim_data.id] = claim_id
        
        # Parse relationships
        edges = []
        
        for rel in result.relationships:
            rel_type = rel.type
            if rel_type not in ['supports', 'contradicts']:
                continue
            
            actual_source = id_to_claim.get(rel.source_id)
            actual_target = id_to_claim.get(rel.target_id)
            
            if not actual_source or not actual_target:
                continue
            if actual_source == actual_target:
                continue
            
            weight = rel.weight
            if weight < 0.3:
                continue
            
            edge = ClaimEdge(
                source_id=actual_source,
                target_id=actual_target,
                type=rel_type,
                weight=float(weight),
                reasoning=rel.reasoning,
                model=self.MODEL,
            )
            edges.append(edge)
        
        print(f"      ✓ Joint extraction (structured): {len(claim_nodes)} claims, {len(edges)} edges")
        
        return claim_nodes, edges


async def extract_claims_from_abstract(
    abstract: str,
    paper_metadata: dict,
    api_key: Optional[str] = None
) -> list[ClaimNode]:
    """
    Fallback: Extract claims from just an abstract when PDF is unavailable.
    Uses beta.chat.completions.parse() with Pydantic for reliable parsing.
    
    Args:
        abstract: Paper abstract text.
        paper_metadata: Paper metadata (arxiv_id, title, authors, etc.)
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        
    Returns:
        List of ClaimNode objects (already classified).
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required")
    
    client = AsyncOpenAI(api_key=api_key)
    
    user_prompt = f"""Extract scientific claims from this paper abstract.

Paper: "{paper_metadata.get('title', 'Unknown')}"
Authors: {', '.join(paper_metadata.get('authors', [])[:3])}

Abstract:
---
{abstract}
---

Extract up to 10 important, verifiable claims. For each claim:
- Extract the original text
- Rephrase to be clear and concise (remove hedging like "may", "might")
- Classify as empirical, ground_truth, or unsupported
- Rate confidence from 0-1"""

    try:
        # Use beta.chat.completions.parse() with Pydantic model
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JOINT_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format=AbstractExtractionResult,  # Pydantic model directly
        )
        
        # response.choices[0].message.parsed is already the Pydantic object
        result = response.choices[0].message.parsed
        
        source_paper = SourcePaperMetadata(
            arxiv_id=paper_metadata.get("arxiv_id"),
            title=paper_metadata.get("title", "Unknown"),
            authors=paper_metadata.get("authors", []),
            year=paper_metadata.get("year"),
            venue=paper_metadata.get("venue") or paper_metadata.get("journal_ref"),
            section="abstract",
            url=paper_metadata.get("url"),
            doi=paper_metadata.get("doi"),
        )
        
        claim_nodes = []
        for claim_data in result.claims[:10]:
            rephrased = claim_data.rephrased or claim_data.original_text
            if not rephrased:
                continue
            
            classification = claim_data.classification
            if classification not in ["empirical", "ground_truth", "unsupported"]:
                classification = "empirical"
            
            claim_node = ClaimNode(
                id=generate_claim_id(rephrased),
                text=rephrased,
                original_text=claim_data.original_text,
                type=classification,
                confidence=float(claim_data.confidence),
                source_paper=source_paper,
                extraction=ExtractionMetadata(
                    model="gpt-4o-mini",
                    timestamp=datetime.utcnow().isoformat(),
                    prompt_version="5.0",
                    context_snippet=abstract[:500] if abstract else None,
                ),
            )
            claim_nodes.append(claim_node)
        
        print(f"      ✓ Abstract extraction: {len(claim_nodes)} claims")
        return claim_nodes
        
    except Exception as e:
        print(f"Error extracting claims from abstract: {e}")
        return []
