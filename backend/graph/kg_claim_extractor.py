"""
Claim Extractor for Knowledge Graph

Uses GPT-4o-mini to extract, classify, and detect relationships between
scientific claims from paper text.

Implements PRD Section 6: Claim Extraction.
"""

import json
import asyncio
from typing import Optional
from datetime import datetime
from openai import AsyncOpenAI
import os

from models.kg_schemas import (
    ClaimNode,
    ClaimEdge,
    SourcePaperMetadata,
    ExtractionMetadata,
    generate_claim_id,
    generate_edge_id,
)
from graph.section_splitter import SectionSplitter, split_into_paragraphs


# ============================================================================
# PROMPT TEMPLATES (from PRD Section 6)
# ============================================================================

CLAIM_EXTRACTION_SYSTEM_PROMPT = """# SYSTEM PROMPT: Scientific Claim Extractor

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

Return a JSON object with this structure:
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

## Guidelines:
- Extract 5-15 claims per paper section (focus on important ones)
- Rephrase to remove hedging ("may", "might", "could")
- Keep quantitative details (numbers, p-values)
- If a claim references another paper, note the citation
- Focus on claims that could be verified or disputed"""


CLAIM_CLASSIFICATION_SYSTEM_PROMPT = """# SYSTEM PROMPT: Claim Classifier

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

Return a JSON object:
{
  "classification": "ground_truth" | "empirical" | "unsupported",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of classification"
}

## IMPORTANT:
- When in doubt, classify as "empirical" (not ground_truth)
- "Unsupported" is NOT a judgment of truth, just that no evidence is provided in context
- A claim can be true AND unsupported (if no justification is given)"""


EDGE_DETECTION_SYSTEM_PROMPT = """# SYSTEM PROMPT: Claim Relationship Detector

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

For each pair of claims provided, return a JSON object:
{
  "relationships": [
    {
      "source_index": 0,
      "target_index": 1,
      "relationship": "supports" | "contradicts" | "none",
      "weight": 0.0-1.0,
      "reasoning": "Why this relationship exists"
    }
  ]
}

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
- Only return relationships that are "supports" or "contradicts", skip "none\""""


class ClaimExtractor:
    """
    Extracts scientific claims from paper text using LLMs.
    """
    
    MODEL = "gpt-4o-mini"
    PROMPT_VERSION = "1.0"
    MAX_CLAIMS_PER_SECTION = 15
    
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
        self.section_splitter = SectionSplitter()
    
    async def extract_claims_from_paper(
        self,
        paper_text: str,
        paper_metadata: dict,
        max_claims: int = 50
    ) -> list[ClaimNode]:
        """
        Extract all claims from a paper.
        
        Args:
            paper_text: Full text of the paper
            paper_metadata: Paper metadata (arxiv_id, title, authors, etc.)
            max_claims: Maximum total claims to extract
            
        Returns:
            List of ClaimNode objects with classifications.
        """
        sections = self.section_splitter.extract_claims_sections(paper_text)
        
        if not sections:
            sections = [('main', paper_text)]
        
        all_claims = []
        
        for section_name, section_content in sections:
            if len(all_claims) >= max_claims:
                break
            
            remaining = max_claims - len(all_claims)
            section_claims = await self._extract_from_section(
                section_content,
                section_name,
                paper_metadata,
                max_claims=min(remaining, self.MAX_CLAIMS_PER_SECTION)
            )
            
            all_claims.extend(section_claims)
        
        classified_claims = await self._classify_claims_batch(all_claims)
        
        return classified_claims
    
    async def _extract_from_section(
        self,
        section_text: str,
        section_name: str,
        paper_metadata: dict,
        max_claims: int = 15
    ) -> list[ClaimNode]:
        """Extract claims from a single section."""
        
        if len(section_text) > 12000:
            section_text = section_text[:12000]
        
        user_prompt = f"""Extract scientific claims from this {section_name.upper()} section of a paper.

Paper: "{paper_metadata.get('title', 'Unknown')}"
Authors: {', '.join(paper_metadata.get('authors', [])[:3])}

Section text:
---
{section_text}
---

Extract up to {max_claims} important claims. Focus on verifiable, specific assertions."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            claims = result.get("claims", [])
            
            source_paper = SourcePaperMetadata(
                arxiv_id=paper_metadata.get("arxiv_id"),
                title=paper_metadata.get("title", "Unknown"),
                authors=paper_metadata.get("authors", []),
                year=paper_metadata.get("year"),
                venue=paper_metadata.get("venue") or paper_metadata.get("journal_ref"),
                section=section_name,
                url=paper_metadata.get("url"),
                doi=paper_metadata.get("doi"),
            )
            
            claim_nodes = []
            for claim_data in claims[:max_claims]:
                rephrased = claim_data.get("rephrased", claim_data.get("original_text", ""))
                if not rephrased:
                    continue
                
                claim_node = ClaimNode(
                    id=generate_claim_id(rephrased),
                    text=rephrased,
                    original_text=claim_data.get("original_text"),
                    type="empirical",
                    confidence=0.5,
                    source_paper=source_paper,
                    extraction=ExtractionMetadata(
                        model=self.MODEL,
                        timestamp=datetime.utcnow().isoformat(),
                        prompt_version=self.PROMPT_VERSION,
                        context_snippet=section_text[:500] if section_text else None,
                    ),
                )
                claim_nodes.append(claim_node)
            
            return claim_nodes
            
        except Exception as e:
            print(f"Error extracting claims from {section_name}: {e}")
            return []
    
    async def _classify_claims_batch(self, claims: list[ClaimNode]) -> list[ClaimNode]:
        """Classify all claims in batch."""
        
        tasks = [self._classify_single_claim(claim) for claim in claims]
        classified = await asyncio.gather(*tasks, return_exceptions=True)
        
        result = []
        for i, claim_result in enumerate(classified):
            if isinstance(claim_result, Exception):
                print(f"Classification error: {claim_result}")
                result.append(claims[i])
            else:
                result.append(claim_result)
        
        return result
    
    async def _classify_single_claim(self, claim: ClaimNode) -> ClaimNode:
        """Classify a single claim."""
        
        user_prompt = f"""Classify this scientific claim:

Claim: "{claim.text}"

Source paper: {claim.source_paper.title}
Section: {claim.source_paper.section or 'unknown'}

Determine if this is ground_truth, empirical, or unsupported."""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": CLAIM_CLASSIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            classification = result.get("classification", "empirical")
            if classification not in ["empirical", "ground_truth", "unsupported"]:
                classification = "empirical"
            
            claim.type = classification
            claim.confidence = float(result.get("confidence", 0.5))
            
            return claim
            
        except Exception as e:
            print(f"Error classifying claim: {e}")
            return claim
    
    async def detect_edges(
        self,
        claims: list[ClaimNode],
        batch_size: int = 10
    ) -> list[ClaimEdge]:
        """
        Detect support/contradiction relationships between claims.
        
        Args:
            claims: List of claims to analyze
            batch_size: Number of claim pairs to analyze per API call
            
        Returns:
            List of ClaimEdge objects for detected relationships.
        """
        if len(claims) < 2:
            return []
        
        edges = []
        
        for i in range(0, len(claims), batch_size):
            batch = claims[i:i + batch_size]
            batch_edges = await self._detect_edges_batch(batch)
            edges.extend(batch_edges)
        
        cross_edges = await self._detect_cross_batch_edges(claims)
        edges.extend(cross_edges)
        
        return edges
    
    async def _detect_edges_batch(self, claims: list[ClaimNode]) -> list[ClaimEdge]:
        """Detect edges within a batch of claims."""
        
        if len(claims) < 2:
            return []
        
        claims_list = "\n".join([
            f"{i}. [{c.source_paper.section or 'unknown'}] {c.text}"
            for i, c in enumerate(claims)
        ])
        
        user_prompt = f"""Analyze relationships between these scientific claims from the same paper.
Only identify clear support or contradiction relationships.

Claims:
{claims_list}

For each significant relationship found, indicate:
- source_index: The claim providing evidence
- target_index: The claim being supported/contradicted
- relationship: "supports" or "contradicts"
- weight: Strength of relationship (0.3-1.0)
- reasoning: Brief explanation"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": EDGE_DETECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            relationships = result.get("relationships", [])
            
            edges = []
            for rel in relationships:
                rel_type = rel.get("relationship")
                if rel_type not in ["supports", "contradicts"]:
                    continue
                
                source_idx = rel.get("source_index", -1)
                target_idx = rel.get("target_index", -1)
                
                if source_idx < 0 or source_idx >= len(claims):
                    continue
                if target_idx < 0 or target_idx >= len(claims):
                    continue
                if source_idx == target_idx:
                    continue
                
                edge = ClaimEdge(
                    source_id=claims[source_idx].id,
                    target_id=claims[target_idx].id,
                    type=rel_type,
                    weight=float(rel.get("weight", 0.5)),
                    reasoning=rel.get("reasoning"),
                    model=self.MODEL,
                )
                edges.append(edge)
            
            return edges
            
        except Exception as e:
            print(f"Error detecting edges: {e}")
            return []
    
    async def _detect_cross_batch_edges(self, claims: list[ClaimNode]) -> list[ClaimEdge]:
        """
        Detect edges between claims from different sections.
        Focuses on claims that might support/contradict across sections.
        """
        
        if len(claims) < 10:
            return []
        
        results_claims = [c for c in claims if c.source_paper.section == 'results']
        intro_claims = [c for c in claims if c.source_paper.section == 'introduction']
        conclusion_claims = [c for c in claims if c.source_paper.section == 'conclusion']
        
        edges = []
        
        if results_claims and intro_claims:
            cross_batch = results_claims[:5] + intro_claims[:5]
            batch_edges = await self._detect_edges_batch(cross_batch)
            edges.extend(batch_edges)
        
        if results_claims and conclusion_claims:
            cross_batch = results_claims[:5] + conclusion_claims[:5]
            batch_edges = await self._detect_edges_batch(cross_batch)
            edges.extend(batch_edges)
        
        return edges
    
    async def extract_and_link_claims(
        self,
        paper_text: str,
        paper_metadata: dict,
        max_claims: int = 50
    ) -> tuple[list[ClaimNode], list[ClaimEdge]]:
        """
        Full extraction pipeline: extract, classify, and link claims.
        
        Args:
            paper_text: Full text of the paper
            paper_metadata: Paper metadata
            max_claims: Maximum claims to extract
            
        Returns:
            Tuple of (claims, edges).
        """
        claims = await self.extract_claims_from_paper(
            paper_text,
            paper_metadata,
            max_claims=max_claims
        )
        
        edges = await self.detect_edges(claims)
        
        return claims, edges


async def extract_claims_from_abstract(
    abstract: str,
    paper_metadata: dict,
    api_key: Optional[str] = None
) -> list[ClaimNode]:
    """
    Quick extraction from just an abstract.
    Useful when full text is not available.
    """
    extractor = ClaimExtractor(api_key=api_key)
    
    paper_metadata_copy = dict(paper_metadata)
    
    claims = await extractor._extract_from_section(
        abstract,
        "abstract",
        paper_metadata_copy,
        max_claims=10
    )
    
    classified = await extractor._classify_claims_batch(claims)
    
    return classified
