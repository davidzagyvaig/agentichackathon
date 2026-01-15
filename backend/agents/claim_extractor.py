"""
Claim Extraction Agent
Uses GPT-4o to extract structured claims from paper abstracts
"""

import os
import json
from typing import Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
import time

from models.schemas import (
    Paper,
    Claim,
    ClaimExtractionResult,
    ClaimType,
    EvidenceType,
    ValidationStatus,
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


CLAIM_EXTRACTION_SYSTEM_PROMPT = """You are an expert scientific claim extractor. Your job is to analyze scientific paper abstracts and extract all notable claims made by the authors.

For each claim you identify, you must classify it with:
1. claim_type: The nature of the claim
   - "causal": One variable directly causes/affects another (X causes Y)
   - "correlational": Two things are associated/correlated without implying causation
   - "comparative": Two methods/approaches/objects are compared (X outperforms Y)
   - "methodological": A claim about a method, approach, or technique
   - "existence": Asserting the existence or discovery of something new

2. evidence_type: How the claim is supported in the paper
   - "citation": Supported by a cited reference
   - "experiment": Backed by the authors' own experimental results
   - "figure": Supported by a figure, table, or diagram
   - "other_claim": Builds on another claim made in the paper
   - "general_knowledge": Based on commonly accepted facts (no citation needed)
   - "unsupported": No clear evidence or support provided

Extract between 2-6 claims per abstract. Focus on the main scientific contributions and findings.

CRITICAL: Output ONLY valid JSON with the structure: {"claims": [...]}"""


CLAIM_EXTRACTION_USER_PROMPT = """Analyze this scientific paper abstract and extract ALL notable claims (3-6 claims expected).

Paper Title: {title}
Authors: {authors}
Year: {year}

Abstract:
{abstract}

Extract every scientific claim, finding, or assertion. Output a JSON object with this EXACT structure:
{{
  "claims": [
    {{
      "text": "The exact claim statement",
      "claim_type": "causal|correlational|comparative|methodological|existence",
      "evidence_type": "citation|experiment|figure|other_claim|general_knowledge|unsupported"
    }},
    ... more claims ...
  ]
}}

You MUST return 3-6 claims. Output ONLY valid JSON:"""


async def extract_claims_from_paper(paper: Paper) -> ClaimExtractionResult:
    """
    Extract structured claims from a paper's abstract using GPT-4o.
    
    Args:
        paper: Paper object with abstract
        
    Returns:
        ClaimExtractionResult with list of claims
    """
    start_time = time.time()
    claims = []
    
    if not paper.abstract:
        return ClaimExtractionResult(
            paper_id=paper.id,
            claims=[],
            extraction_time=0.0
        )
    
    # Format the prompt
    authors_str = ", ".join(paper.authors[:3])
    if len(paper.authors) > 3:
        authors_str += " et al."
    
    user_prompt = CLAIM_EXTRACTION_USER_PROMPT.format(
        title=paper.title,
        authors=authors_str,
        year=paper.year or "Unknown",
        abstract=paper.abstract
    )
    
    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": CLAIM_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent structured output
            max_tokens=2000,
            response_format={"type": "json_object"}  # Force JSON output
        )
        
        # Parse the response
        content = response.choices[0].message.content
        
        # Handle potential JSON wrapping
        if content:
            # Sometimes GPT wraps in {"claims": [...]} format
            try:
                parsed = json.loads(content)
                
                # Debug: print what we got
                print(f"   GPT response type: {type(parsed)}")
                if isinstance(parsed, dict):
                    print(f"   GPT response keys: {list(parsed.keys())}")
                    # Print first 500 chars to see structure
                    print(f"   Raw: {content[:500]}")
                
                claims_data = []
                
                if isinstance(parsed, dict):
                    # Strategy 1: Look for common keys
                    for key in ["claims", "extracted_claims", "results", "data", "scientific_claims"]:
                        if key in parsed and isinstance(parsed[key], list):
                            claims_data = parsed[key]
                            print(f"   Found claims under key: {key}")
                            break
                    
                    # Strategy 2: If no known keys, find ANY list that contains dicts
                    if not claims_data:
                        for key, value in parsed.items():
                            if isinstance(value, list) and len(value) > 0:
                                # Check if first item looks like a claim (has text field)
                                if isinstance(value[0], dict):
                                    if "text" in value[0] or "claim_type" in value[0]:
                                        claims_data = value
                                        print(f"   Found claims under key: {key}")
                                        break
                    
                    # Strategy 3: Maybe the dict itself IS a single claim
                    if not claims_data and "text" in parsed and "claim_type" in parsed:
                        # GPT returned a single claim as a dict
                        claims_data = [parsed]
                        print(f"   GPT returned single claim dict - wrapped in list")
                    
                    # Strategy 4: Maybe the dict values ARE the claims directly (numbered dict)
                    if not claims_data and all(isinstance(v, dict) for v in parsed.values()):
                        # Each value might be a claim
                        claims_data = list(parsed.values())
                        print(f"   Using dict values as claims")
                        
                elif isinstance(parsed, list):
                    claims_data = parsed
                    print(f"   Direct list of {len(claims_data)} items")
                    
            except json.JSONDecodeError as je:
                print(f"   JSON decode error: {je}")
                # Try to extract JSON array from text
                import re
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    try:
                        claims_data = json.loads(match.group())
                    except:
                        claims_data = []
                else:
                    claims_data = []
            
            print(f"   Found {len(claims_data)} claim candidates")
            
            # Convert to Claim objects
            for claim_data in claims_data:
                try:
                    # Skip if not a dict
                    if not isinstance(claim_data, dict):
                        print(f"   Skipping non-dict claim: {type(claim_data)}")
                        continue
                    
                    claim = Claim(
                        text=claim_data.get("text", ""),
                        claim_type=ClaimType(claim_data.get("claim_type", "existence")),
                        evidence_type=EvidenceType(claim_data.get("evidence_type", "unsupported")),
                        paper_id=paper.id,
                        confidence=0.8,
                        validation_status=ValidationStatus.PENDING
                    )
                    claims.append(claim)
                except (ValueError, KeyError, AttributeError) as e:
                    print(f"   Error parsing claim: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error extracting claims: {e}")
        import traceback
        traceback.print_exc()
    
    extraction_time = time.time() - start_time
    
    return ClaimExtractionResult(
        paper_id=paper.id,
        claims=claims,
        extraction_time=extraction_time
    )


async def extract_claims_batch(papers: list[Paper]) -> list[ClaimExtractionResult]:
    """
    Extract claims from multiple papers.
    
    Processes papers sequentially to respect rate limits.
    """
    import asyncio
    
    results = []
    for paper in papers:
        result = await extract_claims_from_paper(paper)
        results.append(result)
        # Small delay to respect rate limits
        await asyncio.sleep(0.5)
    
    return results


# For testing
if __name__ == "__main__":
    import asyncio
    
    async def test():
        # Create a test paper
        test_paper = Paper(
            id="test-123",
            title="Deep Learning for Climate Change Prediction",
            authors=["John Smith", "Jane Doe"],
            year=2024,
            abstract="""We present a novel deep learning approach for predicting climate change 
            patterns with unprecedented accuracy. Our model outperforms traditional methods by 35% 
            on standard benchmarks. We demonstrate that transformer architectures can capture 
            long-range temporal dependencies in climate data, enabling predictions up to 50 years 
            into the future. Our experiments on 40 years of satellite data show that the model 
            accurately identifies key climate tipping points. These findings suggest that AI 
            can play a crucial role in climate science research."""
        )
        
        result = await extract_claims_from_paper(test_paper)
        print(f"Extracted {len(result.claims)} claims in {result.extraction_time:.2f}s:")
        for claim in result.claims:
            print(f"  [{claim.claim_type}] {claim.text[:80]}...")
    
    asyncio.run(test())
