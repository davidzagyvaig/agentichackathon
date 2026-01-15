"""
Citation Validator Agent
5-check validation pipeline for verifying citation integrity
"""

import os
import re
from typing import Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
import httpx

from models.schemas import (
    Claim,
    Paper,
    CitationValidation,
    ClaimValidation,
    ValidationStatus,
)
from agents.paper_search import get_paper_by_doi

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_FAST = os.getenv("OPENAI_MODEL_FAST", "gpt-4o-mini")


# Known "tortured phrases" that indicate AI-generated or paper-mill content
TORTURED_PHRASES = [
    "profound learning",           # deep learning
    "counterfeit consciousness",   # artificial intelligence
    "enormous information",        # big data
    "bitcoin forecast",            # cryptocurrency prediction
    "dark opening",                # black hole
    "mental organization",         # neural network
    "profound neural organization", # deep neural network
    "semantic figuring",           # natural language processing
    "artistic intelligence",       # artificial intelligence
    "enormous information age",    # big data era
    "sham neural systems",         # fake neural networks
    "in the nick of time focus",    # at this point in time
]


async def check_citation_exists(doi: str) -> tuple[bool, Optional[Paper]]:
    """
    Check 1: Does the cited paper actually exist?
    
    Uses OpenAlex to verify the DOI corresponds to a real paper.
    """
    if not doi:
        return False, None
    
    paper = await get_paper_by_doi(doi)
    return paper is not None, paper


async def check_retraction_status(paper: Optional[Paper]) -> bool:
    """
    Check 2: Is the cited paper retracted?
    
    OpenAlex provides is_retracted field directly.
    """
    if paper is None:
        return False  # Can't check if paper doesn't exist
    
    return paper.is_retracted


async def check_citation_relevance(
    claim_text: str,
    cited_paper: Optional[Paper]
) -> tuple[bool, float, str]:
    """
    Check 3: Does the cited paper actually support the claim?
    
    Uses GPT-4o-mini to compare claim text with cited paper's abstract.
    Returns (is_relevant, relevance_score, explanation)
    """
    if cited_paper is None or not cited_paper.abstract:
        return None, 0.0, "Unable to verify - cited paper not accessible"
    
    prompt = f"""Analyze if this cited paper supports the claim being made.

CLAIM: {claim_text}

CITED PAPER:
Title: {cited_paper.title}
Abstract: {cited_paper.abstract[:1500]}

Question: Does the cited paper's content appear to support this claim?

Answer with EXACTLY one of these options:
- SUPPORTS: The paper clearly supports or provides evidence for the claim
- PARTIAL: The paper is somewhat related but doesn't directly support the claim
- IRRELEVANT: The paper has no clear connection to the claim
- CONTRADICTS: The paper contradicts or refutes the claim

Then on a new line, give a confidence score from 0.0 to 1.0.
Then on a new line, provide a brief one-sentence explanation.

Format:
VERDICT
SCORE
EXPLANATION"""

    try:
        response = await client.chat.completions.create(
            model=MODEL_FAST,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200
        )
        
        content = response.choices[0].message.content.strip()
        lines = content.split("\n")
        
        verdict = lines[0].strip().upper() if len(lines) > 0 else "UNKNOWN"
        try:
            score = float(lines[1].strip()) if len(lines) > 1 else 0.5
        except ValueError:
            score = 0.5
        explanation = lines[2].strip() if len(lines) > 2 else "Unable to determine"
        
        is_relevant = verdict in ["SUPPORTS", "PARTIAL"]
        
        return is_relevant, score, explanation
        
    except Exception as e:
        print(f"Error checking relevance: {e}")
        return None, 0.0, f"Error: {str(e)}"


async def check_circular_citation(
    source_doi: str,
    cited_doi: str
) -> bool:
    """
    Check 4: Is there a circular citation?
    
    Checks if the cited paper cites back to the source paper.
    This can indicate citation manipulation or insufficient independent evidence.
    """
    if not source_doi or not cited_doi:
        return False
    
    # Fetch the cited paper's references from OpenAlex
    async with httpx.AsyncClient(timeout=30.0) as client:
        url = f"https://api.openalex.org/works/doi:{cited_doi}"
        params = {
            "mailto": "hackathon@claimgraph.ai",
            "select": "id,referenced_works"
        }
        
        try:
            response = await client.get(url, params=params)
            if response.status_code != 200:
                return False
            
            data = response.json()
            referenced_works = data.get("referenced_works", [])
            
            # Check if source paper is in the cited paper's references
            source_openalex_id = f"https://openalex.org/W{source_doi.replace('10.', '').replace('/', '')}"
            
            # Also check DOI format
            for ref in referenced_works:
                if source_doi in str(ref):
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking circular citation: {e}")
            return False


def check_tortured_phrases(text: str) -> list[str]:
    """
    Check 5a: Detect "tortured phrases" that indicate AI-generated content.
    
    These are bizarre paraphrases that AI/plagiarism tools produce.
    """
    found = []
    text_lower = text.lower()
    
    for phrase in TORTURED_PHRASES:
        if phrase.lower() in text_lower:
            found.append(phrase)
    
    return found


def check_suspicious_patterns(text: str) -> list[str]:
    """
    Check 5b: Detect other suspicious patterns in text.
    """
    issues = []
    
    if text:
        # Check for extremely long sentences (AI often generates these)
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            if len(sentence.split()) > 80:
                issues.append("Unusually long sentence detected")
                break
        
        # Check for repetitive phrases
        words = text.lower().split()
        if len(words) > 10:
            bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
            repeated = set([b for b in bigrams if bigrams.count(b) > 3])
            if repeated:
                issues.append(f"Repetitive phrases: {', '.join(list(repeated)[:3])}")
        
        # Check for generic filler phrases common in AI text
        filler_phrases = [
            "it is important to note",
            "this study aims to",
            "the results clearly show",
            "in this paper we present",
            "it is well established that"
        ]
        for filler in filler_phrases:
            if text.lower().count(filler) > 2:
                issues.append(f"Overused phrase: '{filler}'")
    
    return issues



async def validate_claim(
    claim: Claim,
    source_paper: Paper,
    cited_papers: dict[str, Paper] = None
) -> ClaimValidation:
    """
    Run the validation pipeline on a claim.
    
    Checks:
    1. Citation existence & integrity
    2. Retraction status
    3. Content Relevance (The "Hash" Check): Does the cited paper actually say that?
    4. Circular citations
    5. Tortured phrases (Bullshit detection)
    """
    citation_validations = []
    bullshit_indicators = []
    trust_scores = []
    
    # Check 5: Tortured phrases on the claim itself (Immediate Fail Check)
    tortured = check_tortured_phrases(claim.text)
    if tortured:
        bullshit_indicators.extend([f"Tortured phrase: {p}" for p in tortured])
    
    patterns = check_suspicious_patterns(claim.text)
    bullshit_indicators.extend(patterns)
    
    # If claim has citation references, validate each
    for citation_doi in claim.citation_refs:
        validation = CitationValidation(citation_doi=citation_doi)
        
        # Check 1: Existence
        exists, cited_paper = await check_citation_exists(citation_doi)
        validation.exists = exists
        
        if exists and cited_paper:
            validation.citation_title = cited_paper.title
            
            # Check 2: Retraction (Critical Fail)
            validation.is_retracted = await check_retraction_status(cited_paper)
            if validation.is_retracted:
                bullshit_indicators.append(f"Cites retracted paper: {citation_doi}")
            
            # Check 3: Relevance / Consistency (The "Hash" Verification)
            # We check if the cited paper's abstract actually supports the claim.
            is_relevant, score, notes = await check_citation_relevance(
                claim.text, cited_paper
            )
            validation.is_relevant = is_relevant
            validation.relevance_score = score
            validation.validation_notes = notes
            
            if is_relevant is False:
                 bullshit_indicators.append(f"Citation mismatch: Cited paper {citation_doi} does not support claim")

            
            # Check 4: Circular citation
            if source_paper.doi:
                validation.is_circular = await check_circular_citation(
                    source_paper.doi, citation_doi
                )
                if validation.is_circular:
                    bullshit_indicators.append(f"Circular citation with: {citation_doi}")
            
            # Calculate trust score for this citation validation
            cite_score = 1.0
            if validation.is_retracted:
                cite_score = 0.0
            elif validation.is_relevant is False:
                cite_score = 0.1 # Strong penalty for fake citation
            elif validation.is_circular:
                cite_score *= 0.5
            
            if validation.is_relevant is None: # Could not verify content
                cite_score *= 0.7
            
            trust_scores.append(cite_score)
        else:
            # Citation doesn't exist (Hallucination Candidate)
            bullshit_indicators.append(f"Citation not found: {citation_doi}")
            validation.validation_notes = "Paper not found in databases - potential hallucination"
            trust_scores.append(0.0)
        
        citation_validations.append(validation)
    
    # Calculate overall trust score
    if trust_scores:
        overall_trust = sum(trust_scores) / len(trust_scores)
    elif claim.citation_refs:
         # Has citations but none were valid/checked
         overall_trust = 0.0
    elif bullshit_indicators:
        overall_trust = 0.2  # Suspicious patterns, no citations
    else:
        # No citations, but no obvious red flags. Unverified.
        overall_trust = 0.5 
    
    # Penalty for bullshit indicators
    if bullshit_indicators:
        overall_trust *= (0.8 ** len(bullshit_indicators))
    
    overall_trust = max(0.0, min(1.0, overall_trust))
    
    # Determine Status
    if overall_trust >= 0.8:
        status = ValidationStatus.VERIFIED
    elif overall_trust >= 0.5:
        status = ValidationStatus.PENDING
    else:
        status = ValidationStatus.SUSPICIOUS
        # If it has specific "hard" fail conditions, mark DEBUNKED
        if any("retracted" in i for i in bullshit_indicators) or \
           any("Citation not found" in i for i in bullshit_indicators):
           # Use SUSPICIOUS for now as defined in enum, or map to 'DEBUNKED' if we add it
           pass

    # Generate summary
    summary_parts = []
    if citation_validations:
        verified_count = sum(1 for cv in citation_validations if cv.exists and cv.is_relevant)
        summary_parts.append(f"{verified_count}/{len(citation_validations)} citations verified")
    
    if bullshit_indicators:
        summary_parts.append(f"FLAGS: {', '.join(bullshit_indicators[:3])}")
    
    summary = "; ".join(summary_parts) if summary_parts else "No specific issues found"
    
    return ClaimValidation(
        claim_id=claim.id,
        overall_status=status,
        trust_score=overall_trust,
        citation_validations=citation_validations,
        bullshit_indicators=bullshit_indicators,
        validation_summary=summary
    )


# For testing
if __name__ == "__main__":
    import asyncio
    
    async def test():
        # Test tortured phrase detection
        text = "We use profound learning and counterfeit consciousness for big data analysis."
        phrases = check_tortured_phrases(text)
        print(f"Found tortured phrases: {phrases}")
        
        # Test suspicious patterns
        patterns = check_suspicious_patterns(
            "It is important to note that it is important to note that results show " * 5
        )
        print(f"Found patterns: {patterns}")
    
    asyncio.run(test())
