"""
Paper Search Agent
Searches OpenAlex and Semantic Scholar for scientific papers
"""

import httpx
from typing import Optional
import asyncio

from models.schemas import Paper, PaperSearchResult


OPENALEX_BASE_URL = "https://api.openalex.org"
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"


async def search_openalex(
    query: str,
    max_results: int = 10,
    email: str = "hackathon@claimgraph.ai"
) -> list[Paper]:
    """
    Search OpenAlex for papers matching the query.
    
    OpenAlex is free, has 100k calls/day, and includes retraction data.
    """
    papers = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Build search URL
        url = f"{OPENALEX_BASE_URL}/works"
        params = {
            "search": query,
            "per_page": min(max_results, 25),
            "mailto": email,
            "select": "id,doi,title,authorships,publication_year,abstract_inverted_index,cited_by_count,referenced_works_count,is_retracted"
        }
        
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for work in data.get("results", []):
                # Reconstruct abstract from inverted index
                abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
                
                # Extract author names
                authors = []
                for authorship in work.get("authorships", []):
                    author = authorship.get("author", {})
                    if author.get("display_name"):
                        authors.append(author["display_name"])
                
                # Extract DOI (remove URL prefix if present)
                doi = work.get("doi", "")
                if doi and doi.startswith("https://doi.org/"):
                    doi = doi.replace("https://doi.org/", "")
                
                paper = Paper(
                    openalex_id=work.get("id", ""),
                    doi=doi if doi else None,
                    title=work.get("title", "Unknown Title"),
                    authors=authors[:5],  # Limit to first 5 authors
                    year=work.get("publication_year"),
                    abstract=abstract,
                    source="openalex",
                    cited_by_count=work.get("cited_by_count", 0),
                    references_count=work.get("referenced_works_count", 0),
                    is_retracted=work.get("is_retracted", False),
                    trust_score=0.0 if work.get("is_retracted") else 1.0
                )
                papers.append(paper)
                
        except httpx.HTTPError as e:
            print(f"OpenAlex API error: {e}")
        except Exception as e:
            print(f"Error parsing OpenAlex response: {e}")
    
    return papers


def reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    """
    Reconstruct abstract text from OpenAlex's inverted index format.
    
    OpenAlex stores abstracts as {word: [positions]} for compression.
    """
    if not inverted_index:
        return None
    
    try:
        # Find the maximum position to know array size
        max_pos = 0
        for positions in inverted_index.values():
            if positions:
                max_pos = max(max_pos, max(positions))
        
        # Create array and fill with words
        words = [""] * (max_pos + 1)
        for word, positions in inverted_index.items():
            for pos in positions:
                words[pos] = word
        
        return " ".join(words)
    except Exception:
        return None


async def search_semantic_scholar(
    query: str,
    max_results: int = 10
) -> list[Paper]:
    """
    Search Semantic Scholar for papers.
    
    Good fallback with additional citation context data.
    Rate limit: ~100 requests per 5 minutes without key.
    """
    papers = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        url = f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search"
        params = {
            "query": query,
            "limit": min(max_results, 10),
            "fields": "paperId,externalIds,title,authors,year,abstract,citationCount,referenceCount,isOpenAccess"
        }
        
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for paper_data in data.get("data", []):
                # Extract DOI from external IDs
                external_ids = paper_data.get("externalIds", {})
                doi = external_ids.get("DOI")
                
                # Extract author names
                authors = [
                    a.get("name", "")
                    for a in paper_data.get("authors", [])
                    if a.get("name")
                ]
                
                paper = Paper(
                    openalex_id=paper_data.get("paperId", ""),
                    doi=doi,
                    title=paper_data.get("title", "Unknown Title"),
                    authors=authors[:5],
                    year=paper_data.get("year"),
                    abstract=paper_data.get("abstract"),
                    source="semantic_scholar",
                    cited_by_count=paper_data.get("citationCount", 0),
                    references_count=paper_data.get("referenceCount", 0),
                )
                papers.append(paper)
                
        except httpx.HTTPError as e:
            print(f"Semantic Scholar API error: {e}")
        except Exception as e:
            print(f"Error parsing Semantic Scholar response: {e}")
    
    return papers


async def search_papers(
    query: str,
    max_results: int = 10,
    use_semantic_scholar: bool = False
) -> PaperSearchResult:
    """
    Main search function - searches OpenAlex by default.
    
    Args:
        query: Search query string
        max_results: Maximum number of papers to return
        use_semantic_scholar: If True, also search Semantic Scholar
        
    Returns:
        PaperSearchResult with list of papers
    """
    # Primary search: OpenAlex
    papers = await search_openalex(query, max_results)
    
    # Optional: Also search Semantic Scholar
    if use_semantic_scholar and len(papers) < max_results:
        ss_papers = await search_semantic_scholar(query, max_results - len(papers))
        
        # Deduplicate by DOI
        existing_dois = {p.doi for p in papers if p.doi}
        for paper in ss_papers:
            if paper.doi and paper.doi not in existing_dois:
                papers.append(paper)
                existing_dois.add(paper.doi)
    
    return PaperSearchResult(
        papers=papers[:max_results],
        total_count=len(papers),
        query=query
    )


async def get_paper_by_doi(doi: str) -> Optional[Paper]:
    """
    Fetch a specific paper by DOI from OpenAlex.
    
    Used for validating cited papers.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        url = f"{OPENALEX_BASE_URL}/works/doi:{doi}"
        params = {
            "mailto": "hackathon@claimgraph.ai",
            "select": "id,doi,title,authorships,publication_year,abstract_inverted_index,cited_by_count,is_retracted"
        }
        
        try:
            response = await client.get(url, params=params)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            
            work = response.json()
            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            
            authors = []
            for authorship in work.get("authorships", []):
                author = authorship.get("author", {})
                if author.get("display_name"):
                    authors.append(author["display_name"])
            
            doi_clean = work.get("doi", "")
            if doi_clean and doi_clean.startswith("https://doi.org/"):
                doi_clean = doi_clean.replace("https://doi.org/", "")
            
            return Paper(
                openalex_id=work.get("id", ""),
                doi=doi_clean if doi_clean else None,
                title=work.get("title", "Unknown Title"),
                authors=authors[:5],
                year=work.get("publication_year"),
                abstract=abstract,
                source="openalex",
                cited_by_count=work.get("cited_by_count", 0),
                is_retracted=work.get("is_retracted", False),
            )
            
        except Exception as e:
            print(f"Error fetching paper by DOI: {e}")
            return None


# For testing
if __name__ == "__main__":
    async def test():
        result = await search_papers("machine learning climate change", max_results=5)
        print(f"Found {result.total_count} papers")
        for paper in result.papers:
            print(f"  - {paper.title[:60]}... ({paper.year})")
            if paper.abstract:
                print(f"    Abstract: {paper.abstract[:100]}...")
    
    asyncio.run(test())
