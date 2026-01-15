"""
Semantic Scholar API Client for fetching paper metadata and citations.

API documentation: https://api.semanticscholar.org/api-docs/
Rate limits: 100 requests/5 minutes without API key, higher with key.
"""

import asyncio
import aiohttp
import os
from typing import Optional


class SemanticScholarClient:
    """
    Client for interacting with the Semantic Scholar API.
    Provides paper metadata, citations, and references.
    """
    
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    REQUESTS_PER_WINDOW = 100
    WINDOW_SECONDS = 300
    MIN_REQUEST_INTERVAL = WINDOW_SECONDS / REQUESTS_PER_WINDOW
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Semantic Scholar client.
        
        Args:
            api_key: Optional API key for higher rate limits.
                     Falls back to SEMANTIC_SCHOLAR_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self._last_request_time: Optional[float] = None
        self._lock = asyncio.Lock()
    
    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        async with self._lock:
            if self._last_request_time is not None:
                elapsed = asyncio.get_event_loop().time() - self._last_request_time
                if elapsed < self.MIN_REQUEST_INTERVAL:
                    await asyncio.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
            self._last_request_time = asyncio.get_event_loop().time()
    
    def _get_headers(self) -> dict:
        """Get request headers including API key if available."""
        headers = {
            "Accept": "application/json",
            "User-Agent": "ClaimGraph/1.0"
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers
    
    async def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[dict]:
        """
        Get paper metadata by arXiv ID.
        
        Args:
            arxiv_id: The arXiv paper ID (e.g., '1706.01787')
            
        Returns:
            Paper metadata dictionary or None if not found.
        """
        arxiv_id = arxiv_id.strip().replace('arxiv:', '').replace('arXiv:', '')
        return await self._get_paper(f"ARXIV:{arxiv_id}")
    
    async def get_paper_by_doi(self, doi: str) -> Optional[dict]:
        """
        Get paper metadata by DOI.
        
        Args:
            doi: The paper DOI (e.g., '10.1038/nature12373')
            
        Returns:
            Paper metadata dictionary or None if not found.
        """
        doi = doi.strip()
        if doi.startswith('https://doi.org/'):
            doi = doi[16:]
        elif doi.startswith('doi:'):
            doi = doi[4:]
        return await self._get_paper(doi)
    
    async def get_paper_by_id(self, paper_id: str) -> Optional[dict]:
        """
        Get paper metadata by Semantic Scholar paper ID.
        
        Args:
            paper_id: The Semantic Scholar paper ID
            
        Returns:
            Paper metadata dictionary or None if not found.
        """
        return await self._get_paper(paper_id)
    
    async def _get_paper(self, paper_identifier: str) -> Optional[dict]:
        """
        Internal method to fetch paper by any identifier.
        
        Args:
            paper_identifier: Paper ID (S2 ID, DOI, ARXIV:id, etc.)
            
        Returns:
            Paper metadata dictionary.
        """
        await self._rate_limit()
        
        fields = [
            "paperId", "externalIds", "url", "title", "abstract",
            "venue", "year", "referenceCount", "citationCount",
            "isOpenAccess", "openAccessPdf", "fieldsOfStudy",
            "authors", "publicationDate", "journal"
        ]
        
        url = f"{self.BASE_URL}/paper/{paper_identifier}"
        params = {"fields": ",".join(fields)}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 404:
                        return None
                    if response.status != 200:
                        print(f"Semantic Scholar API returned status {response.status}")
                        return None
                    
                    data = await response.json()
                    return self._normalize_paper(data)
                    
        except asyncio.TimeoutError:
            print(f"Timeout fetching paper {paper_identifier}")
            return None
        except Exception as e:
            print(f"Error fetching paper: {e}")
            return None
    
    def _normalize_paper(self, data: dict) -> dict:
        """Normalize API response to standard format."""
        external_ids = data.get("externalIds", {}) or {}
        authors_data = data.get("authors", []) or []
        journal = data.get("journal", {}) or {}
        
        return {
            "s2_id": data.get("paperId"),
            "arxiv_id": external_ids.get("ArXiv"),
            "doi": external_ids.get("DOI"),
            "title": data.get("title", ""),
            "abstract": data.get("abstract", ""),
            "authors": [a.get("name", "") for a in authors_data if a.get("name")],
            "year": data.get("year"),
            "venue": data.get("venue") or journal.get("name"),
            "citation_count": data.get("citationCount", 0),
            "reference_count": data.get("referenceCount", 0),
            "fields_of_study": data.get("fieldsOfStudy", []),
            "is_open_access": data.get("isOpenAccess", False),
            "open_access_pdf": data.get("openAccessPdf", {}).get("url") if data.get("openAccessPdf") else None,
            "publication_date": data.get("publicationDate"),
            "url": data.get("url")
        }
    
    async def get_citations(
        self,
        paper_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict]:
        """
        Get papers that cite the given paper.
        
        Args:
            paper_id: Paper identifier (S2 ID, DOI, ARXIV:id)
            limit: Maximum number of citations to return (max 1000)
            offset: Offset for pagination
            
        Returns:
            List of citing paper metadata.
        """
        await self._rate_limit()
        
        fields = [
            "paperId", "externalIds", "title", "abstract",
            "year", "citationCount", "authors"
        ]
        
        url = f"{self.BASE_URL}/paper/{paper_id}/citations"
        params = {
            "fields": ",".join(fields),
            "limit": min(limit, 1000),
            "offset": offset
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        print(f"Error fetching citations: status {response.status}")
                        return []
                    
                    data = await response.json()
                    citations = []
                    
                    for item in data.get("data", []):
                        citing_paper = item.get("citingPaper", {})
                        if citing_paper:
                            citations.append(self._normalize_citation(citing_paper))
                    
                    return citations
                    
        except Exception as e:
            print(f"Error fetching citations: {e}")
            return []
    
    async def get_references(
        self,
        paper_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> list[dict]:
        """
        Get papers referenced by the given paper.
        
        Args:
            paper_id: Paper identifier (S2 ID, DOI, ARXIV:id)
            limit: Maximum number of references to return (max 1000)
            offset: Offset for pagination
            
        Returns:
            List of referenced paper metadata.
        """
        await self._rate_limit()
        
        fields = [
            "paperId", "externalIds", "title", "abstract",
            "year", "citationCount", "authors"
        ]
        
        url = f"{self.BASE_URL}/paper/{paper_id}/references"
        params = {
            "fields": ",".join(fields),
            "limit": min(limit, 1000),
            "offset": offset
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        print(f"Error fetching references: status {response.status}")
                        return []
                    
                    data = await response.json()
                    references = []
                    
                    for item in data.get("data", []):
                        cited_paper = item.get("citedPaper", {})
                        if cited_paper and cited_paper.get("paperId"):
                            references.append(self._normalize_citation(cited_paper))
                    
                    return references
                    
        except Exception as e:
            print(f"Error fetching references: {e}")
            return []
    
    def _normalize_citation(self, data: dict) -> dict:
        """Normalize citation data to standard format."""
        external_ids = data.get("externalIds", {}) or {}
        authors_data = data.get("authors", []) or []
        
        return {
            "s2_id": data.get("paperId"),
            "arxiv_id": external_ids.get("ArXiv"),
            "doi": external_ids.get("DOI"),
            "title": data.get("title", ""),
            "abstract": data.get("abstract", ""),
            "authors": [a.get("name", "") for a in authors_data if a.get("name")],
            "year": data.get("year"),
            "citation_count": data.get("citationCount", 0)
        }
    
    async def search_papers(
        self,
        query: str,
        limit: int = 10,
        year_filter: Optional[str] = None,
        fields_of_study: Optional[list[str]] = None,
        open_access_only: bool = False
    ) -> list[dict]:
        """
        Search for papers by keyword query.
        
        Args:
            query: Search query string
            limit: Maximum results to return (max 100)
            year_filter: Year range filter (e.g., '2020-2024', '2020-', '-2020')
            fields_of_study: Filter by fields (e.g., ['Biology', 'Computer Science'])
            open_access_only: Only return open access papers
            
        Returns:
            List of matching paper metadata.
        """
        await self._rate_limit()
        
        fields = [
            "paperId", "externalIds", "title", "abstract",
            "year", "citationCount", "referenceCount", "authors",
            "isOpenAccess", "openAccessPdf", "fieldsOfStudy"
        ]
        
        url = f"{self.BASE_URL}/paper/search"
        params = {
            "query": query,
            "fields": ",".join(fields),
            "limit": min(limit, 100)
        }
        
        if year_filter:
            params["year"] = year_filter
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        if open_access_only:
            params["openAccessPdf"] = ""
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        print(f"Search returned status {response.status}")
                        return []
                    
                    data = await response.json()
                    results = []
                    
                    for paper in data.get("data", []):
                        results.append(self._normalize_paper(paper))
                    
                    return results
                    
        except Exception as e:
            print(f"Error searching papers: {e}")
            return []
    
    async def get_paper_batch(self, paper_ids: list[str]) -> list[dict]:
        """
        Fetch multiple papers in a single batch request.
        
        Args:
            paper_ids: List of paper identifiers (max 500)
            
        Returns:
            List of paper metadata dictionaries.
        """
        await self._rate_limit()
        
        if len(paper_ids) > 500:
            paper_ids = paper_ids[:500]
        
        fields = [
            "paperId", "externalIds", "title", "abstract",
            "year", "citationCount", "authors"
        ]
        
        url = f"{self.BASE_URL}/paper/batch"
        params = {"fields": ",".join(fields)}
        body = {"ids": paper_ids}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    params=params,
                    json=body,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        print(f"Batch request returned status {response.status}")
                        return []
                    
                    data = await response.json()
                    results = []
                    
                    for paper in data:
                        if paper:
                            results.append(self._normalize_citation(paper))
                    
                    return results
                    
        except Exception as e:
            print(f"Error in batch request: {e}")
            return []
