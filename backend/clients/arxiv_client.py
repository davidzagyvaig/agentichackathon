"""
ArXiv API Client for fetching paper metadata and full text.

Rate limiting: arXiv requests 3-second delay between requests.
API documentation: https://arxiv.org/help/api/
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET
import tempfile
import os
import re
import hashlib
from typing import Optional
from datetime import datetime


class ArxivClient:
    """
    Client for interacting with the arXiv API and fetching paper content.
    """
    
    BASE_URL = "http://export.arxiv.org/api/query"
    PDF_BASE_URL = "https://arxiv.org/pdf"
    ABS_BASE_URL = "https://arxiv.org/abs"
    
    # Rate limiting: 3 seconds between requests per arXiv guidelines
    RATE_LIMIT_SECONDS = 3.0
    
    def __init__(self):
        self._last_request_time: Optional[float] = None
        self._lock = asyncio.Lock()
    
    async def _rate_limit(self):
        """Enforce rate limiting between requests."""
        async with self._lock:
            if self._last_request_time is not None:
                elapsed = asyncio.get_event_loop().time() - self._last_request_time
                if elapsed < self.RATE_LIMIT_SECONDS:
                    await asyncio.sleep(self.RATE_LIMIT_SECONDS - elapsed)
            self._last_request_time = asyncio.get_event_loop().time()
    
    def _normalize_arxiv_id(self, arxiv_id: str) -> str:
        """
        Normalize arXiv ID to standard format (e.g., '1706.01787').
        Handles various input formats like 'arxiv:1706.01787', '1706.01787v1', etc.
        """
        arxiv_id = arxiv_id.strip().lower()
        arxiv_id = re.sub(r'^arxiv:', '', arxiv_id)
        arxiv_id = re.sub(r'^abs/', '', arxiv_id)
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
        return arxiv_id
    
    async def fetch_paper_metadata(self, arxiv_id: str) -> Optional[dict]:
        """
        Fetch paper metadata from arXiv API.
        
        Args:
            arxiv_id: The arXiv paper ID (e.g., '1706.01787')
            
        Returns:
            Dictionary with paper metadata or None if not found.
        """
        await self._rate_limit()
        
        arxiv_id = self._normalize_arxiv_id(arxiv_id)
        url = f"{self.BASE_URL}?id_list={arxiv_id}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        print(f"ArXiv API returned status {response.status}")
                        return None
                    
                    text = await response.text()
                    return self._parse_entry(text, arxiv_id)
                    
        except asyncio.TimeoutError:
            print(f"Timeout fetching arXiv metadata for {arxiv_id}")
            return None
        except Exception as e:
            print(f"Error fetching arXiv metadata: {e}")
            return None
    
    def _parse_entry(self, xml_text: str, arxiv_id: str) -> Optional[dict]:
        """Parse arXiv API XML response to extract paper metadata."""
        try:
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            root = ET.fromstring(xml_text)
            entry = root.find('atom:entry', ns)
            
            if entry is None:
                return None
            
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ""
            
            summary_elem = entry.find('atom:summary', ns)
            abstract = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else ""
            
            authors = []
            for author in entry.findall('atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            published_elem = entry.find('atom:published', ns)
            year = None
            published_date = None
            if published_elem is not None:
                published_date = published_elem.text
                try:
                    year = int(published_date[:4])
                except (ValueError, TypeError):
                    pass
            
            categories = []
            for cat in entry.findall('atom:category', ns):
                term = cat.get('term')
                if term:
                    categories.append(term)
            
            primary_category_elem = entry.find('arxiv:primary_category', ns)
            primary_category = primary_category_elem.get('term') if primary_category_elem is not None else None
            
            doi_elem = entry.find('arxiv:doi', ns)
            doi = doi_elem.text if doi_elem is not None else None
            
            journal_ref_elem = entry.find('arxiv:journal_ref', ns)
            journal_ref = journal_ref_elem.text if journal_ref_elem is not None else None
            
            return {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "year": year,
                "published_date": published_date,
                "categories": categories,
                "primary_category": primary_category,
                "doi": doi,
                "journal_ref": journal_ref,
                "url": f"{self.ABS_BASE_URL}/{arxiv_id}",
                "pdf_url": f"{self.PDF_BASE_URL}/{arxiv_id}.pdf"
            }
            
        except ET.ParseError as e:
            print(f"XML parse error: {e}")
            return None
    
    async def fetch_paper_pdf(self, arxiv_id: str) -> Optional[bytes]:
        """
        Fetch raw PDF bytes from arXiv.
        
        Args:
            arxiv_id: The arXiv paper ID
            
        Returns:
            Raw PDF bytes or None if download fails.
        """
        await self._rate_limit()
        
        arxiv_id = self._normalize_arxiv_id(arxiv_id)
        pdf_url = f"{self.PDF_BASE_URL}/{arxiv_id}.pdf"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(pdf_url, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    if response.status != 200:
                        print(f"      Failed to download PDF: status {response.status}")
                        return None
                    
                    pdf_bytes = await response.read()
                    return pdf_bytes
            
        except asyncio.TimeoutError:
            print(f"      Timeout downloading PDF for {arxiv_id}")
            return None
        except Exception as e:
            print(f"      Error fetching PDF: {e}")
            return None
    
    async def fetch_paper_text(self, arxiv_id: str) -> Optional[str]:
        """
        Fetch and extract text from arXiv paper PDF.
        
        Args:
            arxiv_id: The arXiv paper ID
            
        Returns:
            Extracted text content or None if extraction fails.
        """
        pdf_bytes = await self.fetch_paper_pdf(arxiv_id)
        if not pdf_bytes:
            return None
        
        text = await self._extract_text_from_pdf(pdf_bytes)
        return text
    
    async def _extract_text_from_pdf(self, pdf_bytes: bytes) -> Optional[str]:
        """
        Extract text from PDF bytes using PyMuPDF (fitz).
        
        Falls back to basic extraction if structured extraction fails.
        """
        try:
            import fitz
        except ImportError:
            print("PyMuPDF (fitz) not installed. Install with: pip install pymupdf")
            return None
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(pdf_bytes)
                tmp_path = tmp_file.name
            
            try:
                doc = fitz.open(tmp_path)
                text_parts = []
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("text")
                    if text.strip():
                        text_parts.append(text)
                
                doc.close()
                
                full_text = "\n\n".join(text_parts)
                full_text = self._clean_extracted_text(full_text)
                
                return full_text
                
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return None
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted PDF text."""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'-\n', '', text)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
            elif cleaned_lines and cleaned_lines[-1] != '':
                cleaned_lines.append('')
        
        return '\n'.join(cleaned_lines)
    
    async def search_papers(
        self,
        query: str,
        max_results: int = 10,
        category: Optional[str] = None,
        sort_by: str = "relevance",
        sort_order: str = "descending"
    ) -> list[dict]:
        """
        Search for papers on arXiv.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            category: Optional category filter (e.g., 'q-bio', 'cs.AI')
            sort_by: Sort method ('relevance', 'lastUpdatedDate', 'submittedDate')
            sort_order: Sort order ('ascending', 'descending')
            
        Returns:
            List of paper metadata dictionaries.
        """
        await self._rate_limit()
        
        search_query = f"all:{query}"
        if category:
            search_query = f"cat:{category} AND ({search_query})"
        
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order
        }
        
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.BASE_URL}?{query_string}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        print(f"ArXiv search returned status {response.status}")
                        return []
                    
                    text = await response.text()
                    return self._parse_search_results(text)
                    
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            return []
    
    def _parse_search_results(self, xml_text: str) -> list[dict]:
        """Parse search results from arXiv API XML response."""
        results = []
        
        try:
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            root = ET.fromstring(xml_text)
            
            for entry in root.findall('atom:entry', ns):
                id_elem = entry.find('atom:id', ns)
                if id_elem is None:
                    continue
                
                arxiv_url = id_elem.text
                match = re.search(r'abs/(.+?)(?:v\d+)?$', arxiv_url)
                if not match:
                    continue
                
                arxiv_id = match.group(1)
                
                title_elem = entry.find('atom:title', ns)
                title = title_elem.text.strip().replace('\n', ' ') if title_elem is not None else ""
                
                summary_elem = entry.find('atom:summary', ns)
                abstract = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else ""
                
                authors = []
                for author in entry.findall('atom:author', ns):
                    name_elem = author.find('atom:name', ns)
                    if name_elem is not None:
                        authors.append(name_elem.text.strip())
                
                published_elem = entry.find('atom:published', ns)
                year = None
                if published_elem is not None:
                    try:
                        year = int(published_elem.text[:4])
                    except (ValueError, TypeError):
                        pass
                
                results.append({
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "year": year,
                    "url": f"{self.ABS_BASE_URL}/{arxiv_id}",
                    "pdf_url": f"{self.PDF_BASE_URL}/{arxiv_id}.pdf"
                })
                
        except ET.ParseError as e:
            print(f"XML parse error in search results: {e}")
        
        return results
    
    async def fetch_paper_citations(self, arxiv_id: str) -> list[str]:
        """
        Extract citation references from paper text.
        
        Note: arXiv API doesn't provide citation data directly.
        This extracts DOIs and arXiv IDs from the paper text.
        For comprehensive citation data, use Semantic Scholar.
        
        Args:
            arxiv_id: The arXiv paper ID
            
        Returns:
            List of extracted reference identifiers (DOIs, arXiv IDs).
        """
        text = await self.fetch_paper_text(arxiv_id)
        if not text:
            return []
        
        references = []
        
        doi_pattern = r'10\.\d{4,}/[^\s\]\)\}]+'
        dois = re.findall(doi_pattern, text)
        references.extend([f"doi:{d.rstrip('.,;')}" for d in dois])
        
        arxiv_pattern = r'arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)'
        arxiv_refs = re.findall(arxiv_pattern, text, re.IGNORECASE)
        references.extend([f"arxiv:{a}" for a in arxiv_refs])
        
        return list(set(references))


def generate_claim_id(text: str) -> str:
    """Generate a claim ID from text using SHA256 hash."""
    hash_value = hashlib.sha256(text.encode()).hexdigest()[:12]
    return f"claim_{hash_value}"
