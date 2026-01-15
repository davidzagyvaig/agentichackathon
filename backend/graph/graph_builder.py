"""
Graph Builder Pipeline

Orchestrates the three-phase graph building process:
1. Seed Phase: Process anchor papers sequentially
2. Expand Phase: Parallel workers process citation queue
3. Analyze Phase: Compute depths, detect cycles, generate stats

Implements PRD Section 4: Graph Building Pipeline.
"""

import asyncio
import json
import os
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, field

from clients.arxiv_client import ArxivClient
from clients.semantic_scholar import SemanticScholarClient
from clients.embeddings import EmbeddingsClient
from graph.kg_claim_extractor import ClaimExtractor
from graph.graph_cache import GraphCache
from models.kg_schemas import (
    ClaimNode,
    ClaimEdge,
    BuildRun,
    BuildRunConfig,
    BuildRunProgress,
)


# Keywords for relevance filtering (from PRD Section 12.4)
ANCHOR_KEYWORDS = {
    # Core biology
    "gene", "protein", "cell", "genome", "transcriptome", "proteome",
    "dna", "rna", "mrna", "expression", "regulation", "pathway",
    "metabolic", "metabolism", "enzyme", "amino acid",
    
    # Microbiome specific
    "microbiome", "microbiota", "gut", "intestinal", "bacteria", "microbial",
    "scfa", "short-chain fatty acid", "butyrate", "propionate", "acetate",
    "vagus", "gut-brain", "enteric", "colonocyte", "dysbiosis",
    
    # Single-cell genomics
    "single-cell", "scrna-seq", "rna-seq", "transcriptomics", "spatial",
    "variational", "autoencoder", "vae", "latent", "embedding",
    
    # Structural biology
    "protein structure", "folding", "conformation", "alphafold",
    "language model", "plm", "esm", "sequence",
    
    # Cheminformatics / Drug discovery
    "molecular", "molecule", "drug", "compound", "property prediction",
    "graph neural", "gnn", "message passing", "mpnn",
    "binding", "affinity", "toxicity", "solubility",
    
    # Systems biology / Quantitative methods
    "network", "dynamics", "dynamical", "differential equation",
    "stochastic", "bayesian", "inference", "parameter",
    "model", "simulation", "computational",
    
    # Machine learning general
    "neural network", "deep learning", "machine learning", "prediction",
    "classification", "clustering", "attention", "transformer"
}


@dataclass
class CitationQueueItem:
    """An item in the citation expansion queue."""
    paper_id: str  # arXiv ID or DOI
    source: str  # "arxiv", "semantic_scholar", "doi"
    depth: int
    priority: int = 0
    parent_paper_id: Optional[str] = None


@dataclass
class BuildResult:
    """Result of a graph build operation."""
    build_id: int
    status: str
    claims_extracted: int
    edges_created: int
    papers_processed: int
    errors: list = field(default_factory=list)
    duration_seconds: float = 0.0


class GraphBuilder:
    """
    Orchestrates the graph building pipeline.
    """
    
    DEFAULT_MAX_DEPTH = 3
    DEFAULT_MAX_CLAIMS = 500
    DEFAULT_PARALLEL_WORKERS = 5
    DEFAULT_RELEVANCE_THRESHOLD = 0.15
    
    def __init__(
        self,
        graph_cache: GraphCache,
        supabase_client=None,
        arxiv_client: Optional[ArxivClient] = None,
        semantic_scholar_client: Optional[SemanticScholarClient] = None,
        embeddings_client: Optional[EmbeddingsClient] = None,
        claim_extractor: Optional[ClaimExtractor] = None,
    ):
        """
        Initialize the graph builder.
        
        Args:
            graph_cache: The graph cache instance.
            supabase_client: Supabase client for persistence.
            arxiv_client: ArXiv API client.
            semantic_scholar_client: Semantic Scholar API client.
            embeddings_client: Embeddings generation client.
            claim_extractor: Claim extraction service.
        """
        self.graph_cache = graph_cache
        self.supabase = supabase_client
        
        self.arxiv_client = arxiv_client or ArxivClient()
        self.semantic_scholar_client = semantic_scholar_client or SemanticScholarClient()
        self.embeddings_client = embeddings_client or EmbeddingsClient()
        self.claim_extractor = claim_extractor or ClaimExtractor()
        
        self._build_run: Optional[BuildRun] = None
        self._citation_queue: list[CitationQueueItem] = []
        self._processed_papers: set[str] = set()
        self._total_claims: int = 0
        self._total_edges: int = 0
    
    async def build_graph(
        self,
        anchor_papers: Optional[list[str]] = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        max_claims: int = DEFAULT_MAX_CLAIMS,
        parallel_workers: int = DEFAULT_PARALLEL_WORKERS,
    ) -> BuildResult:
        """
        Execute the full graph building pipeline.
        
        Args:
            anchor_papers: List of arXiv IDs to seed the graph. If None, loads from anchor_papers.json.
            max_depth: Maximum citation traversal depth.
            max_claims: Maximum total claims to extract.
            parallel_workers: Number of parallel workers for expansion.
            
        Returns:
            BuildResult with statistics.
        """
        start_time = datetime.utcnow()
        
        if anchor_papers is None:
            anchor_papers = self._load_anchor_papers()
        
        self._build_run = await self._create_build_run(
            anchor_papers,
            BuildRunConfig(
                max_depth=max_depth,
                max_claims=max_claims,
                parallel_workers=parallel_workers,
            )
        )
        
        self._citation_queue = []
        self._processed_papers = set()
        self._total_claims = 0
        self._total_edges = 0
        errors = []
        
        try:
            print(f"📚 Phase 1: Seeding with {len(anchor_papers)} anchor papers (parallel, workers={parallel_workers})...")
            await self._seed_phase(anchor_papers, max_claims, parallel_workers)
            
            print(f"🔄 Phase 2: Expanding citations (depth={max_depth}, workers={parallel_workers})...")
            await self._expand_phase(max_depth, max_claims, parallel_workers)
            
            print("📊 Phase 3: Analyzing graph...")
            await self._analyze_phase()
            
            await self._update_build_run("completed")
            
        except Exception as e:
            print(f"❌ Build failed: {e}")
            errors.append({"error": str(e), "phase": "unknown"})
            await self._update_build_run("failed", errors)
        
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        return BuildResult(
            build_id=self._build_run.id if self._build_run else 0,
            status="completed" if not errors else "failed",
            claims_extracted=self._total_claims,
            edges_created=self._total_edges,
            papers_processed=len(self._processed_papers),
            errors=errors,
            duration_seconds=duration,
        )
    
    async def _seed_phase(self, anchor_papers: list[str], max_claims: int, parallel_workers: int):
        """
        Phase 1: Process anchor papers in parallel.
        """
        semaphore = asyncio.Semaphore(parallel_workers)
        results_lock = asyncio.Lock()
        
        async def process_anchor(arxiv_id: str):
            async with semaphore:
                if self._total_claims >= max_claims:
                    return
                
                print(f"   Processing anchor: {arxiv_id}")
                
                try:
                    claims, edges, citations = await self._process_paper(arxiv_id, depth=0)
                    
                    # Update shared state with lock
                    async with results_lock:
                        self._total_claims += len(claims)
                        self._total_edges += len(edges)
                        self._processed_papers.add(arxiv_id)
                        
                        for citation in citations:
                            self._citation_queue.append(CitationQueueItem(
                                paper_id=citation["id"],
                                source=citation["source"],
                                depth=1,
                                priority=citation.get("priority", 0),
                                parent_paper_id=arxiv_id,
                            ))
                    
                except Exception as e:
                    print(f"   Error processing {arxiv_id}: {e}")
        
        # Process all anchor papers in parallel
        await asyncio.gather(*[process_anchor(arxiv_id) for arxiv_id in anchor_papers])
        await self._update_build_progress()
        
        print(f"   Seed phase complete: {self._total_claims} claims from {len(self._processed_papers)} papers")
    
    async def _expand_phase(
        self,
        max_depth: int,
        max_claims: int,
        parallel_workers: int
    ):
        """
        Phase 2: Parallel workers process citation queue.
        """
        semaphore = asyncio.Semaphore(parallel_workers)
        
        async def process_citation(item: CitationQueueItem):
            async with semaphore:
                if item.paper_id in self._processed_papers:
                    return
                if self._total_claims >= max_claims:
                    return
                
                try:
                    claims, edges, citations = await self._process_paper(
                        item.paper_id,
                        depth=item.depth,
                        source=item.source
                    )
                    
                    self._total_claims += len(claims)
                    self._total_edges += len(edges)
                    self._processed_papers.add(item.paper_id)
                    
                    if item.depth < max_depth:
                        for citation in citations:
                            if citation["id"] not in self._processed_papers:
                                self._citation_queue.append(CitationQueueItem(
                                    paper_id=citation["id"],
                                    source=citation["source"],
                                    depth=item.depth + 1,
                                    parent_paper_id=item.paper_id,
                                ))
                    
                except Exception as e:
                    print(f"   Error expanding {item.paper_id}: {e}")
        
        while self._citation_queue and self._total_claims < max_claims:
            batch_size = min(parallel_workers, len(self._citation_queue))
            batch = [self._citation_queue.pop(0) for _ in range(batch_size)]
            
            await asyncio.gather(*[process_citation(item) for item in batch])
            await self._update_build_progress()
            
            print(f"   Progress: {self._total_claims} claims, {len(self._processed_papers)} papers, {len(self._citation_queue)} in queue")
    
    async def _analyze_phase(self):
        """
        Phase 3: Compute graph metrics and update claims.
        """
        for claim_id in list(self.graph_cache.G.nodes):
            depth = self.graph_cache.compute_depth_to_ground_truth(claim_id)
            
            if depth is not None:
                self.graph_cache.G.nodes[claim_id]["depth_to_ground_truth"] = depth
                
                if self.supabase:
                    try:
                        self.supabase.table("kg_claims").update({
                            "depth_to_ground_truth": depth
                        }).eq("id", claim_id).execute()
                    except Exception as e:
                        print(f"   Error updating depth for {claim_id}: {e}")
        
        cycles = self.graph_cache.detect_cycles()
        if cycles:
            print(f"   Detected {len(cycles)} cycles (circular reasoning)")
        
        stats = self.graph_cache.get_statistics()
        print(f"   Final stats: {stats['total_claims']} claims, {stats['total_edges']} edges")
        print(f"   Grounded: {stats['grounded_percentage']*100:.1f}%")
    
    async def _process_paper(
        self,
        paper_id: str,
        depth: int,
        source: str = "arxiv"
    ) -> tuple[list[ClaimNode], list[ClaimEdge], list[dict]]:
        """
        Process a single paper: construct PDF URL, perform JOINT extraction (claims + edges in one call).
        
        Uses file_url to pass PDF directly to OpenAI API (no download needed).
        
        Returns:
            Tuple of (claims, edges, citations_to_expand).
        """
        if source == "arxiv" or source == "":
            metadata = await self.arxiv_client.fetch_paper_metadata(paper_id)
            if not metadata:
                return [], [], []
            
            # Construct PDF URL directly (no download needed)
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            
            print(f"      📄 Joint extraction from PDF URL: {pdf_url}")
            # SINGLE API CALL with file_url: extracts, classifies, and detects edges
            claims, edges = await self.claim_extractor.extract_claims_from_pdf(
                pdf_url,
                metadata,
                max_claims=20
            )
            
            # Fallback to abstract if PDF extraction returned nothing
            if not claims:
                from graph.kg_claim_extractor import extract_claims_from_abstract
                abstract = metadata.get("abstract", "")
                if abstract:
                    print(f"      📝 PDF extraction failed, using abstract only")
                    claims = await extract_claims_from_abstract(abstract, metadata)
                edges = []
        else:
            metadata = await self.semantic_scholar_client.get_paper_by_arxiv_id(paper_id)
            if not metadata:
                metadata = await self.semantic_scholar_client.get_paper_by_doi(paper_id)
            
            if not metadata:
                return [], [], []
            
            from graph.kg_claim_extractor import extract_claims_from_abstract
            abstract = metadata.get("abstract", "")
            if abstract:
                claims = await extract_claims_from_abstract(abstract, {
                    "arxiv_id": metadata.get("arxiv_id"),
                    "title": metadata.get("title", ""),
                    "authors": metadata.get("authors", []),
                    "year": metadata.get("year"),
                    "venue": metadata.get("venue"),
                    "doi": metadata.get("doi"),
                })
            else:
                claims = []
            edges = []
        
        if claims:
            texts = [c.text for c in claims]
            embeddings = await self.embeddings_client.get_embeddings_batch(texts)
            for i, claim in enumerate(claims):
                if embeddings[i]:
                    claim.embedding = embeddings[i]
        
        for claim in claims:
            await self.graph_cache.add_claim(claim)
        
        for edge in edges:
            await self.graph_cache.add_edge(edge)
        
        citations = await self._get_relevant_citations(paper_id, metadata, depth)
        
        return claims, edges, citations
    
    async def _get_relevant_citations(
        self,
        paper_id: str,
        metadata: dict,
        current_depth: int
    ) -> list[dict]:
        """
        Get citations from a paper, filtered by relevance.
        """
        if current_depth >= self.DEFAULT_MAX_DEPTH - 1:
            return []
        
        try:
            references = await self.semantic_scholar_client.get_references(
                f"ARXIV:{paper_id}",
                limit=50
            )
        except:
            references = []
        
        relevant = []
        for ref in references:
            ref_arxiv = ref.get("arxiv_id")
            ref_doi = ref.get("doi")
            ref_title = ref.get("title", "")
            ref_abstract = ref.get("abstract", "")
            
            if not ref_arxiv and not ref_doi:
                continue
            
            if ref_arxiv and ref_arxiv in self._processed_papers:
                continue
            
            text = f"{ref_title} {ref_abstract}".lower()
            overlap = self._compute_keyword_overlap(text)
            
            if overlap >= self.DEFAULT_RELEVANCE_THRESHOLD:
                relevant.append({
                    "id": ref_arxiv or ref_doi,
                    "source": "arxiv" if ref_arxiv else "doi",
                    "priority": ref.get("citation_count", 0),
                    "overlap": overlap,
                })
        
        relevant.sort(key=lambda x: x["priority"], reverse=True)
        return relevant[:20]
    
    def _compute_keyword_overlap(self, text: str) -> float:
        """Compute keyword overlap with anchor keywords."""
        text_lower = text.lower()
        matches = sum(1 for kw in ANCHOR_KEYWORDS if kw in text_lower)
        return matches / len(ANCHOR_KEYWORDS) if ANCHOR_KEYWORDS else 0.0
    
    def _load_anchor_papers(self) -> list[str]:
        """Load anchor papers from JSON file."""
        anchor_path = os.path.join(
            os.path.dirname(__file__),
            "..", "data", "anchor_papers.json"
        )
        
        try:
            with open(anchor_path, "r") as f:
                data = json.load(f)
                papers = data.get("anchor_papers", [])
                papers.sort(key=lambda x: x.get("priority", 99))
                return [p["arxiv_id"] for p in papers]
        except FileNotFoundError:
            print(f"Warning: anchor_papers.json not found at {anchor_path}")
            return []
        except Exception as e:
            print(f"Error loading anchor papers: {e}")
            return []
    
    async def _create_build_run(
        self,
        anchor_papers: list[str],
        config: BuildRunConfig
    ) -> BuildRun:
        """Create a new build run record in the database."""
        build_run = BuildRun(
            anchor_papers=anchor_papers,
            config=config,
            status="running",
        )
        
        if self.supabase:
            try:
                response = self.supabase.table("kg_build_runs").insert(
                    build_run.to_db_dict()
                ).execute()
                if response.data:
                    build_run.id = response.data[0]["id"]
            except Exception as e:
                print(f"Error creating build run record: {e}")
        
        return build_run
    
    async def _update_build_run(
        self,
        status: str,
        errors: Optional[list] = None
    ):
        """Update build run status in database."""
        if not self._build_run or not self.supabase:
            return
        
        update_data = {
            "status": status,
            "claims_extracted": self._total_claims,
            "edges_created": self._total_edges,
            "papers_processed": len(self._processed_papers),
        }
        
        if status in ["completed", "failed"]:
            update_data["completed_at"] = datetime.utcnow().isoformat()
        
        if errors:
            update_data["errors"] = errors
        
        try:
            self.supabase.table("kg_build_runs").update(
                update_data
            ).eq("id", self._build_run.id).execute()
        except Exception as e:
            print(f"Error updating build run: {e}")
    
    async def _update_build_progress(self):
        """Update build progress in database."""
        if not self._build_run or not self.supabase:
            return
        
        try:
            self.supabase.table("kg_build_runs").update({
                "claims_extracted": self._total_claims,
                "edges_created": self._total_edges,
                "papers_processed": len(self._processed_papers),
            }).eq("id", self._build_run.id).execute()
        except:
            pass


async def build_knowledge_graph(
    supabase_client=None,
    anchor_papers: Optional[list[str]] = None,
    max_depth: int = 3,
    max_claims: int = 500,
) -> BuildResult:
    """
    Convenience function to build the knowledge graph.
    
    Args:
        supabase_client: Supabase client for persistence.
        anchor_papers: List of arXiv IDs to seed (uses defaults if None).
        max_depth: Maximum citation traversal depth.
        max_claims: Maximum claims to extract.
        
    Returns:
        BuildResult with build statistics.
    """
    from graph.graph_cache import init_graph_cache
    
    graph_cache = init_graph_cache(supabase_client)
    
    builder = GraphBuilder(
        graph_cache=graph_cache,
        supabase_client=supabase_client,
    )
    
    return await builder.build_graph(
        anchor_papers=anchor_papers,
        max_depth=max_depth,
        max_claims=max_claims,
    )
