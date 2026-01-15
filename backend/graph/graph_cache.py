"""
Graph Cache Layer

In-memory NetworkX graph that syncs with Supabase for fast traversal queries.
Implements PRD Section 5: Storage Architecture.
"""

import asyncio
import networkx as nx
from typing import Optional
from datetime import datetime

from models.kg_schemas import ClaimNode, ClaimEdge


class GraphCache:
    """
    In-memory NetworkX graph that syncs with Supabase.
    
    Provides sub-millisecond traversal operations while maintaining
    consistency with the persistent Supabase storage.
    """
    
    def __init__(self, supabase_client=None):
        """
        Initialize the graph cache.
        
        Args:
            supabase_client: Supabase client instance for persistence.
        """
        self.G = nx.DiGraph()
        self.supabase = supabase_client
        self._loaded = False
        self._lock = asyncio.Lock()
        self._last_sync: Optional[datetime] = None
    
    @property
    def is_loaded(self) -> bool:
        """Check if graph has been loaded from database."""
        return self._loaded
    
    async def load_from_db(self) -> tuple[int, int]:
        """
        Load entire graph from Supabase into NetworkX.
        
        Returns:
            Tuple of (nodes_loaded, edges_loaded).
        """
        if not self.supabase:
            print("Warning: No Supabase client configured")
            return 0, 0
        
        async with self._lock:
            self.G.clear()
            
            try:
                claims_response = self.supabase.table("kg_claims").select("*").execute()
                claims = claims_response.data or []
                
                for claim in claims:
                    self.G.add_node(
                        claim["id"],
                        text=claim["text"],
                        original_text=claim.get("original_text"),
                        type=claim["type"],
                        confidence=claim["confidence"],
                        source_paper=claim["source_paper"],
                        external_source=claim.get("external_source"),
                        extraction=claim["extraction"],
                        support_count=claim.get("support_count", 0),
                        contradict_count=claim.get("contradict_count", 0),
                        depth_to_ground_truth=claim.get("depth_to_ground_truth"),
                        created_at=claim.get("created_at"),
                    )
                
                edges_response = self.supabase.table("kg_edges").select("*").execute()
                edges = edges_response.data or []
                
                for edge in edges:
                    if edge["source_id"] in self.G and edge["target_id"] in self.G:
                        self.G.add_edge(
                            edge["source_id"],
                            edge["target_id"],
                            id=edge["id"],
                            type=edge["type"],
                            weight=edge["weight"],
                            reasoning=edge.get("reasoning"),
                            model=edge.get("model"),
                            created_at=edge.get("created_at"),
                        )
                
                self._loaded = True
                self._last_sync = datetime.utcnow()
                
                return len(claims), len(edges)
                
            except Exception as e:
                print(f"Error loading graph from DB: {e}")
                raise
    
    async def add_claim(self, claim: ClaimNode) -> bool:
        """
        Write-through: add claim to both cache and database.
        
        Args:
            claim: The claim node to add.
            
        Returns:
            True if successful, False otherwise.
        """
        async with self._lock:
            if self.supabase:
                try:
                    self.supabase.table("kg_claims").upsert(
                        claim.to_db_dict()
                    ).execute()
                except Exception as e:
                    print(f"Error saving claim to DB: {e}")
                    return False
            
            self.G.add_node(
                claim.id,
                text=claim.text,
                original_text=claim.original_text,
                type=claim.type,
                confidence=claim.confidence,
                source_paper=claim.source_paper.model_dump(),
                external_source=claim.external_source.model_dump() if claim.external_source else None,
                extraction=claim.extraction.model_dump(),
                support_count=claim.support_count,
                contradict_count=claim.contradict_count,
                depth_to_ground_truth=claim.depth_to_ground_truth,
            )
            
            return True
    
    async def add_edge(self, edge: ClaimEdge) -> bool:
        """
        Write-through: add edge to both cache and database.
        
        Args:
            edge: The edge to add.
            
        Returns:
            True if successful, False otherwise.
        """
        if edge.source_id not in self.G or edge.target_id not in self.G:
            print(f"Cannot add edge: source or target node not found")
            return False
        
        async with self._lock:
            if self.supabase:
                try:
                    self.supabase.table("kg_edges").upsert(
                        edge.to_db_dict()
                    ).execute()
                except Exception as e:
                    print(f"Error saving edge to DB: {e}")
                    return False
            
            self.G.add_edge(
                edge.source_id,
                edge.target_id,
                id=edge.id,
                type=edge.type,
                weight=edge.weight,
                reasoning=edge.reasoning,
                model=edge.model,
            )
            
            return True
    
    async def add_claims_batch(self, claims: list[ClaimNode]) -> int:
        """
        Add multiple claims in a batch.
        
        Args:
            claims: List of claims to add.
            
        Returns:
            Number of successfully added claims.
        """
        success_count = 0
        for claim in claims:
            if await self.add_claim(claim):
                success_count += 1
        return success_count
    
    async def add_edges_batch(self, edges: list[ClaimEdge]) -> int:
        """
        Add multiple edges in a batch.
        
        Args:
            edges: List of edges to add.
            
        Returns:
            Number of successfully added edges.
        """
        success_count = 0
        for edge in edges:
            if await self.add_edge(edge):
                success_count += 1
        return success_count
    
    # ========================================================================
    # Query Methods (use NetworkX for fast traversal)
    # ========================================================================
    
    def get_claim(self, claim_id: str) -> Optional[dict]:
        """Get a claim by ID."""
        if claim_id not in self.G:
            return None
        return {"id": claim_id, **self.G.nodes[claim_id]}
    
    def get_supporters(
        self,
        claim_id: str,
        depth: int = 1,
        edge_type: str = "supports"
    ) -> list[dict]:
        """
        Get claims that support the given claim.
        
        Args:
            claim_id: The claim to find supporters for.
            depth: How many hops to traverse (1 = direct supporters only).
            edge_type: "supports" or "contradicts".
            
        Returns:
            List of supporting claims with their edge data.
        """
        if claim_id not in self.G:
            return []
        
        results = []
        
        if depth == 1:
            for pred in self.G.predecessors(claim_id):
                edge_data = self.G[pred][claim_id]
                if edge_data.get("type") == edge_type:
                    claim_data = self.G.nodes[pred]
                    results.append({
                        "claim": {"id": pred, **claim_data},
                        "edge": edge_data,
                        "depth": 1
                    })
        else:
            reversed_g = self.G.reverse()
            visited = set()
            queue = [(claim_id, 0)]
            
            while queue:
                current, current_depth = queue.pop(0)
                if current_depth >= depth:
                    continue
                
                for neighbor in reversed_g.neighbors(current):
                    if neighbor in visited:
                        continue
                    
                    edge_data = self.G[neighbor][current]
                    if edge_data.get("type") != edge_type:
                        continue
                    
                    visited.add(neighbor)
                    claim_data = self.G.nodes[neighbor]
                    results.append({
                        "claim": {"id": neighbor, **claim_data},
                        "edge": edge_data,
                        "depth": current_depth + 1
                    })
                    queue.append((neighbor, current_depth + 1))
        
        return results
    
    def get_contradictions(self, claim_id: str) -> list[dict]:
        """Get claims that contradict the given claim."""
        return self.get_supporters(claim_id, depth=1, edge_type="contradicts")
    
    def get_supported_claims(
        self,
        claim_id: str,
        depth: int = 1,
        edge_type: str = "supports"
    ) -> list[dict]:
        """
        Get claims that the given claim supports.
        
        Args:
            claim_id: The claim providing support.
            depth: How many hops to traverse.
            edge_type: "supports" or "contradicts".
            
        Returns:
            List of supported claims with edge data.
        """
        if claim_id not in self.G:
            return []
        
        results = []
        
        if depth == 1:
            for succ in self.G.successors(claim_id):
                edge_data = self.G[claim_id][succ]
                if edge_data.get("type") == edge_type:
                    claim_data = self.G.nodes[succ]
                    results.append({
                        "claim": {"id": succ, **claim_data},
                        "edge": edge_data,
                        "depth": 1
                    })
        
        return results
    
    def trace_to_ground_truths(
        self,
        claim_id: str,
        max_depth: int = 10
    ) -> list[list[str]]:
        """
        Trace paths from a claim to ground truth claims.
        
        Args:
            claim_id: Starting claim.
            max_depth: Maximum path length.
            
        Returns:
            List of paths (each path is a list of claim IDs).
        """
        if claim_id not in self.G:
            return []
        
        ground_truths = [
            n for n in self.G.nodes 
            if self.G.nodes[n].get("type") == "ground_truth"
        ]
        
        if not ground_truths:
            return []
        
        reversed_g = self.G.reverse()
        
        all_paths = []
        for gt in ground_truths:
            try:
                paths = list(nx.all_simple_paths(
                    reversed_g, claim_id, gt, cutoff=max_depth
                ))
                all_paths.extend(paths)
            except nx.NetworkXNoPath:
                continue
            except nx.NodeNotFound:
                continue
        
        return all_paths
    
    def compute_chain_confidence(self, path: list[str]) -> float:
        """
        Compute confidence of a provenance chain as product of edge weights.
        
        A chain [A, B, C] means A←B←C (C supports B supports A).
        Confidence = weight(B→A) * weight(C→B)
        """
        if len(path) < 2:
            return 1.0
        
        confidence = 1.0
        
        for i in range(len(path) - 1):
            source = path[i + 1]
            target = path[i]
            
            if self.G.has_edge(source, target):
                edge_data = self.G[source][target]
                confidence *= edge_data.get("weight", 0.5)
            else:
                confidence *= 0.1
        
        return confidence
    
    def detect_cycles(self, claim_id: Optional[str] = None) -> list[list[str]]:
        """
        Detect cycles in the graph (circular reasoning).
        
        Args:
            claim_id: If provided, only detect cycles involving this claim.
            
        Returns:
            List of cycles (each cycle is a list of claim IDs).
        """
        try:
            if claim_id:
                ancestors = nx.ancestors(self.G, claim_id) | {claim_id}
                subgraph = self.G.subgraph(ancestors)
                cycles = list(nx.simple_cycles(subgraph))
            else:
                cycles = list(nx.simple_cycles(self.G))
            
            return cycles
        except:
            return []
    
    def compute_depth_to_ground_truth(self, claim_id: str) -> Optional[int]:
        """
        Compute shortest path length to any ground truth.
        
        Returns:
            Shortest path length or None if ungrounded.
        """
        if claim_id not in self.G:
            return None
        
        if self.G.nodes[claim_id].get("type") == "ground_truth":
            return 0
        
        reversed_g = self.G.reverse()
        
        ground_truths = [
            n for n in self.G.nodes 
            if self.G.nodes[n].get("type") == "ground_truth"
        ]
        
        min_depth = None
        for gt in ground_truths:
            try:
                path_length = nx.shortest_path_length(reversed_g, claim_id, gt)
                if min_depth is None or path_length < min_depth:
                    min_depth = path_length
            except nx.NetworkXNoPath:
                continue
        
        return min_depth
    
    def get_claims_by_paper(self, arxiv_id: str) -> list[dict]:
        """Get all claims from a specific paper."""
        results = []
        for node_id in self.G.nodes:
            node_data = self.G.nodes[node_id]
            source_paper = node_data.get("source_paper", {})
            if source_paper.get("arxiv_id") == arxiv_id:
                results.append({"id": node_id, **node_data})
        return results
    
    def get_all_ground_truths(self) -> list[dict]:
        """Get all ground truth claims."""
        results = []
        for node_id in self.G.nodes:
            node_data = self.G.nodes[node_id]
            if node_data.get("type") == "ground_truth":
                results.append({"id": node_id, **node_data})
        return results
    
    def get_ungrounded_claims(self) -> list[dict]:
        """Get all claims that have no path to ground truth."""
        results = []
        for node_id in self.G.nodes:
            node_data = self.G.nodes[node_id]
            if node_data.get("type") == "ground_truth":
                continue
            depth = self.compute_depth_to_ground_truth(node_id)
            if depth is None:
                results.append({"id": node_id, **node_data})
        return results
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    def get_statistics(self) -> dict:
        """Get graph statistics."""
        nodes = list(self.G.nodes(data=True))
        edges = list(self.G.edges(data=True))
        
        type_counts = {"empirical": 0, "ground_truth": 0, "unsupported": 0}
        edge_type_counts = {"supports": 0, "contradicts": 0}
        grounded_count = 0
        papers = set()
        
        for node_id, data in nodes:
            claim_type = data.get("type", "empirical")
            if claim_type in type_counts:
                type_counts[claim_type] += 1
            
            source_paper = data.get("source_paper", {})
            if source_paper.get("arxiv_id"):
                papers.add(source_paper["arxiv_id"])
            
            if data.get("type") != "ground_truth":
                if self.compute_depth_to_ground_truth(node_id) is not None:
                    grounded_count += 1
        
        for _, _, data in edges:
            edge_type = data.get("type")
            if edge_type in edge_type_counts:
                edge_type_counts[edge_type] += 1
        
        total_non_gt = type_counts["empirical"] + type_counts["unsupported"]
        grounded_pct = grounded_count / total_non_gt if total_non_gt > 0 else 0.0
        
        return {
            "total_claims": len(nodes),
            "claims_by_type": type_counts,
            "total_edges": len(edges),
            "edges_by_type": edge_type_counts,
            "grounded_percentage": round(grounded_pct, 4),
            "papers_indexed": len(papers),
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
        }
    
    def node_count(self) -> int:
        """Get number of nodes in graph."""
        return self.G.number_of_nodes()
    
    def edge_count(self) -> int:
        """Get number of edges in graph."""
        return self.G.number_of_edges()


_graph_cache: Optional[GraphCache] = None


def get_graph_cache() -> GraphCache:
    """Get the singleton graph cache instance."""
    global _graph_cache
    if _graph_cache is None:
        _graph_cache = GraphCache()
    return _graph_cache


def init_graph_cache(supabase_client) -> GraphCache:
    """Initialize the graph cache with a Supabase client."""
    global _graph_cache
    _graph_cache = GraphCache(supabase_client)
    return _graph_cache
