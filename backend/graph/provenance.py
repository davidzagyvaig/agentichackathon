"""
Provenance Tracer

Traces claims back to their foundational ground truths or identifies
ungrounded foundations and circular reasoning.

Implements PRD Section 9.4: trace_provenance tool specification.
"""

import networkx as nx
from typing import Optional
from dataclasses import dataclass, field

from models.kg_schemas import ClaimNode, ProvenanceChain, ProvenanceResult
from graph.graph_cache import GraphCache


@dataclass
class ProvenanceAnalysis:
    """Detailed provenance analysis for a claim."""
    claim_id: str
    is_grounded: bool
    chains: list[ProvenanceChain] = field(default_factory=list)
    ground_truths: list[dict] = field(default_factory=list)
    ungrounded_foundations: list[dict] = field(default_factory=list)
    cycles: list[list[str]] = field(default_factory=list)
    max_chain_confidence: float = 0.0
    avg_chain_confidence: float = 0.0
    shortest_path_length: Optional[int] = None


class ProvenanceTracer:
    """
    Traces provenance chains from claims to ground truths.
    
    Key capabilities:
    - Find all paths from a claim to ground truths
    - Compute chain confidence (product of edge weights)
    - Detect cycles (circular reasoning)
    - Identify ungrounded foundations
    """
    
    def __init__(self, graph_cache: GraphCache):
        """
        Initialize the provenance tracer.
        
        Args:
            graph_cache: The graph cache to use for traversal.
        """
        self.graph_cache = graph_cache
    
    def trace_provenance(
        self,
        claim_id: str,
        max_depth: int = 10
    ) -> ProvenanceResult:
        """
        Trace a claim back to its foundational ground truths.
        
        Args:
            claim_id: The claim to trace.
            max_depth: Maximum depth to search.
            
        Returns:
            ProvenanceResult with full analysis.
        """
        G = self.graph_cache.G
        
        if claim_id not in G:
            return ProvenanceResult(
                claim_id=claim_id,
                claim_text="",
                is_grounded=False,
            )
        
        claim_data = G.nodes[claim_id]
        claim_text = claim_data.get("text", "")
        
        if claim_data.get("type") == "ground_truth":
            claim_node = self._node_to_claim(claim_id, claim_data)
            return ProvenanceResult(
                claim_id=claim_id,
                claim_text=claim_text,
                is_grounded=True,
                ground_truths_found=[claim_node],
                provenance_chains=[ProvenanceChain(
                    path=[claim_id],
                    chain_confidence=1.0,
                    ground_truth_text=claim_text,
                )],
            )
        
        ground_truths = [
            n for n in G.nodes
            if G.nodes[n].get("type") == "ground_truth"
        ]
        
        reversed_g = G.reverse()
        
        all_chains = []
        found_ground_truths = []
        
        for gt_id in ground_truths:
            try:
                paths = list(nx.all_simple_paths(
                    reversed_g, claim_id, gt_id, cutoff=max_depth
                ))
                
                for path in paths:
                    confidence = self._compute_chain_confidence(path)
                    gt_text = G.nodes[gt_id].get("text", "")
                    
                    chain = ProvenanceChain(
                        path=path,
                        chain_confidence=confidence,
                        ground_truth_text=gt_text,
                    )
                    all_chains.append(chain)
                    
                    if gt_id not in [gt.id for gt in found_ground_truths]:
                        gt_node = self._node_to_claim(gt_id, G.nodes[gt_id])
                        found_ground_truths.append(gt_node)
                        
            except nx.NetworkXNoPath:
                continue
            except nx.NodeNotFound:
                continue
        
        all_chains.sort(key=lambda x: x.chain_confidence, reverse=True)
        
        ancestors = self._get_ancestors_with_supports(claim_id, max_depth)
        ungrounded = []
        
        for ancestor_id in ancestors:
            if ancestor_id == claim_id:
                continue
            
            ancestor_data = G.nodes[ancestor_id]
            
            if ancestor_data.get("type") == "ground_truth":
                continue
            
            has_supporters = any(
                G[pred][ancestor_id].get("type") == "supports"
                for pred in G.predecessors(ancestor_id)
            )
            
            if not has_supporters:
                ungrounded.append(self._node_to_claim(ancestor_id, ancestor_data))
        
        cycles = self._detect_cycles_for_claim(claim_id)
        
        is_grounded = len(all_chains) > 0
        
        return ProvenanceResult(
            claim_id=claim_id,
            claim_text=claim_text,
            is_grounded=is_grounded,
            ground_truths_found=found_ground_truths,
            ungrounded_foundations=ungrounded,
            cycles_detected=cycles,
            provenance_chains=all_chains[:20],
        )
    
    def _compute_chain_confidence(self, path: list[str]) -> float:
        """
        Compute confidence of a provenance chain as product of edge weights.
        
        The path is in reversed graph order (claim -> ... -> ground_truth),
        so actual edges in the original graph are target <- source.
        """
        if len(path) < 2:
            return 1.0
        
        G = self.graph_cache.G
        confidence = 1.0
        
        for i in range(len(path) - 1):
            target = path[i]
            source = path[i + 1]
            
            if G.has_edge(source, target):
                edge_data = G[source][target]
                
                if edge_data.get("type") == "supports":
                    confidence *= edge_data.get("weight", 0.5)
                else:
                    confidence *= 0.1
            else:
                confidence *= 0.1
        
        return confidence
    
    def _get_ancestors_with_supports(
        self,
        claim_id: str,
        max_depth: int
    ) -> set[str]:
        """Get all ancestors connected by support edges."""
        G = self.graph_cache.G
        ancestors = set()
        
        queue = [(claim_id, 0)]
        visited = {claim_id}
        
        while queue:
            current, depth = queue.pop(0)
            ancestors.add(current)
            
            if depth >= max_depth:
                continue
            
            for pred in G.predecessors(current):
                if pred in visited:
                    continue
                
                edge_data = G[pred][current]
                if edge_data.get("type") == "supports":
                    visited.add(pred)
                    queue.append((pred, depth + 1))
        
        return ancestors
    
    def _detect_cycles_for_claim(self, claim_id: str) -> list[list[str]]:
        """Detect cycles involving the given claim."""
        G = self.graph_cache.G
        
        try:
            ancestors = nx.ancestors(G, claim_id) | {claim_id}
            descendants = nx.descendants(G, claim_id) | {claim_id}
            relevant_nodes = ancestors | descendants
            
            subgraph = G.subgraph(relevant_nodes)
            
            cycles = list(nx.simple_cycles(subgraph))
            
            claim_cycles = [
                cycle for cycle in cycles
                if claim_id in cycle
            ]
            
            return claim_cycles[:10]
            
        except Exception:
            return []
    
    def _node_to_claim(self, node_id: str, node_data: dict) -> ClaimNode:
        """Convert a graph node to a ClaimNode object."""
        from models.kg_schemas import SourcePaperMetadata, ExtractionMetadata
        
        source_paper_data = node_data.get("source_paper", {})
        extraction_data = node_data.get("extraction", {})
        
        return ClaimNode(
            id=node_id,
            text=node_data.get("text", ""),
            original_text=node_data.get("original_text"),
            type=node_data.get("type", "empirical"),
            confidence=node_data.get("confidence", 0.5),
            source_paper=SourcePaperMetadata(**source_paper_data) if source_paper_data else SourcePaperMetadata(title="Unknown"),
            extraction=ExtractionMetadata(**extraction_data) if extraction_data else ExtractionMetadata(),
            support_count=node_data.get("support_count", 0),
            contradict_count=node_data.get("contradict_count", 0),
            depth_to_ground_truth=node_data.get("depth_to_ground_truth"),
        )
    
    def find_all_ungrounded_claims(self) -> list[ClaimNode]:
        """Find all claims that have no path to any ground truth."""
        G = self.graph_cache.G
        ungrounded = []
        
        for node_id in G.nodes:
            node_data = G.nodes[node_id]
            
            if node_data.get("type") == "ground_truth":
                continue
            
            depth = self.graph_cache.compute_depth_to_ground_truth(node_id)
            
            if depth is None:
                ungrounded.append(self._node_to_claim(node_id, node_data))
        
        return ungrounded
    
    def find_all_cycles(self) -> list[list[str]]:
        """Find all cycles in the graph (circular reasoning)."""
        return self.graph_cache.detect_cycles()
    
    def get_strongest_provenance_chain(
        self,
        claim_id: str,
        max_depth: int = 10
    ) -> Optional[ProvenanceChain]:
        """Get the provenance chain with highest confidence."""
        result = self.trace_provenance(claim_id, max_depth)
        
        if result.provenance_chains:
            return result.provenance_chains[0]
        
        return None
    
    def compute_grounding_score(self, claim_id: str) -> float:
        """
        Compute an overall grounding score for a claim.
        
        Score factors:
        - Whether it's grounded at all (0 if not)
        - Number of provenance chains
        - Average chain confidence
        - Shortest path length (shorter = better)
        """
        result = self.trace_provenance(claim_id)
        
        if not result.is_grounded:
            return 0.0
        
        chains = result.provenance_chains
        if not chains:
            return 0.0
        
        max_confidence = max(c.chain_confidence for c in chains)
        avg_confidence = sum(c.chain_confidence for c in chains) / len(chains)
        
        shortest_path = min(len(c.path) for c in chains)
        path_score = 1.0 / shortest_path if shortest_path > 0 else 0.0
        
        chain_count_bonus = min(len(chains) / 5.0, 1.0)
        
        score = (
            0.4 * max_confidence +
            0.2 * avg_confidence +
            0.2 * path_score +
            0.2 * chain_count_bonus
        )
        
        if result.cycles_detected:
            score *= 0.5
        
        return min(score, 1.0)
    
    def analyze_claim_foundation(self, claim_id: str) -> ProvenanceAnalysis:
        """
        Perform detailed analysis of a claim's epistemic foundation.
        """
        result = self.trace_provenance(claim_id)
        
        chains = result.provenance_chains
        
        if chains:
            max_conf = max(c.chain_confidence for c in chains)
            avg_conf = sum(c.chain_confidence for c in chains) / len(chains)
            shortest = min(len(c.path) for c in chains)
        else:
            max_conf = 0.0
            avg_conf = 0.0
            shortest = None
        
        return ProvenanceAnalysis(
            claim_id=claim_id,
            is_grounded=result.is_grounded,
            chains=chains,
            ground_truths=[gt.model_dump() for gt in result.ground_truths_found],
            ungrounded_foundations=[uf.model_dump() for uf in result.ungrounded_foundations],
            cycles=result.cycles_detected,
            max_chain_confidence=max_conf,
            avg_chain_confidence=avg_conf,
            shortest_path_length=shortest,
        )
