#!/usr/bin/env python3
"""
Graph Building Pipeline Runner

Standalone script to run the knowledge graph building pipeline with
detailed logging and progress visualization.

Usage:
    python scripts/run_graph_build.py [options]

Options:
    --max-depth N       Maximum citation traversal depth (default: 3)
    --max-claims N      Maximum claims to extract (default: 500)
    --workers N         Parallel workers for expansion (default: 5)
    --papers ID1,ID2    Comma-separated arXiv IDs (default: use anchor_papers.json)
    --dry-run           Show what would be processed without running
    --verbose           Extra detailed logging
"""

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def print_banner():
    """Print startup banner."""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   🧬 ClaimGraph Knowledge Graph Builder                       ║
    ║   Building verified scientific knowledge, one claim at a time ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


def check_environment():
    """Check required environment variables."""
    print("🔍 Checking environment...")
    
    required = {
        "OPENAI_API_KEY": "For claim extraction and embeddings",
        "SUPABASE_URL": "Database connection",
        "SUPABASE_KEY": "Database authentication",
    }
    
    missing = []
    for var, desc in required.items():
        value = os.getenv(var)
        if value:
            masked = value[:8] + "..." if len(value) > 8 else "***"
            print(f"   ✓ {var}: {masked}")
        else:
            print(f"   ✗ {var}: MISSING ({desc})")
            missing.append(var)
    
    if missing:
        print(f"\n❌ Missing environment variables: {', '.join(missing)}")
        print("   Create a .env file in the backend/ directory with these values.")
        return False
    
    print("   ✓ All environment variables configured\n")
    return True


def load_anchor_papers(extended: bool = False):
    """Load anchor papers from JSON file."""
    import json
    
    filename = "anchor_papers_extended.json" if extended else "anchor_papers.json"
    anchor_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", filename
    )
    
    try:
        with open(anchor_path, "r") as f:
            data = json.load(f)
            papers = data.get("anchor_papers", [])
            papers.sort(key=lambda x: x.get("priority", 99))
            return papers
    except FileNotFoundError:
        print(f"❌ {filename} not found at {anchor_path}")
        if extended:
            print("   Falling back to anchor_papers.json...")
            return load_anchor_papers(extended=False)
        return []
    except Exception as e:
        print(f"❌ Error loading anchor papers: {e}")
        return []


async def search_for_papers(query: str, limit: int = 20):
    """Dynamically search for papers to seed the graph."""
    from clients.arxiv_client import ArxivClient
    
    print(f"🔍 Searching arXiv for: {query}")
    client = ArxivClient()
    
    results = await client.search_papers(
        query=query,
        max_results=limit,
        category="q-bio",  # Quantitative Biology
        sort_by="relevance"
    )
    
    papers = []
    for i, r in enumerate(results):
        papers.append({
            "arxiv_id": r["arxiv_id"],
            "title": r.get("title", "")[:60],
            "priority": 1 if i < 5 else 2 if i < 15 else 3,
            "subfield": "search_result"
        })
    
    print(f"   Found {len(papers)} papers")
    return papers


def show_anchor_papers(papers: list, dry_run: bool = False):
    """Display anchor papers that will be processed."""
    print(f"📚 Anchor Papers ({len(papers)} total):")
    print("-" * 70)
    
    for i, paper in enumerate(papers, 1):
        arxiv_id = paper.get("arxiv_id", "unknown")
        title = paper.get("title", "Untitled")[:50]
        priority = paper.get("priority", "?")
        subfield = paper.get("subfield", "?")
        expected = paper.get("expected_claims", "?")
        
        print(f"   {i:2}. [{arxiv_id}] (P{priority}) {title}...")
        print(f"       Subfield: {subfield}, Expected claims: ~{expected}")
    
    print("-" * 70)
    
    if dry_run:
        print("\n🔸 DRY RUN - No changes will be made")
        return


async def run_build(
    anchor_papers: list[str],
    max_depth: int,
    max_claims: int,
    parallel_workers: int,
    verbose: bool = False,
):
    """Run the graph building pipeline."""
    from supabase import create_client
    from graph.graph_cache import init_graph_cache
    from graph.graph_builder import GraphBuilder
    
    # Initialize Supabase
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    supabase = create_client(supabase_url, supabase_key)
    
    # Initialize graph cache
    print("🔗 Connecting to database...")
    graph_cache = init_graph_cache(supabase)
    
    # Load existing graph state
    existing_nodes, existing_edges = await graph_cache.load_from_db()
    print(f"   Existing graph: {existing_nodes} claims, {existing_edges} edges\n")
    
    # Create builder
    builder = GraphBuilder(
        graph_cache=graph_cache,
        supabase_client=supabase,
    )
    
    # Run the build
    print("=" * 70)
    print("🚀 STARTING GRAPH BUILD")
    print("=" * 70)
    print(f"   Max Depth: {max_depth}")
    print(f"   Max Claims: {max_claims}")
    print(f"   Parallel Workers: {parallel_workers}")
    print(f"   Anchor Papers: {len(anchor_papers)}")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    result = await builder.build_graph(
        anchor_papers=anchor_papers,
        max_depth=max_depth,
        max_claims=max_claims,
        parallel_workers=parallel_workers,
    )
    
    elapsed = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 BUILD COMPLETE")
    print("=" * 70)
    print(f"   Status: {result.status.upper()}")
    print(f"   Duration: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"   Papers Processed: {result.papers_processed}")
    print(f"   Claims Extracted: {result.claims_extracted}")
    print(f"   Edges Created: {result.edges_created}")
    
    if result.errors:
        print(f"\n   ⚠️  Errors: {len(result.errors)}")
        for err in result.errors[:5]:
            print(f"      - {err}")
    
    print("=" * 70)
    
    # Reload from DB to get accurate final stats
    print("\n🔄 Reloading from database for accurate stats...")
    final_nodes, final_edges = await graph_cache.load_from_db()
    print(f"   Loaded: {final_nodes} claims, {final_edges} edges")
    
    # Show final graph stats
    stats = graph_cache.get_statistics()
    claims_by_type = stats.get('claims_by_type', {})
    edges_by_type = stats.get('edges_by_type', {})
    
    print("\n📈 Final Graph Statistics:")
    print(f"   Total Claims: {stats.get('total_claims', 0)}")
    print(f"   Total Edges: {stats.get('total_edges', 0)}")
    print(f"   Papers Indexed: {stats.get('papers_indexed', 0)}")
    print(f"   Grounded: {stats.get('grounded_percentage', 0)*100:.1f}%")
    print(f"\n   Claims by Type:")
    print(f"      Ground Truths: {claims_by_type.get('ground_truth', 0)}")
    print(f"      Empirical: {claims_by_type.get('empirical', 0)}")
    print(f"      Unsupported: {claims_by_type.get('unsupported', 0)}")
    print(f"\n   Edges by Type:")
    print(f"      Supports: {edges_by_type.get('supports', 0)}")
    print(f"      Contradicts: {edges_by_type.get('contradicts', 0)}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run the ClaimGraph knowledge graph building pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--max-depth", "-d",
        type=int,
        default=3,
        help="Maximum citation traversal depth (default: 3)"
    )
    parser.add_argument(
        "--max-claims", "-c",
        type=int,
        default=500,
        help="Maximum claims to extract (default: 500)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=5,
        help="Parallel workers for expansion phase (default: 5)"
    )
    parser.add_argument(
        "--papers", "-p",
        type=str,
        default=None,
        help="Comma-separated arXiv IDs to process (overrides anchor_papers.json)"
    )
    parser.add_argument(
        "--extended", "-e",
        action="store_true",
        help="Use extended anchor papers list (50 papers instead of 12)"
    )
    parser.add_argument(
        "--search", "-s",
        type=str,
        default=None,
        help="Search query to find papers dynamically (e.g., 'protein folding')"
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=20,
        help="Max papers to find when using --search (default: 20)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Load anchor papers
    if args.papers:
        paper_ids = [p.strip() for p in args.papers.split(",")]
        anchor_papers_data = [{"arxiv_id": pid, "title": f"Paper {pid}", "priority": 1} for pid in paper_ids]
    elif args.search:
        anchor_papers_data = asyncio.run(search_for_papers(args.search, args.search_limit))
    else:
        anchor_papers_data = load_anchor_papers(extended=args.extended)
        if args.extended:
            print(f"📚 Using EXTENDED paper list (50 papers)\n")
    
    if not anchor_papers_data:
        print("❌ No anchor papers to process")
        sys.exit(1)
    
    # Show what will be processed
    show_anchor_papers(anchor_papers_data, args.dry_run)
    
    if args.dry_run:
        print("\n✅ Dry run complete. Remove --dry-run to execute.")
        sys.exit(0)
    
    # Extract just the IDs for the builder
    paper_ids = [p["arxiv_id"] for p in anchor_papers_data]
    
    # Run the build
    try:
        result = asyncio.run(run_build(
            anchor_papers=paper_ids,
            max_depth=args.max_depth,
            max_claims=args.max_claims,
            parallel_workers=args.workers,
            verbose=args.verbose,
        ))
        
        if result.status == "completed":
            print("\n✅ Graph build completed successfully!")
            sys.exit(0)
        else:
            print("\n⚠️  Graph build completed with errors")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Build interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
