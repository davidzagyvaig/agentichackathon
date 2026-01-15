
import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import Database
from agents.recursive_agent import RecursiveResearcher

async def test_deep_research():
    print("🔬 Initializing Deep Research Test...")
    db = Database()
    researcher = RecursiveResearcher(db)
    
    query = "Effect of blue light on circadian rhythm"
    print(f"🔎 Searching for: '{query}' (Depth: 1, Max Papers: 2 per layer)")
    
    try:
        graph = await researcher.start_research(
            query=query,
            max_depth=1,
            max_papers=2
        )
        
        print("\n✅ Research Complete!")
        print(f"📊 Graph Nodes: {len(graph.nodes)}")
        print(f"🔗 Graph Edges: {len(graph.edges)}")
        
        verified_count = 0
        suspicious_count = 0
        
        print("\n--- Nodes Preview ---")
        for node in graph.nodes:
            status = node.data.get("validation_status", "unknown")
            if status == "verified":
                verified_count += 1
            elif status == "suspicious":
                suspicious_count += 1
                
            print(f"[{node.type.upper()}] {node.label[:60]}... (Status: {status})")
            
        print(f"\n📈 Stats: Verified={verified_count}, Suspicious={suspicious_count}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_deep_research())
