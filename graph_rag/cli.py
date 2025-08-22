#!/usr/bin/env python3
"""
Specialized CLI for GraphRAG operations.

This CLI focuses on GraphRAG-specific tasks:
- Search knowledge graphs
- Generate visualizations  
- Create explanations
- Manage search cache

Usage:
    python -m graph_rag search "find load balancers" diagram_001
    python -m graph_rag visualize "show servers" diagram_001 --backend graphviz
    python -m graph_rag explain "network topology" diagram_001 --detailed
    python -m graph_rag cache invalidate diagram_001
"""

import argparse
import sys
import json
from pathlib import Path

from .client import GraphRAG


def search_command(args):
    """Search the knowledge graph with natural language"""
    rag = GraphRAG()
    
    results = rag.search(
        query=args.query,
        diagram_id=args.diagram_id,
        method=getattr(args, 'method', 'auto'),
        top_k=getattr(args, 'top_k', 8)
    )
    
    if results.get('nodes'):
        print(f"🔍 Found {len(results['nodes'])} nodes:")
        for node in results['nodes'][:10]:  # Show first 10
            props = ", ".join([f"{k}: {v}" for k, v in node.get('properties', {}).items() 
                             if k not in {'id', 'diagram_id', 'embedding'}])
            label = f"  - {node.get('label', 'Unknown')} ({node.get('type', 'Unknown')})"
            if props and getattr(args, 'show_properties', False):
                label += f" - {props}"
            print(label)
        
        if len(results['nodes']) > 10:
            print(f"  ... and {len(results['nodes']) - 10} more")
        
        print(f"\\n🔗 Found {len(results.get('relationships', []))} relationships")
        
        # Save results if requested
        if getattr(args, 'output', None):
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {args.output}")
    else:
        print("❌ No results found")
        rag.close()
        return 1
    
    rag.close()
    return 0


def visualize_command(args):
    """Create visualization with search results"""
    rag = GraphRAG()
    
    results = rag.query_and_visualize(
        natural_query=args.query,
        diagram_id=args.diagram_id,
        backend=getattr(args, 'backend', 'graphviz'),
        layout=getattr(args, 'layout', None),
        output_path=getattr(args, 'output', None),
        include_explanation=getattr(args, 'explain', False),
        detailed_explanation=getattr(args, 'detailed', False),
        method=getattr(args, 'method', 'auto'),
        show_node_properties=getattr(args, 'show_node_properties', True),
        show_edge_properties=getattr(args, 'show_edge_properties', True),
        generate_property_summary=getattr(args, 'property_summary', False)
    )
    
    if results.get('error'):
        print(f"❌ Visualization failed: {results['error']}")
        return 1
    
    if results.get('image_path'):
        print(f"📈 Visualization saved: {results['image_path']}")
    else:
        print("📊 Visualization displayed interactively")
    
    if results.get('explanation'):
        print(f"\\n💡 Explanation:\\n{results['explanation']}")
    
    if results.get('property_summary'):
        print(f"\\n📋 Property Summary:\\n{results['property_summary']}")
    
    rag.close()
    return 0


def explain_command(args):
    """Generate explanation for search results"""
    rag = GraphRAG()
    
    # First search for results
    search_results = rag.search(
        query=args.query,
        diagram_id=args.diagram_id,
        method=getattr(args, 'method', 'auto'),
        top_k=getattr(args, 'top_k', 8)
    )
    
    if not search_results.get('nodes'):
        print("❌ No results found to explain")
        rag.close()
        return 1
    
    # Generate explanation
    explanation = rag.explain_subgraph(
        nodes=search_results['nodes'],
        relationships=search_results.get('relationships', []),
        original_query=args.query,
        detailed=getattr(args, 'detailed', False)
    )
    
    print(f"💡 Explanation for: '{args.query}'\\n")
    print(explanation)
    
    # Save explanation if requested
    if getattr(args, 'output', None):
        with open(args.output, 'w') as f:
            f.write(f"Query: {args.query}\\n\\n{explanation}")
        print(f"\\n💾 Explanation saved to: {args.output}")
    
    rag.close()
    return 0


def cache_command(args):
    """Manage GraphRAG search cache"""
    rag = GraphRAG()
    
    if args.action == 'invalidate':
        rag.invalidate_cache(args.diagram_id)
        print(f"✅ Search cache invalidated for {args.diagram_id}")
    
    rag.close()
    return 0


def main():
    """GraphRAG CLI entry point"""
    parser = argparse.ArgumentParser(
        prog='graph_rag',
        description='GraphRAG: Query and visualize knowledge graphs with AI'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='GraphRAG operations')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search knowledge graph with natural language')
    search_parser.add_argument('query', help='Natural language search query')
    search_parser.add_argument('diagram_id', help='Diagram identifier')
    search_parser.add_argument('--method', choices=['vector', 'cypher', 'auto'], default='auto', help='Search method')
    search_parser.add_argument('--top-k', type=int, default=8, help='Number of results')
    search_parser.add_argument('--show-properties', action='store_true', help='Show node properties')
    search_parser.add_argument('--output', help='Save results to JSON file')
    search_parser.set_defaults(func=search_command)
    
    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Search and create visualization')
    visualize_parser.add_argument('query', help='Natural language search query')
    visualize_parser.add_argument('diagram_id', help='Diagram identifier')
    visualize_parser.add_argument('--backend', choices=['networkx', 'graphviz'], default='graphviz', help='Visualization backend')
    visualize_parser.add_argument('--layout', help='Layout algorithm (backend-specific)')
    visualize_parser.add_argument('--output', help='Output file path')
    visualize_parser.add_argument('--method', choices=['vector', 'cypher', 'auto'], default='auto', help='Search method')
    visualize_parser.add_argument('--explain', action='store_true', help='Include explanation')
    visualize_parser.add_argument('--detailed', action='store_true', help='Detailed explanation')
    visualize_parser.add_argument('--no-node-properties', dest='show_node_properties', action='store_false', help='Hide node properties')
    visualize_parser.add_argument('--no-edge-properties', dest='show_edge_properties', action='store_false', help='Hide edge properties') 
    visualize_parser.add_argument('--property-summary', action='store_true', help='Generate property summary')
    visualize_parser.set_defaults(func=visualize_command)
    
    # Explain command
    explain_parser = subparsers.add_parser('explain', help='Generate explanation for search results')
    explain_parser.add_argument('query', help='Natural language search query')
    explain_parser.add_argument('diagram_id', help='Diagram identifier')
    explain_parser.add_argument('--method', choices=['vector', 'cypher', 'auto'], default='auto', help='Search method')
    explain_parser.add_argument('--top-k', type=int, default=8, help='Number of results')
    explain_parser.add_argument('--detailed', action='store_true', help='Generate detailed explanation')
    explain_parser.add_argument('--output', help='Save explanation to file')
    explain_parser.set_defaults(func=explain_command)
    
    # Cache command
    cache_parser = subparsers.add_parser('cache', help='Manage GraphRAG search cache')
    cache_parser.add_argument('action', choices=['invalidate'], help='Cache action')
    cache_parser.add_argument('diagram_id', help='Diagram identifier')
    cache_parser.set_defaults(func=cache_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        print("\\n💡 Examples:")
        print("  python -m graph_rag search 'find load balancers' diagram_001")
        print("  python -m graph_rag visualize 'show network topology' diagram_001 --explain")
        print("  python -m graph_rag explain 'security architecture' diagram_001 --detailed")
        print("  python -m graph_rag cache invalidate diagram_001")
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\\n⚠️ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())