#!/usr/bin/env python3
"""
Main orchestration CLI for netbot-v2 system.

This CLI coordinates multiple modules:
- diagram_processing: Convert images to knowledge graphs
- embeddings: Add semantic search capabilities
- graph_rag: Query and visualize knowledge graphs

Usage:
    python cli.py quickstart image.png diagram_001 "find load balancers"
    python cli.py process-and-search image.png diagram_001 "show servers"
"""

import argparse
import sys
from pathlib import Path

from client import NetBot


def quickstart_command(args):
    """Run complete workflow: process + embeddings + search + visualization"""
    print(f"🚀 Running complete workflow for {args.image_path}")
    
    netbot = NetBot()
    results = netbot.quickstart(
        image_path=args.image_path,
        diagram_id=args.diagram_id,
        query=args.query
    )
    
    if results.get('error'):
        print(f"❌ Workflow failed: {results['error']}")
        return 1
    
    print("✅ Workflow completed successfully!")
    if results.get('nodes'):
        print(f"📊 Found {len(results['nodes'])} relevant nodes")
    
    return 0


def process_and_search_command(args):
    """Process diagram then search with optional visualization"""
    netbot = NetBot()
    
    print(f"📋 Processing {args.image_path}...")
    result = netbot.process_diagram(
        image_path=args.image_path,
        diagram_id=args.diagram_id,
        output_dir=getattr(args, 'output_dir', 'data/processed')
    )
    
    if result.get('status') != 'success':
        print(f"❌ Failed to process {args.image_path}")
        print(f"Error: {result.get('error', 'Unknown error')}")
        return 1
    
    print(f"✅ Processed: {len(result.get('nodes', []))} nodes, {len(result.get('relationships', []))} relationships")
    
    # Add embeddings if requested
    if not getattr(args, 'no_embeddings', False):
        print("🧠 Adding embeddings...")
        success = netbot.add_embeddings(args.diagram_id)
        if success:
            print("✅ Embeddings added successfully")
        else:
            print("⚠️ Failed to add embeddings, continuing with basic search...")
    
    # Search
    print(f"🔍 Searching: {args.query}")
    search_results = netbot.search(
        query=args.query,
        diagram_id=args.diagram_id,
        method=getattr(args, 'method', 'auto'),
        top_k=getattr(args, 'top_k', 8)
    )
    
    if not search_results.get('nodes'):
        print("❌ No results found")
        return 1
    
    # Visualize if requested
    if getattr(args, 'visualize', False):
        print("📊 Creating visualization...")
        viz_result = netbot.query_and_visualize(
            query=args.query,
            diagram_id=args.diagram_id,
            backend=getattr(args, 'backend', 'graphviz'),
            include_explanation=True
        )
        
        if viz_result.get('image_path'):
            print(f"📈 Visualization saved: {viz_result['image_path']}")
        
        if viz_result.get('explanation'):
            print(f"\n💡 Explanation:\n{viz_result['explanation']}")
    else:
        print(f"🔍 Found {len(search_results['nodes'])} nodes:")
        for node in search_results['nodes'][:5]:
            print(f"  - {node.get('label', 'Unknown')} ({node.get('type', 'Unknown')})")
        
        if len(search_results['nodes']) > 5:
            print(f"  ... and {len(search_results['nodes']) - 5} more")
    
    return 0


def bulk_quickstart_command(args):
    """Run bulk workflow: process directory + embeddings (no search aggregation)"""
    print(f"🚀 Running bulk workflow for directory {args.image_directory}")
    
    netbot = NetBot()
    results = netbot.bulk_quickstart(
        image_directory=args.image_directory,
        query=args.query  # query parameter is used for guidance in help text
    )
    
    if results.get('error'):
        print(f"❌ Bulk workflow failed: {results['error']}")
        return 1
    
    # Results already include the summary printout from the method
    return 0


def bulk_embeddings_command(args):
    """Add embeddings to multiple diagrams"""
    netbot = NetBot()
    
    # Get diagrams to process
    if args.all_missing:
        print("🔍 Finding diagrams without embeddings...")
        diagram_ids = netbot.get_diagrams_without_embeddings()
        if not diagram_ids:
            print("✅ All diagrams already have embeddings")
            return 0
        print(f"📋 Found {len(diagram_ids)} diagrams without embeddings")
    else:
        diagram_ids = args.diagram_ids
    
    print(f"🧠 Adding embeddings to {len(diagram_ids)} diagrams...")
    results = netbot.bulk_add_embeddings(diagram_ids, args.batch_size)
    
    # Summary
    successful = sum(1 for success in results.values() if success)
    failed = len(diagram_ids) - successful
    
    print(f"✅ Bulk embedding completed!")
    print(f"📊 Successful: {successful}/{len(diagram_ids)}")
    if failed > 0:
        print(f"❌ Failed: {failed}")
        failed_diagrams = [did for did, success in results.items() if not success]
        for diagram_id in failed_diagrams:
            print(f"  • {diagram_id}")
    
    return 0 if failed == 0 else 1


def main():
    """Main orchestration CLI entry point"""
    parser = argparse.ArgumentParser(
        prog='netbot-v2',
        description='NetBot-v2: AI-powered diagram analysis and knowledge graph system'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available workflows')
    
    # Quickstart command - full automated workflow
    quickstart_parser = subparsers.add_parser('quickstart', 
                                            help='Complete automated workflow (recommended)')
    quickstart_parser.add_argument('image_path', help='Path to diagram image')
    quickstart_parser.add_argument('diagram_id', help='Unique diagram identifier')
    quickstart_parser.add_argument('query', help='Search query')
    quickstart_parser.set_defaults(func=quickstart_command)
    
    # Process-and-search command - more control
    process_search_parser = subparsers.add_parser('process-and-search',
                                                help='Process diagram then search with options')
    process_search_parser.add_argument('image_path', help='Path to diagram image')
    process_search_parser.add_argument('diagram_id', help='Unique diagram identifier')
    process_search_parser.add_argument('query', help='Search query')
    process_search_parser.add_argument('--output-dir', default='data/processed', help='Processing output directory')
    process_search_parser.add_argument('--no-embeddings', action='store_true', help='Skip embeddings generation')
    process_search_parser.add_argument('--method', choices=['vector', 'cypher', 'auto'], default='auto', help='Search method')
    process_search_parser.add_argument('--top-k', type=int, default=8, help='Number of results')
    process_search_parser.add_argument('--visualize', action='store_true', help='Create visualization')
    process_search_parser.add_argument('--backend', choices=['networkx', 'graphviz'], default='graphviz', help='Visualization backend')
    process_search_parser.set_defaults(func=process_and_search_command)
    
    # Bulk quickstart command - process entire directory
    bulk_quickstart_parser = subparsers.add_parser('bulk-quickstart',
                                                  help='Bulk workflow: process directory + embeddings + search')
    bulk_quickstart_parser.add_argument('image_directory', help='Directory containing diagram images')
    bulk_quickstart_parser.add_argument('query', help='Search query to run across all diagrams')
    bulk_quickstart_parser.add_argument('--explanation-detail', choices=['none', 'basic', 'detailed'], 
                                       default='basic', help='Explanation detail level')
    bulk_quickstart_parser.set_defaults(func=bulk_quickstart_command)
    
    # Bulk embeddings command - add embeddings to multiple diagrams
    bulk_embeddings_parser = subparsers.add_parser('bulk-embeddings',
                                                  help='Add embeddings to multiple diagrams')
    bulk_embeddings_parser.add_argument('--diagram-ids', nargs='*', default=[], 
                                       help='Specific diagram IDs to process')
    bulk_embeddings_parser.add_argument('--all-missing', action='store_true',
                                       help='Process all diagrams without embeddings')
    bulk_embeddings_parser.add_argument('--batch-size', type=int, default=100,
                                       help='Batch size for processing (default: 100)')
    bulk_embeddings_parser.set_defaults(func=bulk_embeddings_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        print("\n💡 Quick start: python cli.py quickstart diagram.png my_diagram 'find servers'")
        print("💡 Advanced: python cli.py process-and-search diagram.png my_diagram 'find servers' --visualize")
        print("💡 Bulk processing: python cli.py bulk-quickstart diagrams/ 'find load balancers'")
        print("💡 Bulk embeddings: python cli.py bulk-embeddings --all-missing")
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())