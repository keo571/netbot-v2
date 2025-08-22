#!/usr/bin/env python3
"""
CLI for embedding operations on existing graphs.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from embeddings.client import EmbeddingManager
from dotenv import load_dotenv

load_dotenv()


def add_embeddings_command(args):
    """Add embeddings to an existing diagram"""
    # Get Neo4j connection details
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    if not neo4j_password:
        print("❌ NEO4J_PASSWORD environment variable not set")
        return 1
    
    # Initialize processor with credentials
    processor = EmbeddingManager(neo4j_uri, neo4j_user, neo4j_password)
    
    try:
        # Add embeddings to diagram
        success = processor.add_embeddings(args.diagram_id, args.batch_size)
        return 0 if success else 1
    finally:
        processor.close()


def remove_embeddings_command(args):
    """Remove embeddings from a diagram"""
    print("❌ Remove embeddings functionality has been removed.")
    print("💡 Embeddings are managed as all-or-nothing per diagram.")
    print("💡 To update embeddings, simply run 'add' command again.")
    return 1


def list_diagrams_command(args):
    """List diagrams with embeddings"""
    # Get Neo4j connection details
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    if not neo4j_password:
        print("❌ NEO4J_PASSWORD environment variable not set")
        return 1
    
    # Initialize processor with credentials
    processor = EmbeddingManager(neo4j_uri, neo4j_user, neo4j_password)
    
    try:
        # List diagrams
        diagrams = processor.list_diagrams_with_embeddings()
        
        if diagrams:
            print("📊 Diagrams with embeddings:")
            for diagram_id in diagrams:
                print(f"  • {diagram_id}")
        else:
            print("📭 No diagrams with embeddings found")
        
        return 0
    finally:
        processor.close()


def bulk_add_command(args):
    """Add embeddings to multiple diagrams"""
    # Get Neo4j connection details
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    if not neo4j_password:
        print("❌ NEO4J_PASSWORD environment variable not set")
        return 1
    
    # Initialize processor with credentials
    processor = EmbeddingManager(neo4j_uri, neo4j_user, neo4j_password)
    
    try:
        # Parse diagram IDs from command line
        diagram_ids = args.diagram_ids
        
        # Bulk add embeddings
        results = processor.bulk_add_embeddings(diagram_ids, args.batch_size)
        
        # Check if any failed
        failed_count = sum(1 for success in results.values() if not success)
        return 0 if failed_count == 0 else 1
        
    finally:
        processor.close()


def bulk_check_command(args):
    """Check embedding status for multiple diagrams"""
    # Get Neo4j connection details
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    if not neo4j_password:
        print("❌ NEO4J_PASSWORD environment variable not set")
        return 1
    
    # Initialize processor with credentials
    processor = EmbeddingManager(neo4j_uri, neo4j_user, neo4j_password)
    
    try:
        # Parse diagram IDs from command line
        diagram_ids = args.diagram_ids
        
        # Check embedding status
        results = processor.bulk_check_embeddings(diagram_ids)
        
        return 0
        
    finally:
        processor.close()


def list_missing_command(args):
    """List diagrams that don't have embeddings"""
    # Get Neo4j connection details
    neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
    neo4j_password = os.getenv('NEO4J_PASSWORD')
    
    if not neo4j_password:
        print("❌ NEO4J_PASSWORD environment variable not set")
        return 1
    
    # Initialize processor with credentials
    processor = EmbeddingManager(neo4j_uri, neo4j_user, neo4j_password)
    
    try:
        # Get diagrams without embeddings
        missing_diagrams = processor.get_diagrams_without_embeddings()
        
        if missing_diagrams:
            print(f"📭 Diagrams without embeddings ({len(missing_diagrams)}):")
            for diagram_id in missing_diagrams:
                print(f"  • {diagram_id}")
        else:
            print("✅ All diagrams have embeddings")
        
        return 0
        
    finally:
        processor.close()


def main():
    parser = argparse.ArgumentParser(description="Manage embeddings for existing graphs")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add embeddings command
    add_parser = subparsers.add_parser('add', help='Add embeddings to an existing diagram')
    add_parser.add_argument('diagram_id', help='Diagram ID to add embeddings to')
    add_parser.add_argument('--batch-size', type=int, default=100, 
                           help='Batch size for processing (default: 100)')
    add_parser.set_defaults(func=add_embeddings_command)
    
    # Remove embeddings command
    remove_parser = subparsers.add_parser('remove', help='Remove embeddings from a diagram')
    remove_parser.add_argument('diagram_id', help='Diagram ID to remove embeddings from')
    remove_parser.set_defaults(func=remove_embeddings_command)
    
    # List diagrams command
    list_parser = subparsers.add_parser('list', help='List diagrams with embeddings')
    list_parser.set_defaults(func=list_diagrams_command)
    
    # Bulk add embeddings command
    bulk_add_parser = subparsers.add_parser('bulk-add', help='Add embeddings to multiple diagrams')
    bulk_add_parser.add_argument('diagram_ids', nargs='+', help='Diagram IDs to add embeddings to')
    bulk_add_parser.add_argument('--batch-size', type=int, default=100, 
                                help='Batch size for processing (default: 100)')
    bulk_add_parser.set_defaults(func=bulk_add_command)
    
    # Bulk check command
    bulk_check_parser = subparsers.add_parser('bulk-check', help='Check embedding status for multiple diagrams')
    bulk_check_parser.add_argument('diagram_ids', nargs='+', help='Diagram IDs to check')
    bulk_check_parser.set_defaults(func=bulk_check_command)
    
    # List missing command
    missing_parser = subparsers.add_parser('list-missing', help='List diagrams without embeddings')
    missing_parser.set_defaults(func=list_missing_command)
    
    # Parse and execute
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())