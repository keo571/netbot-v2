#!/usr/bin/env python3
"""
Command-line interface for diagram processing.

Usage:
    python -m diagram_processing process image.png diagram_001
    python -m diagram_processing batch data/examples/ data/processed/
"""

import argparse
import os
import sys

from .client import DiagramProcessor


def process_command(args):
    """Process a single diagram"""
    try:
        processor = DiagramProcessor(
            gemini_api_key=getattr(args, 'gemini_key', None)
        )
        
        result = processor.process_diagram(
            image_path=args.image_path,
            diagram_id=args.diagram_id,
            output_dir=args.output_dir,
            store_neo4j=not args.no_neo4j
        )
        
        if result['status'] == 'success':
            summary = result.get('graph_summary', {})
            nodes_count = summary.get('nodes_generated', 0)
            rels_count = summary.get('relationships_generated', 0)
            
            print(f"✅ Successfully processed {args.image_path}")
            print(f"📊 Generated {nodes_count} nodes and {rels_count} relationships")
            
            folder_name = os.path.splitext(os.path.basename(args.image_path))[0]
            actual_output_dir = os.path.join(args.output_dir, folder_name)
            print(f"📁 Output: {actual_output_dir}/")
            
            if result.get('neo4j_stored'):
                print("💾 Stored in Neo4j database")
        else:
            print(f"❌ Failed to process {args.image_path}")
            print(f"Error: {result.get('message', 'Unknown error')}")
            return 1
        
        processor.close()
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def batch_command(args):
    """Process multiple diagrams in a directory"""
    try:
        processor = DiagramProcessor(
            gemini_api_key=getattr(args, 'gemini_key', None)
        )
        
        results = processor.batch_process(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            store_neo4j=not args.no_neo4j
        )
        
        if results.get('status') == 'error':
            print(f"❌ Batch processing failed: {results.get('message')}")
            return 1
        
        total = results['total_images']
        successful = results['successful']
        failed = results['failed']
        skipped = results.get('skipped', 0)
        
        print(f"✅ Batch processing completed!")
        print(f"📊 Processed: {successful}/{total} images successfully")
        
        if skipped > 0:
            print(f"⏭️ Skipped: {skipped} duplicates")
        if failed > 0:
            print(f"❌ Failed: {failed} images")
            
        if successful > 0:
            print(f"📈 Total: {results['total_nodes']} nodes, {results['total_relationships']} relationships")
        
        processor.close()
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog='diagram_processing',
        description='Convert network diagrams and flowcharts into knowledge graphs'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process single diagram')
    process_parser.add_argument('image_path', help='Path to diagram image')
    process_parser.add_argument('diagram_id', help='Unique diagram identifier')
    process_parser.add_argument('--output-dir', default='data/processed/diagrams', help='Output directory')
    process_parser.add_argument('--no-neo4j', action='store_true', help='Skip Neo4j storage')
    process_parser.add_argument('--gemini-key', help='Gemini API key')
    process_parser.set_defaults(func=process_command)
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Process multiple diagrams')
    batch_parser.add_argument('input_dir', help='Input directory containing images')
    batch_parser.add_argument('output_dir', help='Output directory')
    batch_parser.add_argument('--no-neo4j', action='store_true', help='Skip Neo4j storage')
    batch_parser.add_argument('--gemini-key', help='Gemini API key')
    batch_parser.set_defaults(func=batch_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        return 1


if __name__ == "__main__":
    sys.exit(main())