"""
Simple client interface for diagram processing pipeline.

Usage:
    from diagram_processing import DiagramProcessor
    
    processor = DiagramProcessor(gemini_api_key="your-key")
    result = processor.process("image.png", "diagram_001")
"""

import os
from typing import Dict, List
from dotenv import load_dotenv

import json
from pathlib import Path
from .core.pipeline import KnowledgeGraphPipeline

# Load environment variables
load_dotenv()


class DiagramProcessor:
    """
    Simple interface for diagram processing pipeline.
    
    Converts network diagrams and flowcharts into knowledge graphs.
    """
    
    def __init__(self, 
                 gemini_api_key: str = None,
                 neo4j_uri: str = None,
                 neo4j_user: str = None, 
                 neo4j_password: str = None):
        """
        Initialize diagram processor.
        
        Args:
            gemini_api_key: Gemini API key (defaults to env GEMINI_API_KEY)
            neo4j_uri: Neo4j database URI (defaults to env NEO4J_URI)
            neo4j_user: Neo4j username (defaults to env NEO4J_USER)
            neo4j_password: Neo4j password (defaults to env NEO4J_PASSWORD)
        """
        # Use provided credentials or fall back to environment variables
        self.gemini_api_key = gemini_api_key or os.getenv('GEMINI_API_KEY')
        self.neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI')
        self.neo4j_user = neo4j_user or os.getenv('NEO4J_USER')
        self.neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD')
        
        # Validate required credentials
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY must be provided via parameter or environment variable")
        
        # Initialize pipeline (lazy loading)
        self._pipeline = None
    
    def process_diagram(self, image_path: str, diagram_id: str, 
                       output_dir: str = "data/processed/diagrams", store_neo4j: bool = True, 
                       force_reprocess: bool = False) -> Dict:
        """
        Process a single diagram image into a knowledge graph.
        
        Args:
            image_path: Path to the diagram image
            diagram_id: Unique identifier for this diagram
            output_dir: Directory to save results (optional)
            store_neo4j: Whether to store in Neo4j database (optional)
            force_reprocess: If True, reprocess even if diagram exists in Neo4j
            
        Returns:
            Dict with processing results including nodes, relationships, and status
        """
        if not self._pipeline:
            self._pipeline = KnowledgeGraphPipeline(
                gemini_api_key=self.gemini_api_key,
                neo4j_uri=self.neo4j_uri,
                neo4j_user=self.neo4j_user,
                neo4j_password=self.neo4j_password
            )
        
        return self._pipeline.process_image(
            image_path=image_path,
            diagram_id=diagram_id,
            output_dir=output_dir,
            store_neo4j=store_neo4j,
            force_reprocess=force_reprocess
        )
    
    def batch_process(self, input_dir: str, output_dir: str = "data/processed/batch", 
                     store_neo4j: bool = True, file_extensions: List[str] = None) -> Dict:
        """
        Process multiple diagram images in a directory.
        
        Args:
            input_dir: Directory containing diagram images
            output_dir: Directory to save batch results
            store_neo4j: Whether to store in Neo4j database
            file_extensions: List of file extensions to process (default: ['.png', '.jpg', '.jpeg'])
            
        Returns:
            Dict with batch processing results and summary
        """
        if file_extensions is None:
            file_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        
        print(f"Starting batch processing...")
        print(f"Input: {input_dir}")
        print(f"Output: {output_dir}")
        
        # Find image files
        image_files = self._find_image_files(input_dir, file_extensions)
        if not image_files:
            return {'status': 'error', 'message': 'No images found'}
        
        print(f"Found {len(image_files)} images to process")
        
        # Initialize results tracking
        batch_results = {
            'total_images': len(image_files),
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total_nodes': 0,
            'total_relationships': 0,
            'results': []
        }
        
        processed_diagrams = set()  # Track processed diagram IDs
        
        # Process each image
        for i, image_file in enumerate(image_files, 1):
            print(f"\nProcessing {i}/{len(image_files)}: {image_file.name}")
            
            diagram_id = image_file.stem
            
            # Check for duplicates
            if diagram_id in processed_diagrams:
                print(f"Skipping {diagram_id} - already processed in this batch")
                batch_results['skipped'] += 1
                continue
            
            try:
                result = self.process_diagram(
                    str(image_file), 
                    diagram_id,
                    output_dir,
                    store_neo4j
                )
                
                if result['status'] == 'success':
                    processed_diagrams.add(diagram_id)
                    
                    summary = result.get('graph_summary', {})
                    nodes_count = summary.get('nodes_generated', 0)
                    rels_count = summary.get('relationships_generated', 0)
                    
                    batch_results['successful'] += 1
                    batch_results['total_nodes'] += nodes_count
                    batch_results['total_relationships'] += rels_count
                    
                    result['source_image'] = str(image_file)
                    result['image_name'] = image_file.name
                    batch_results['results'].append(result)
                    
                    print(f"Success: {nodes_count} nodes, {rels_count} relationships")
                else:
                    batch_results['failed'] += 1
                    print(f"Failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                batch_results['failed'] += 1
                print(f"Failed: {str(e)}")
        
        # Save batch results summary
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        batch_results_file = output_path / 'batch_results.json'
        with open(batch_results_file, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        print(f"\n✅ Batch processing complete!")
        print(f"📊 Results: {batch_results['successful']} successful, {batch_results['failed']} failed, {batch_results['skipped']} skipped")
        print(f"📈 Generated: {batch_results['total_nodes']} nodes, {batch_results['total_relationships']} relationships")
        print(f"💾 Summary saved to: {batch_results_file}")
        
        return batch_results
    
    def _find_image_files(self, input_dir: str, file_extensions: List[str]) -> List[Path]:
        """Find all image files in the input directory."""
        input_path = Path(input_dir)
        image_files = []
        
        for ext in file_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
            
        return sorted(image_files)  # Sort for consistent processing order
    
    def close(self):
        """Clean up resources."""
        if self._pipeline:
            self._pipeline.close()


# Convenience function
def process_diagram(image_path: str, diagram_id: str, **kwargs) -> Dict:
    """
    Quick function to process a single diagram.
    
    Args:
        image_path: Path to diagram image
        diagram_id: Unique diagram identifier
        **kwargs: Additional arguments for DiagramProcessor
        
    Returns:
        Processing results
    """
    processor = DiagramProcessor(**kwargs)
    try:
        return processor.process_diagram(image_path, diagram_id)
    finally:
        processor.close()


if __name__ == "__main__":
    print("🔧 Diagram Processing Client")
    print("For CLI usage, use: python -m diagram_processing")