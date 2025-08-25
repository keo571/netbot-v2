"""
Main pipeline orchestrator that coordinates all 3 phases.
"""

import os
import json
from typing import Dict, Optional
from dataclasses import asdict

from .preprocessor import ImagePreprocessor
from .generator import GeminiGraphGenerator
from .exporter import KnowledgeGraphExporter
from models.graph_models import GraphNode, GraphRelationship


class KnowledgeGraphPipeline:
    """Main pipeline orchestrator for all 3 phases"""
    
    def __init__(self, gemini_api_key: str, neo4j_uri: Optional[str] = None, 
                 neo4j_user: Optional[str] = None, neo4j_password: Optional[str] = None):
        """
        Initialize the complete pipeline
        
        Args:
            gemini_api_key: API key for Gemini 2.5 Pro
            neo4j_uri: Neo4j database URI (optional)
            neo4j_user: Neo4j username (optional)
            neo4j_password: Neo4j password (optional)
        """
        self.preprocessor = ImagePreprocessor()
        self.graph_generator = GeminiGraphGenerator(gemini_api_key)
        self.exporter = KnowledgeGraphExporter(neo4j_uri, neo4j_user, neo4j_password)
    
    def process_image(self, image_path: str, diagram_id: str, output_dir: str = "output",
                      store_neo4j: bool = True, force_reprocess: bool = False) -> Dict:
        """
        Run complete pipeline on an image
        
        Args:
            image_path: Path to the input image
            diagram_id: A unique identifier for this specific diagram
            output_dir: Directory for output files
            store_neo4j: Whether to store results in Neo4j
            force_reprocess: If True, reprocess even if diagram exists in Neo4j
            
        Returns:
            dict: Complete pipeline results
        """
        print(f"Starting diagram-to-graph processing for: {image_path}")
        
        # Early check: Skip processing if diagram already exists (unless force reprocess)
        if store_neo4j and not force_reprocess and self.exporter._ensure_neo4j_connection():
            if self.exporter._diagram_exists(diagram_id):
                print(f"✅ Diagram '{diagram_id}' already exists in Neo4j, skipping processing")
                print("💡 Use force_reprocess=True to reprocess existing diagrams")
                # Get existing data from Neo4j
                existing_nodes, existing_relationships = self.exporter._get_existing_diagram_data(diagram_id)
                self.exporter.close()
                return {
                    'status': 'success',
                    'diagram_id': diagram_id,
                    'message': f"Using existing diagram '{diagram_id}' from Neo4j",
                    'nodes': existing_nodes,
                    'relationships': existing_relationships,
                    'neo4j_stored': True,
                    'skipped_processing': True
                }
        
        # If force_reprocess=True and diagram exists, delete existing data first
        if store_neo4j and force_reprocess and self.exporter._ensure_neo4j_connection():
            if self.exporter._diagram_exists(diagram_id):
                print(f"🔄 Force reprocessing: deleting existing diagram '{diagram_id}' from Neo4j")
                self.exporter._delete_diagram(diagram_id)
        
        # Phase 1: Preprocessing
        preprocessed = self.preprocessor.process_image(image_path)
        if preprocessed['status'] != 'success':
            return preprocessed
        
        print(f"Detected {preprocessed['diagram_type']} diagram: {preprocessed['total_text_elements']} text elements, {preprocessed['total_shapes']} shapes")
        
        # Phase 2: Relationship generation
        nodes, relationships = self.graph_generator.generate_graph(image_path, preprocessed, diagram_id)
        if not relationships or not nodes:
            return {'status': 'error', 'message': 'Failed to generate graph data from Gemini'}
        
        # Phase 3: Export
        # Use diagram_id for folder name instead of temp image path
        nodes_file, rels_file = self.exporter.generate_csv_files(nodes, relationships, output_dir, diagram_id)
        
        # Neo4j storage
        neo4j_success = False
        if store_neo4j:
            neo4j_success = self.exporter.store_in_neo4j(diagram_id, nodes, relationships)
        
        # Prepare full results for return (includes all data for programmatic access)
        result = {
            'status': 'success',
            'diagram_id': diagram_id,
            'diagram_type': preprocessed['diagram_type'],
            'total_elements': preprocessed['total_text_elements'],
            'total_shapes': preprocessed['total_shapes'],
            'graph_summary': {
                'nodes_generated': len(nodes),
                'relationships_generated': len(relationships)
            },
            'nodes': [asdict(n) for n in nodes],
            'relationships': [asdict(r) for r in relationships],
            'csv_files': {
                'nodes': nodes_file,
                'relationships': rels_file
            },
            'neo4j_stored': neo4j_success
        }
        
        # Save lightweight metadata file (no duplicate data, only unique insights)
        # Use diagram_id for consistent folder naming
        results_output_dir = os.path.join(output_dir, diagram_id)
        results_file = os.path.join(results_output_dir, 'pipeline_metadata.json')
        os.makedirs(results_output_dir, exist_ok=True)
        
        metadata = {
            'status': 'success',
            'diagram_id': diagram_id,
            'diagram_type': preprocessed['diagram_type'],
            'preprocessing': {
                'total_text_elements': preprocessed['total_text_elements'],
                'total_shapes': preprocessed['total_shapes']
            },
            'output_files': {
                'nodes_csv': nodes_file,
                'relationships_csv': rels_file
            },
            'neo4j_stored': neo4j_success
        }
        
        with open(results_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return result
    
    def close(self):
        """Close any open connections"""
        self.exporter.close()
