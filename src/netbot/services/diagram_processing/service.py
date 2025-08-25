"""
Diagram Processing Service implementation.

Provides a clean, modern service layer that leverages the shared infrastructure
and integrates with the existing diagram processing pipeline.
"""

import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from ...shared import (
    get_logger, get_metrics, get_model_client, get_database,
    GraphNode, GraphRelationship, ProcessingError, AIError, DatabaseError
)
from .models import ProcessingRequest, ProcessingResult
from .repository import DiagramRepository


class DiagramProcessingService:
    """
    Service for processing diagrams into knowledge graphs.
    
    Provides a high-level interface for diagram processing operations
    while leveraging shared infrastructure for consistency and performance.
    """
    
    def __init__(self):
        """Initialize the service."""
        self.logger = get_logger(__name__)
        self.metrics = get_metrics()
        self.model_client = get_model_client()
        self.repository = DiagramRepository()
        
        self.logger.info("Diagram Processing Service initialized")
    
    def process_diagram(self, request: ProcessingRequest) -> ProcessingResult:
        """
        Process a diagram into structured graph data.
        
        Args:
            request: Processing request with parameters
            
        Returns:
            Processing result with extracted data
        """
        start_time = time.time()
        result = ProcessingResult(request=request)
        
        try:
            self.logger.info(f"Starting diagram processing: {request.diagram_id}")
            
            # Phase 1: OCR and preprocessing
            ocr_start = time.time()
            ocr_data = self._extract_text_and_shapes(request) if request.ocr_enabled else {}
            result.ocr_duration_seconds = time.time() - ocr_start
            
            # Phase 2: AI-powered relationship generation
            ai_start = time.time()
            graph_data = self._generate_graph_structure(request, ocr_data)
            result.ai_duration_seconds = time.time() - ai_start
            
            # Phase 3: Parse and validate results
            nodes, relationships = self._parse_graph_data(graph_data, request.diagram_id)
            result.nodes = nodes
            result.relationships = relationships
            
            # Phase 4: Store in database if requested
            if request.store_in_database and (nodes or relationships):
                storage_start = time.time()
                self._store_graph_data(request.diagram_id, nodes, relationships)
                result.storage_duration_seconds = time.time() - storage_start
            
            # Calculate metrics
            result.success = True
            result.nodes_detected = len(nodes)
            result.relationships_detected = len(relationships)
            result.confidence_score = self._calculate_confidence(nodes, relationships)
            
            # Save output files if directory specified
            if request.output_dir:
                result.output_files = self._save_output_files(
                    request.output_dir, request.diagram_id, nodes, relationships
                )
            
            self.logger.info(
                f"Successfully processed diagram {request.diagram_id}: "
                f"{len(nodes)} nodes, {len(relationships)} relationships"
            )
            
            # Record metrics
            self.metrics.record_api_request(
                endpoint="diagram_processing",
                method="POST",
                duration_seconds=time.time() - start_time,
                status_code=200
            )
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            
            self.logger.error(f"Diagram processing failed for {request.diagram_id}: {e}")
            
            # Record error metrics
            self.metrics.record_api_request(
                endpoint="diagram_processing",
                method="POST", 
                duration_seconds=time.time() - start_time,
                status_code=500
            )
            
            if not isinstance(e, (ProcessingError, AIError, DatabaseError)):
                # Wrap unexpected errors
                raise ProcessingError(f"Diagram processing failed: {e}")
        
        finally:
            result.total_duration_seconds = time.time() - start_time
        
        return result
    
    def _extract_text_and_shapes(self, request: ProcessingRequest) -> Dict[str, Any]:
        """
        Extract text and shapes from the image using OCR and computer vision.
        
        This is a simplified version - in the full implementation,
        this would integrate with the existing OCR and shape detection pipeline.
        """
        try:
            # TODO: Integrate with existing OCR pipeline
            # For now, return placeholder data
            return {
                'text_elements': [],
                'shapes': [],
                'image_metadata': {
                    'width': 800,
                    'height': 600,
                    'format': 'PNG'
                }
            }
        except Exception as e:
            raise ProcessingError(f"OCR extraction failed: {e}")
    
    def _generate_graph_structure(self, 
                                 request: ProcessingRequest, 
                                 ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use AI to generate graph structure from image and OCR data.
        
        This leverages the centralized model client for consistent AI operations.
        """
        try:
            # Read the image file
            image_path = Path(request.image_path)
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Create prompt for graph extraction
            prompt = self._create_extraction_prompt(ocr_data)
            
            # Use the centralized model client
            response = self.model_client.analyze_image(
                image_data=image_data,
                prompt=prompt,
                model_name=request.model_name
            )
            
            # Parse JSON response
            import json
            try:
                graph_data = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\\{.*\\}', response, re.DOTALL)
                if json_match:
                    graph_data = json.loads(json_match.group())
                else:
                    raise AIError("Could not parse JSON from AI response")
            
            return graph_data
            
        except Exception as e:
            raise AIError(f"Graph structure generation failed: {e}")
    
    def _create_extraction_prompt(self, ocr_data: Dict[str, Any]) -> str:
        """Create the prompt for AI-powered graph extraction."""
        base_prompt = """
        Analyze this diagram and extract the nodes and relationships to create a knowledge graph.
        
        Return a JSON object with this exact structure:
        {
            "nodes": [
                {
                    "id": "unique_identifier",
                    "label": "human_readable_name", 
                    "type": "node_type",
                    "properties": {"key": "value"}
                }
            ],
            "relationships": [
                {
                    "id": "unique_identifier",
                    "source_id": "source_node_id",
                    "target_id": "target_node_id", 
                    "type": "relationship_type",
                    "properties": {"key": "value"}
                }
            ]
        }
        
        Focus on:
        1. Identifying distinct entities (servers, processes, decisions, etc.)
        2. Understanding the flow and connections between entities
        3. Extracting meaningful properties and metadata
        4. Using consistent naming conventions
        
        Return only valid JSON.
        """
        
        # Add OCR context if available
        if ocr_data.get('text_elements'):
            base_prompt += f"\\n\\nDetected text elements: {ocr_data['text_elements']}"
        
        return base_prompt
    
    def _parse_graph_data(self, 
                         graph_data: Dict[str, Any], 
                         diagram_id: str) -> tuple[List[GraphNode], List[GraphRelationship]]:
        """Parse AI-generated graph data into domain models."""
        nodes = []
        relationships = []
        
        try:
            # Parse nodes
            for node_data in graph_data.get('nodes', []):
                node = GraphNode(
                    id=node_data['id'],
                    label=node_data['label'],
                    type=node_data['type'],
                    diagram_id=diagram_id,
                    properties=node_data.get('properties', {}),
                    confidence_score=0.8  # TODO: Calculate actual confidence
                )
                nodes.append(node)
            
            # Parse relationships
            for rel_data in graph_data.get('relationships', []):
                relationship = GraphRelationship(
                    id=rel_data['id'],
                    source_id=rel_data['source_id'],
                    target_id=rel_data['target_id'],
                    type=rel_data['type'],
                    diagram_id=diagram_id,
                    properties=rel_data.get('properties', {}),
                    confidence_score=0.8  # TODO: Calculate actual confidence
                )
                relationships.append(relationship)
            
            return nodes, relationships
            
        except Exception as e:
            raise ProcessingError(f"Graph data parsing failed: {e}")
    
    def _store_graph_data(self,
                         diagram_id: str,
                         nodes: List[GraphNode],
                         relationships: List[GraphRelationship]) -> None:
        """Store graph data in the database."""
        try:
            # Clear existing data for this diagram
            self.repository.clear_diagram(diagram_id)
            
            # Store nodes
            for node in nodes:
                self.repository.create_node(node)
            
            # Store relationships
            for relationship in relationships:
                self.repository.create_relationship(relationship)
                
            self.logger.info(f"Stored {len(nodes)} nodes and {len(relationships)} relationships for {diagram_id}")
            
        except Exception as e:
            raise DatabaseError(f"Graph data storage failed: {e}")
    
    def _calculate_confidence(self,
                            nodes: List[GraphNode],
                            relationships: List[GraphRelationship]) -> float:
        """Calculate overall confidence score for the processing result."""
        if not nodes and not relationships:
            return 0.0
        
        all_confidences = []
        all_confidences.extend([node.confidence_score for node in nodes])
        all_confidences.extend([rel.confidence_score for rel in relationships])
        
        return sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    
    def _save_output_files(self,
                          output_dir: str,
                          diagram_id: str,
                          nodes: List[GraphNode],
                          relationships: List[GraphRelationship]) -> Dict[str, str]:
        """Save processing results to output files."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            diagram_dir = output_path / diagram_id
            diagram_dir.mkdir(exist_ok=True)
            
            output_files = {}
            
            # Save nodes CSV
            if nodes:
                nodes_file = diagram_dir / "nodes.csv"
                self._save_nodes_csv(nodes, nodes_file)
                output_files['nodes_csv'] = str(nodes_file)
            
            # Save relationships CSV
            if relationships:
                relationships_file = diagram_dir / "relationships.csv"
                self._save_relationships_csv(relationships, relationships_file)
                output_files['relationships_csv'] = str(relationships_file)
            
            # Save metadata JSON
            metadata_file = diagram_dir / "metadata.json"
            self._save_metadata_json(diagram_id, nodes, relationships, metadata_file)
            output_files['metadata_json'] = str(metadata_file)
            
            return output_files
            
        except Exception as e:
            self.logger.warning(f"Failed to save output files: {e}")
            return {}
    
    def _save_nodes_csv(self, nodes: List[GraphNode], file_path: Path) -> None:
        """Save nodes to CSV file."""
        import csv
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'label', 'type', 'diagram_id', 'confidence_score', 'properties'])
            
            for node in nodes:
                writer.writerow([
                    node.id,
                    node.label,
                    node.type,
                    node.diagram_id,
                    node.confidence_score,
                    str(node.properties)
                ])
    
    def _save_relationships_csv(self, relationships: List[GraphRelationship], file_path: Path) -> None:
        """Save relationships to CSV file."""
        import csv
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'source_id', 'target_id', 'type', 'diagram_id', 'confidence_score', 'properties'])
            
            for rel in relationships:
                writer.writerow([
                    rel.id,
                    rel.source_id,
                    rel.target_id,
                    rel.type,
                    rel.diagram_id,
                    rel.confidence_score,
                    str(rel.properties)
                ])
    
    def _save_metadata_json(self,
                           diagram_id: str,
                           nodes: List[GraphNode],
                           relationships: List[GraphRelationship],
                           file_path: Path) -> None:
        """Save processing metadata to JSON file."""
        import json
        
        metadata = {
            'diagram_id': diagram_id,
            'node_count': len(nodes),
            'relationship_count': len(relationships),
            'node_types': list(set(node.type for node in nodes)),
            'relationship_types': list(set(rel.type for rel in relationships)),
            'processing_timestamp': time.time()
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def get_processing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent processing history."""
        try:
            return self.repository.get_recent_diagrams(limit)
        except Exception as e:
            self.logger.error(f"Failed to get processing history: {e}")
            return []
    
    def get_diagram_stats(self, diagram_id: str) -> Dict[str, Any]:
        """Get statistics for a processed diagram."""
        try:
            return self.repository.get_diagram_stats(diagram_id)
        except Exception as e:
            self.logger.error(f"Failed to get diagram stats for {diagram_id}: {e}")
            return {}