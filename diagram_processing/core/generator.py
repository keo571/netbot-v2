"""
Phase 2: Gemini 2.5 Pro for reasoning and relationship generation.
"""

import mimetypes
from typing import Dict, List, Tuple
import google.generativeai as genai

from models.graph_models import GraphNode, GraphRelationship, Shape
from ..utils.json_utils import LLMJsonParser


class GeminiGraphGenerator:
    """Phase 2: Use Gemini 2.5 Pro for complete graph generation from diagrams"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        # Configure generation parameters for longer responses
        self.generation_config = genai.types.GenerationConfig(
            max_output_tokens=32768, # Gemini 2.5 Pro supports up to 65,535 tokens
            temperature=0.1,         # Lower temperature for more consistent JSON
        )
    
    def generate_graph(self, image_path: str, preprocessed_data: Dict, diagram_id: str) -> Tuple[List[GraphNode], List[GraphRelationship]]:
        """
        Generate complete graph structure (nodes and relationships) using Gemini 2.5 Pro.
        
        Args:
            image_path: Path to the original image
            preprocessed_data: Results from Phase 1 preprocessing
            
        Returns:
            A tuple containing (nodes, relationships) for the complete graph.
        """
        print("Phase 2: Generating complete graph with Gemini 2.5 Pro...")
        
        try:
            # Guess the MIME type from the file path
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith('image/'):
                print(f"⚠️ Could not determine MIME type for {image_path}. Defaulting to image/png.")
                mime_type = 'image/png'

            # Upload image to Gemini
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
            # Create appropriate prompt based on diagram type
            diagram_type = preprocessed_data.get('diagram_type', 'network')
            shapes = preprocessed_data.get('shapes', [])
            
            if 'flowchart' in diagram_type:
                prompt = self._create_flowchart_prompt(preprocessed_data, shapes)
            else:
                prompt = self._create_network_prompt(preprocessed_data, shapes)
            
            # Generate response
            response = self.model.generate_content([
                prompt,
                {
                    'mime_type': mime_type,
                    'data': image_data
                }
            ], generation_config=self.generation_config)
            
            # Parse and return nodes and relationships
            return self._parse_gemini_response(response.text, diagram_id)
            
        except Exception as e:
            print(f"❌ Error generating graph data with Gemini: {e}")
            return [], []
    
    def _create_network_prompt(self, ocr_data: Dict, shapes: List[Shape]) -> str:
        """Create specialized prompt for network diagrams"""
        prompt = f"""
You are an expert network engineer analyzing an architecture diagram. Your task is to extract all components and their connections into a structured knowledge graph.

**CONTEXT:**
- This is a {ocr_data.get('diagram_type', 'network')} diagram.
- Nodes are network devices or services.
- Lines are the connections between them.

**EXTRACTED TEXT ELEMENTS:**
"""
        
        # Add categorized text elements
        if ocr_data.get('diagram_type') == 'mixed':
            for category_type, categories in ocr_data['text_elements'].items():
                prompt += f"\n{category_type.upper()}:\n"
                for category, items in categories.items():
                    if items:
                        prompt += f"  {category}: {[item['text'] for item in items]}\n"
        else:
            for category, items in ocr_data['text_elements'].items():
                if items:
                    prompt += f"  {category}: {[item['text'] for item in items]}\n"
        
        prompt += f"""
**DETECTED SHAPES:**
- Total shapes detected: {len(shapes)}
- Rectangles/Squares: {len([s for s in shapes if s.type in ['rectangle', 'square']])}
- Circles: {len([s for s in shapes if s.type == 'circle'])}
- Lines: {len([s for s in shapes if s.type == 'line'])}

**TASK:**
Generate a structured JSON object with two keys: "nodes" and "relationships".

1.  **Nodes**: Identify every network component (e.g., router, switch, firewall, server, cloud service).
    -   Create a unique ID for each node (e.g., "node-1", "node-2").
    -   The `label` should be the descriptive name from the diagram.
    -   The `type` should be a classification (e.g., "Router", "Firewall", "Cloud").
    -   Use the `properties` field for extra information like IP addresses.

2.  **Relationships**: Identify the connections between the nodes.
    -   Create a unique ID for each relationship (e.g., "rel-1", "rel-2").
    -   Use the node IDs you created for `source_id` and `target_id`.
    -   The `type` should describe the connection type (e.g., "CONNECTS_TO", "PART_OF").
    -   Use the `properties` field for connection-specific details like interface names or descriptions found on the line itself.

**OUTPUT FORMAT:**
Provide ONLY a JSON object with this exact structure:
```json
{{
  "nodes": {{
    "node-1": {{
      "label": "RouterA",
      "type": "Router",
      "properties": {{
        "ip_address": "192.168.1.1"
      }}
    }},
    "node-2": {{
      "label": "Firewall",
      "type": "Firewall",
      "properties": {{}}
    }}
  }},
  "relationships": [
    {{
      "id": "rel-1",
      "source_id": "node-1",
      "target_id": "node-2",
      "type": "CONNECTS_TO",
      "properties": {{
        "interface": "GigabitEthernet0/1",
        "description": "Main link"
      }}
    }}
  ]
}}
```

**IMPORTANT:**
- For large diagrams (50+ nodes), focus on the most important connections to avoid response truncation.

Generate the structured JSON now:
"""
        return prompt
    
    def _create_flowchart_prompt(self, ocr_data: Dict, shapes: List[Shape]) -> str:
        """Create specialized prompt for flowcharts"""
        prompt = f"""
You are an expert process analyst interpreting a flowchart. Your task is to extract all steps, decisions, and process flows into a structured knowledge graph.

**CONTEXT:**
- This is a {ocr_data.get('diagram_type', 'flowchart')} diagram.
- Shapes represent process steps or decisions.
- Arrows represent the flow and conditions.

**EXTRACTED TEXT ELEMENTS:**
"""
        
        # Add categorized text elements
        if ocr_data.get('diagram_type') == 'mixed':
            for category_type, categories in ocr_data['text_elements'].items():
                if 'flowchart' in category_type:
                    prompt += f"\n{category_type.upper()}:\n"
                    for category, items in categories.items():
                        if items:
                            prompt += f"  {category}: {[item['text'] for item in items]}\n"
        else:
            for category, items in ocr_data['text_elements'].items():
                if items:
                    prompt += f"  {category}: {[item['text'] for item in items]}\n"
        
        prompt += f"""
**DETECTED SHAPES:**
- Total shapes detected: {len(shapes)}
- Process boxes: {len([s for s in shapes if s.type in ['rectangle', 'square']])}
- Decision diamonds: {len([s for s in shapes if s.type == 'circle'])}
- Flow lines: {len([s for s in shapes if s.type == 'line'])}

**TASK:**
Generate a structured JSON object with two keys: "nodes" and "relationships".

1.  **Nodes**: Identify every process step, decision point, start, and end point.
    -   Create a unique ID for each node (e.g., "node-1", "node-2").
    -   The `label` should be a concise summary of the step.
    -   The `type` must be one of: "process", "decision", "start", "end".
    -   If a step has a long paragraph of text, place the full text in the `description` property of the node.

2.  **Relationships**: Identify the arrows connecting the nodes.
    -   Create a unique ID for each relationship (e.g., "rel-1", "rel-2").
    -   Use the node IDs you created for `source_id` and `target_id`.
    -   The `type` should be "FLOWS_TO".
    -   If an arrow has a text label, use it to populate the relationship's properties. Use the `condition` property for decision branches (e.g., "Yes", "No") and the `description` property for other labels.

**OUTPUT FORMAT:**
Provide ONLY a JSON object with this exact structure:
```json
{{
  "nodes": {{
    "node-1": {{
      "label": "Start Process",
      "type": "start",
      "properties": {{
        "description": "The process begins here."
      }}
    }},
    "node-2": {{
      "label": "Is User Valid?",
      "type": "decision",
      "properties": {{
        "description": "Check user credentials against the database."
      }}
    }}
  }},
  "relationships": [
    {{
      "id": "rel-1",
      "source_id": "node-1",
      "target_id": "node-2",
      "type": "FLOWS_TO",
      "properties": {{
        "description": "Initial flow"
      }}
    }},
    {{
      "id": "rel-2",
      "source_id": "node-2",
      "target_id": "some-other-node-id",
      "type": "FLOWS_TO",
      "properties": {{
        "condition": "Yes",
        "description": "User is valid"
      }}
    }}
  ]
}}
```

**IMPORTANT:**
- For large diagrams (50+ nodes), focus on the most important connections to avoid response truncation.

Generate the structured JSON now:
"""
        return prompt
    

    def _parse_gemini_response(self, response_text: str, diagram_id: str) -> Tuple[List[GraphNode], List[GraphRelationship]]:
        """Parse Gemini's JSON response into GraphNode and GraphRelationship objects."""
        try:
            # Use the dedicated JSON parser utility
            data = LLMJsonParser.parse_llm_response(response_text)
            
            if data is None:
                print("❌ JSON parsing failed completely, trying fallback extraction...")
                return LLMJsonParser.fallback_extraction(response_text, diagram_id)
            
            # Parse nodes (expecting dictionary format: {"node_id": {attributes}})
            nodes_data = data.get('nodes', {})
            nodes = []
            for node_id, node_attrs in nodes_data.items():
                node = GraphNode(
                    id=node_id,
                    label=node_attrs.get('label', ''),
                    type=node_attrs.get('type', 'Unknown'),
                    diagram_id=diagram_id,
                    properties=node_attrs.get('properties', {})
                )
                nodes.append(node)
            
            # Parse relationships (expecting list format)
            relationships_data = data.get('relationships', [])
            relationships = []
            for rel_attrs in relationships_data:
                relationship = GraphRelationship(
                    id=f"rel_{len(relationships)}",
                    source_id=rel_attrs.get('source_id', ''),
                    target_id=rel_attrs.get('target_id', ''),
                    type=rel_attrs.get('type', rel_attrs.get('label', '')),  # Support both 'type' and 'label' for backward compatibility
                    diagram_id=diagram_id,
                    properties=rel_attrs.get('properties', {})
                )
                relationships.append(relationship)
            
            return nodes, relationships
            
        except Exception as e:
            print(f"❌ An unexpected error occurred during parsing: {e}")
            print("🔄 Attempting fallback extraction...")
            return LLMJsonParser.fallback_extraction(response_text, diagram_id)
