"""
JSON parsing utilities for handling LLM responses.

This module provides robust JSON parsing capabilities specifically designed
for handling imperfect JSON responses from Large Language Models like Gemini.
"""

import json
import re
from typing import Dict, Any, Optional


class LLMJsonParser:
    """Parser for handling imperfect JSON responses from LLMs."""
    
    @staticmethod
    def parse_llm_response(response_text: str) -> Optional[Dict[Any, Any]]:
        """
        Parse JSON from LLM response with multiple fallback strategies.
        
        Args:
            response_text: Raw response from LLM (may contain markdown, comments, etc.)
            
        Returns:
            Parsed JSON dict or None if parsing fails completely
        """
        try:
            # Step 1: Extract JSON from markdown if present
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text  # Assume raw JSON
            
            # Step 2: Try parsing raw JSON first
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
            
            # Step 3: Apply gentle cleaning and retry
            cleaned_json = LLMJsonParser._clean_json_response(json_str)
            try:
                return json.loads(cleaned_json)
            except json.JSONDecodeError:
                pass
            
            # Step 4: Apply aggressive fixes and retry
            fixed_json = LLMJsonParser._aggressive_json_fix(cleaned_json)
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError as e:
                print(f"❌ All JSON parsing attempts failed: {e}")
                print(f"Final JSON length: {len(fixed_json)} characters")
                print(f"First 200 chars: {repr(fixed_json[:200])}")
                return None
                
        except Exception as e:
            print(f"❌ Unexpected error during JSON parsing: {e}")
            return None
    
    @staticmethod
    def _clean_json_response(json_str: str) -> str:
        """Clean up common JSON issues while preserving string content."""
        # Remove JavaScript-style comments
        json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
        
        # Smart control character removal (preserve content inside strings)
        cleaned_chars = []
        in_string = False
        escape_next = False
        
        for char in json_str:
            if escape_next:
                cleaned_chars.append(char)
                escape_next = False
            elif char == '\\' and in_string:
                cleaned_chars.append(char)
                escape_next = True
            elif char == '"' and not escape_next:
                cleaned_chars.append(char)
                in_string = not in_string
            elif in_string:
                # Inside string - preserve all content (URLs, etc.)
                cleaned_chars.append(char)
            elif ord(char) >= 32 or char in '\t\n\r':
                # Outside string - only keep printable characters
                cleaned_chars.append(char)
            # Skip invalid control characters only when outside strings
        
        json_str = ''.join(cleaned_chars)
        
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix incomplete JSON structures
        open_braces = json_str.count('{') - json_str.count('}')
        open_brackets = json_str.count('[') - json_str.count(']')
        
        if open_braces > 0:
            json_str += '}' * open_braces
            print(f"⚠️  Fixed {open_braces} missing closing braces")
        
        if open_brackets > 0:
            json_str += ']' * open_brackets
            print(f"⚠️  Fixed {open_brackets} missing closing brackets")
        
        return json_str.strip()
    
    @staticmethod
    def _aggressive_json_fix(json_str: str) -> str:
        """Apply aggressive JSON fixes as a last resort."""
        print("🔧 Applying aggressive JSON repair...")
        
        # Remove ALL control characters aggressively
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
        
        # Fix broken URLs by removing incomplete ones
        json_str = re.sub(r'"https?:[^"]*"(?=\s*[,\]}])', '""', json_str)
        
        # Remove empty string values that resulted from URL fixes
        json_str = re.sub(r',\s*""', '', json_str)
        json_str = re.sub(r':\s*"",', ': null,', json_str)
        json_str = re.sub(r':\s*""([}\]])', r': null\1', json_str)
        
        # Fix malformed arrays
        json_str = re.sub(r'\[\s*,', '[', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)
        
        # Fix malformed objects
        json_str = re.sub(r'{\s*,', '{', json_str)
        json_str = re.sub(r',\s*}', '}', json_str)
        
        return json_str.strip()
    
    @staticmethod
    def fallback_extraction(text: str, diagram_id: str):
        """Extract nodes and relationships using regex when JSON parsing fails."""
        from models.graph_models import GraphNode, GraphRelationship
        
        print("🆘 Using fallback regex extraction...")
        nodes = []
        relationships = []
        
        try:
            # Extract node patterns: "node-X": { "label": "...", "type": "...", ...}
            node_pattern = r'"(node-\d+)":\s*\{\s*"label":\s*"([^"]+)"\s*,\s*"type":\s*"([^"]+)"'
            node_matches = re.findall(node_pattern, text)
            
            for node_id, label, node_type in node_matches:
                node = GraphNode(
                    id=node_id,
                    label=label,
                    type=node_type,
                    diagram_id=diagram_id,
                    properties={}
                )
                nodes.append(node)
            
            # Extract relationship patterns - support both 'type' and 'label'
            rel_pattern_type = r'"source_id":\s*"(node-\d+)"\s*,\s*"target_id":\s*"(node-\d+)"\s*,\s*"type":\s*"([^"]+)"'
            rel_pattern_label = r'"source_id":\s*"(node-\d+)"\s*,\s*"target_id":\s*"(node-\d+)"\s*,\s*"label":\s*"([^"]+)"'
            
            rel_matches = re.findall(rel_pattern_type, text)
            if not rel_matches:  # Fallback to 'label' pattern for backward compatibility
                rel_matches = re.findall(rel_pattern_label, text)
            
            for source_id, target_id, rel_type in rel_matches:
                relationship = GraphRelationship(
                    id=f"rel_{len(relationships)}",
                    source_id=source_id,
                    target_id=target_id,
                    type=rel_type,
                    diagram_id=diagram_id,
                    properties={}
                )
                relationships.append(relationship)
            
            print(f"🔄 Fallback extracted {len(nodes)} nodes and {len(relationships)} relationships")
            return nodes, relationships
            
        except Exception as e:
            print(f"❌ Fallback extraction failed: {e}")
            return [], []


