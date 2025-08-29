"""
Schema information extraction and formatting for Neo4j diagrams.
"""

from typing import Dict


class SchemaExtractor:
    """Handles schema information extraction and formatting with caching"""
    
    def __init__(self, connection):
        self.connection = connection
        self._schema_cache: Dict[str, str] = {}  # diagram_id -> schema_string
    
    def extract_schema(self, session, diagram_id: str, use_cache: bool = True) -> str:
        """Extract complete schema info for a diagram with caching"""
        # Check cache first
        if use_cache and diagram_id in self._schema_cache:
            print(f"📋 Using cached schema for diagram: {diagram_id}")
            return self._schema_cache[diagram_id]
        
        # Extract schema from database
        print(f"🔍 Extracting schema from database for diagram: {diagram_id}")
        try:
            schema_info = self._extract_with_apoc(session, diagram_id)
        except Exception as apoc_error:
            print(f"⚠️ APOC not available ({apoc_error}), falling back to standard queries...")
            schema_info = self._extract_with_standard_queries(session, diagram_id)
        
        # Cache the result
        if use_cache:
            self._schema_cache[diagram_id] = schema_info
            print(f"💾 Cached schema for diagram: {diagram_id}")
        
        return schema_info
    
    def _extract_with_apoc(self, session, diagram_id: str) -> str:
        """Extract schema using APOC functions"""
        query = """
        MATCH (n) WHERE n.diagram_id = $diagram_id
        WITH collect(DISTINCT labels(n)[0]) as node_labels,
             apoc.coll.toSet(apoc.coll.flatten(collect(DISTINCT keys(n)))) as node_properties
        MATCH (a) WHERE a.diagram_id = $diagram_id  
        OPTIONAL MATCH (a)-[r]->(b) WHERE b.diagram_id = $diagram_id
        WITH node_labels, node_properties, 
             collect(DISTINCT type(r)) as relationship_types,
             apoc.coll.toSet(apoc.coll.flatten(collect(DISTINCT keys(r)))) as relationship_properties
        RETURN {
          node_labels: node_labels,
          node_props: node_properties,
          rel_types: relationship_types,
          rel_props: relationship_properties
        } as schema
        """
        
        result = session.run(query, {"diagram_id": diagram_id})
        schema_record = result.single()
        
        if schema_record:
            schema = schema_record["schema"]
            return self._format_schema_info(
                diagram_id,
                schema['node_labels'],
                list(schema['node_props']),
                schema['rel_types'],
                list(schema['rel_props'])
            )
        return f"No data found for diagram '{diagram_id}'"
    
    def _extract_with_standard_queries(self, session, diagram_id: str) -> str:
        """Extract schema using standard Cypher queries"""
        query = """
        MATCH (n) WHERE n.diagram_id = $diagram_id
        OPTIONAL MATCH (n)-[r]-(m) WHERE m.diagram_id = $diagram_id
        RETURN 
            collect(DISTINCT labels(n)[0]) as node_labels,
            collect(DISTINCT keys(n)) as node_props,
            collect(DISTINCT type(r)) as rel_types,
            collect(DISTINCT keys(r)) as rel_props
        """
        
        result = session.run(query, {"diagram_id": diagram_id})
        schema_record = result.single()
        
        if schema_record:
            # Remove nulls and flatten lists
            node_labels = [label for label in schema_record['node_labels'] if label]
            node_props = list(set([prop for sublist in schema_record['node_props'] 
                                 for prop in sublist if prop]))
            rel_types = [rel_type for rel_type in schema_record['rel_types'] if rel_type]
            rel_props = list(set([prop for sublist in schema_record['rel_props'] 
                                for prop in sublist if prop]))
            
            return self._format_schema_info(diagram_id, node_labels, node_props, rel_types, rel_props)
        return f"No data found for diagram '{diagram_id}'"
    
    @staticmethod
    def _format_schema_info(diagram_id: str, node_types: list, node_props: list, 
                           rel_types: list, rel_props: list) -> str:
        """Format schema information consistently"""
        return f"""
=== SCHEMA FOR DIAGRAM '{diagram_id}' ===

NODE TYPES: {node_types}
NODE PROPERTIES: {sorted(node_props)}
RELATIONSHIP TYPES: {rel_types}
RELATIONSHIP PROPERTIES: {sorted(rel_props)}

SEARCH STRATEGY:
- For node queries: Check NODE TYPES and NODE PROPERTIES for content matching
- For relationship queries: Check RELATIONSHIP TYPES and RELATIONSHIP PROPERTIES for content matching  
- For mixed queries: Check both nodes AND relationships
- Use CONTAINS, =, IS NOT NULL to match user intent across all available fields
- Always include n.diagram_id = '{diagram_id}' in WHERE clause

NOTE: NODE TYPES are Neo4j labels like 'Process', 'Decision'. NODE PROPERTIES include 'label' field which contains actual node names.
"""
    
    def clear_cache(self, diagram_id: str = None):
        """Clear schema cache for specific diagram or all diagrams"""
        if diagram_id:
            if diagram_id in self._schema_cache:
                del self._schema_cache[diagram_id]
                print(f"🗑️ Cleared schema cache for diagram: {diagram_id}")
        else:
            self._schema_cache.clear()
            print("🗑️ Cleared all schema cache")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for debugging"""
        return {
            "cached_diagrams": len(self._schema_cache),
            "diagram_ids": list(self._schema_cache.keys())
        }