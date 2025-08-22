"""
LLM-powered Cypher query generation for graph retrieval.
"""

import re
import google.generativeai as genai


class CypherGenerator:
    """Generates Cypher queries from natural language using LLM"""
    
    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def generate_cypher_query(self, natural_query: str, diagram_id: str, schema_info: str) -> str:
        """Use LLM to generate Cypher query from natural language for a specific diagram"""
        
        prompt = f"""You are a Neo4j Cypher expert. Generate a precise Cypher query from the user's natural language query.

DATABASE SCHEMA:
{schema_info}

USER QUERY: {natural_query}

QUERY TEMPLATE:
MATCH (n) WHERE n.diagram_id = '{diagram_id}' AND [conditions] 
OPTIONAL MATCH (n)-[r]-(m) WHERE m.diagram_id = '{diagram_id}' 
RETURN n, r, m LIMIT 50

SEARCH STRATEGY:
- Node types: labels(n)[0] = 'ProcessType' OR 'Type' IN labels(n)
- Node properties: n.label CONTAINS 'keyword' OR n.description CONTAINS 'keyword'
- Relationship types: type(r) = 'FLOWS_TO' OR type(r) CONTAINS 'CONNECT'
- Use CONTAINS for partial matching on properties only (not on labels() or type())
- Combine with OR for broader results, AND for specificity

FORBIDDEN: id(), startNode(), endNode(), SQL comments (--)

Generate ONE clean Cypher query:"""
        
        try:
            response = self.model.generate_content(prompt)
            cypher_query = response.text.strip()
            
            # Remove markdown code blocks if present
            if cypher_query.startswith('```'):
                cypher_query = cypher_query.split('```')[1]
                if cypher_query.startswith('cypher'):
                    cypher_query = cypher_query[6:].strip()
            
            # Clean up invalid syntax
            cypher_query = self._clean_cypher_query(cypher_query)
            
            return cypher_query
        except Exception as e:
            print(f"Error generating Cypher query: {e}")
            return ""
    
    def _clean_cypher_query(self, query: str) -> str:
        """Clean up common LLM-generated Cypher syntax issues"""
        
        # Remove SQL-style comments (-- comments) which are invalid in Cypher
        lines = query.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove lines that start with -- comments
            if line.strip().startswith('--'):
                continue
            # Remove inline -- comments
            if '--' in line:
                line = line.split('--')[0].rstrip()
            cleaned_lines.append(line)
        
        query = '\n'.join(cleaned_lines)
        
        # Remove extra whitespace and empty lines
        query = re.sub(r'\n\s*\n+', '\n', query)
        query = query.strip()
        
        return query