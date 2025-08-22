"""
Prompt engineering utilities for context-aware RAG systems.

Builds dynamic prompts that incorporate user preferences, conversation history,
session context, and retrieved information to create personalized and
contextually appropriate responses.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from ..models import (
        SessionState, UserPreferences, ConversationHistory, ConversationExchange,
        ResponseStyle, ExpertiseLevel
    )
except ImportError:
    from models import (
        SessionState, UserPreferences, ConversationHistory, ConversationExchange,
        ResponseStyle, ExpertiseLevel
    )


logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds context-aware prompts for LLM interactions.
    """
    
    def __init__(self):
        """Initialize the prompt builder with templates."""
        
        # Base prompt template
        self.base_template = """You are an AI assistant helping users understand and work with network diagrams and flowcharts. You have access to both textual documentation and structured diagram data.

{user_context}

{conversation_context}

{retrieved_context}

{instructions}

User Query: "{user_query}"

Your Response:"""
        
        # Response style instructions
        self.style_instructions = {
            ResponseStyle.CONCISE: "Provide a concise, to-the-point response. Focus on the essential information without unnecessary elaboration.",
            ResponseStyle.DETAILED: "Provide a comprehensive, detailed response. Include technical specifics, examples, and thorough explanations where relevant.",
            ResponseStyle.BALANCED: "Provide a well-balanced response that is informative but not overwhelming. Include key details and examples as needed."
        }
        
        # Expertise level instructions
        self.expertise_instructions = {
            ExpertiseLevel.BEGINNER: "Explain concepts in simple, accessible terms. Avoid technical jargon and provide clear, step-by-step explanations. Include basic context where helpful.",
            ExpertiseLevel.INTERMEDIATE: "Use clear explanations with appropriate technical terms. Provide practical examples and assume familiarity with basic concepts.",
            ExpertiseLevel.EXPERT: "Use precise technical language and provide detailed implementation specifics. Assume deep domain knowledge and focus on advanced concepts and nuances."
        }
        
        # Format preference templates
        self.format_templates = {
            'mermaid_diagram': "When appropriate, include Mermaid diagrams to visualize processes or relationships.",
            'bullet_points': "Structure your response using bullet points or numbered lists when presenting multiple items or steps.",
            'paragraph': "Present information in well-organized paragraphs with clear topic sentences.",
            'code_blocks': "Use code blocks for technical implementations or configurations.",
            'tables': "Use tables to present structured data or comparisons when appropriate."
        }
    
    def build_prompt(self,
                    query_context: Dict[str, Any],
                    retrieval_results: List[Dict[str, Any]],
                    graph_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Build a complete context-aware prompt.
        
        Args:
            query_context: Context from ContextManager.process_query
            retrieval_results: Filtered and ranked retrieval results
            graph_data: Optional diagram/graph data
            
        Returns:
            Complete prompt string
        """
        try:
            # Extract components from query context
            user_preferences = query_context.get('user_preferences')
            session_state = query_context.get('session_state')
            recent_history = query_context.get('recent_history', [])
            user_query = query_context.get('rewritten_query', query_context.get('original_query', ''))
            
            # Build prompt sections
            user_context = self._build_user_context(user_preferences)
            conversation_context = self._build_conversation_context(recent_history, session_state)
            retrieved_context = self._build_retrieved_context(retrieval_results, graph_data)
            instructions = self._build_instructions(user_preferences)
            
            # Assemble final prompt
            prompt = self.base_template.format(
                user_context=user_context,
                conversation_context=conversation_context,
                retrieved_context=retrieved_context,
                instructions=instructions,
                user_query=user_query
            )
            
            logger.debug("Built context-aware prompt with all components")
            
            return prompt.strip()
            
        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            # Fallback to simple prompt
            user_query = query_context.get('original_query', 'Please help me.')
            return f"Please answer the following question: {user_query}"
    
    def _build_user_context(self, preferences: UserPreferences) -> str:
        """
        Build user context section of the prompt.
        
        Args:
            preferences: User preferences
            
        Returns:
            User context string
        """
        if not preferences:
            return ""
        
        context_parts = [
            "User Profile:",
            f"- Expertise Level: {preferences.expertise_level.value}",
            f"- Preferred Response Style: {preferences.response_style.value}"
        ]
        
        if preferences.preferred_formats:
            formats = ', '.join(preferences.preferred_formats)
            context_parts.append(f"- Preferred Formats: {formats}")
        
        if preferences.topic_interest_profile:
            topics = ', '.join(preferences.topic_interest_profile[:5])  # Limit to top 5
            context_parts.append(f"- Areas of Interest: {topics}")
        
        return '\n'.join(context_parts)
    
    def _build_conversation_context(self,
                                  recent_history: List[ConversationExchange],
                                  session_state: SessionState) -> str:
        """
        Build conversation context section.
        
        Args:
            recent_history: Recent conversation exchanges
            session_state: Current session state
            
        Returns:
            Conversation context string
        """
        context_parts = []
        
        # Add conversation history
        if recent_history:
            context_parts.append("Recent Conversation:")
            
            for i, exchange in enumerate(recent_history[-3:], 1):  # Last 3 exchanges
                # Truncate long responses for context
                query_text = exchange.query_text[:200] + "..." if len(exchange.query_text) > 200 else exchange.query_text
                response_text = exchange.llm_response[:300] + "..." if len(exchange.llm_response) > 300 else exchange.llm_response
                
                context_parts.append(f"  {i}. User: {query_text}")
                context_parts.append(f"     Assistant: {response_text}")
        
        # Add session context
        if session_state:
            session_parts = []
            
            if session_state.active_entities:
                entities = ', '.join(session_state.active_entities[-5:])  # Last 5 entities
                session_parts.append(f"Active Topics: {entities}")
            
            if session_state.last_retrieved_diagram_ids:
                diagrams = ', '.join(session_state.last_retrieved_diagram_ids[-3:])  # Last 3 diagrams
                session_parts.append(f"Recent Diagrams Referenced: {diagrams}")
            
            if session_parts:
                context_parts.append("Session Context:")
                for part in session_parts:
                    context_parts.append(f"- {part}")
        
        return '\n'.join(context_parts) if context_parts else ""
    
    def _build_retrieved_context(self,
                               retrieval_results: List[Dict[str, Any]],
                               graph_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Build retrieved context section.
        
        Args:
            retrieval_results: Search results
            graph_data: Optional graph data
            
        Returns:
            Retrieved context string
        """
        context_parts = []
        
        # Add text retrieval results
        if retrieval_results:
            context_parts.append("Retrieved Information:")
            
            for i, result in enumerate(retrieval_results[:5], 1):  # Limit to top 5
                content = self._extract_result_content(result)
                # Truncate very long content
                if len(content) > 500:
                    content = content[:500] + "..."
                
                source = result.get('source', result.get('title', f'Source {i}'))
                context_parts.append(f"{i}. From {source}:")
                context_parts.append(f"   {content}")
        
        # Add graph/diagram data
        if graph_data:
            context_parts.append("\nDiagram Data:")
            
            # Format nodes and relationships
            nodes = graph_data.get('nodes', [])
            relationships = graph_data.get('relationships', [])
            
            if nodes:
                context_parts.append("Nodes:")
                for node in nodes[:10]:  # Limit to first 10 nodes
                    name = node.get('name', node.get('label', 'Unknown'))
                    node_type = node.get('type', node.get('category', ''))
                    type_info = f" ({node_type})" if node_type else ""
                    context_parts.append(f"  - {name}{type_info}")
            
            if relationships:
                context_parts.append("Relationships:")
                for rel in relationships[:10]:  # Limit to first 10 relationships
                    source = rel.get('source', rel.get('from', 'Unknown'))
                    target = rel.get('target', rel.get('to', 'Unknown'))
                    rel_type = rel.get('type', rel.get('relationship', 'connected to'))
                    context_parts.append(f"  - {source} {rel_type} {target}")
        
        return '\n'.join(context_parts) if context_parts else ""
    
    def _build_instructions(self, preferences: UserPreferences) -> str:
        """
        Build instruction section based on user preferences.
        
        Args:
            preferences: User preferences
            
        Returns:
            Instructions string
        """
        if not preferences:
            return ""
        
        instructions = ["Instructions:"]
        
        # Add style instruction
        style_instruction = self.style_instructions.get(preferences.response_style)
        if style_instruction:
            instructions.append(f"- Response Style: {style_instruction}")
        
        # Add expertise instruction
        expertise_instruction = self.expertise_instructions.get(preferences.expertise_level)
        if expertise_instruction:
            instructions.append(f"- Expertise Level: {expertise_instruction}")
        
        # Add format preferences
        if preferences.preferred_formats:
            format_instructions = []
            for format_pref in preferences.preferred_formats[:3]:  # Limit to top 3
                template = self.format_templates.get(format_pref)
                if template:
                    format_instructions.append(template)
            
            if format_instructions:
                instructions.append(f"- Formatting: {' '.join(format_instructions)}")
        
        return '\n'.join(instructions)
    
    def _extract_result_content(self, result: Dict[str, Any]) -> str:
        """
        Extract content from a search result.
        
        Args:
            result: Search result dictionary
            
        Returns:
            Extracted content string
        """
        # Try different content fields
        content_fields = ['content', 'text', 'body', 'description', 'summary']
        
        for field in content_fields:
            if field in result and result[field]:
                return str(result[field]).strip()
        
        return "Content not available"
    
    def build_system_prompt(self, preferences: UserPreferences) -> str:
        """
        Build a system prompt for chat-based interfaces.
        
        Args:
            preferences: User preferences
            
        Returns:
            System prompt string
        """
        system_parts = [
            "You are an AI assistant specializing in network diagrams and flowchart analysis.",
            "You help users understand complex systems through both textual explanations and visual diagram data."
        ]
        
        if preferences:
            # Add expertise-specific guidance
            if preferences.expertise_level == ExpertiseLevel.BEGINNER:
                system_parts.append("Focus on clear, beginner-friendly explanations with minimal jargon.")
            elif preferences.expertise_level == ExpertiseLevel.EXPERT:
                system_parts.append("Provide technical depth and assume advanced domain knowledge.")
            
            # Add style guidance
            if preferences.response_style == ResponseStyle.CONCISE:
                system_parts.append("Keep responses concise and focused on essential information.")
            elif preferences.response_style == ResponseStyle.DETAILED:
                system_parts.append("Provide comprehensive, detailed explanations with examples.")
        
        return ' '.join(system_parts)
    
    def build_follow_up_prompt(self,
                              original_query: str,
                              original_response: str,
                              follow_up_query: str,
                              preferences: UserPreferences) -> str:
        """
        Build a prompt for follow-up questions.
        
        Args:
            original_query: The original user query
            original_response: The assistant's response
            follow_up_query: The follow-up question
            preferences: User preferences
            
        Returns:
            Follow-up prompt string
        """
        prompt_parts = [
            "Previous conversation:",
            f"User: {original_query}",
            f"Assistant: {original_response[:500]}...",  # Truncate if too long
            "",
            f"Follow-up question: {follow_up_query}",
            "",
            "Please answer the follow-up question in the context of our previous conversation."
        ]
        
        # Add user preference instructions
        if preferences:
            style_instruction = self.style_instructions.get(preferences.response_style)
            if style_instruction:
                prompt_parts.append(f"Style: {style_instruction}")
        
        return '\n'.join(prompt_parts)
    
    def validate_prompt_length(self, prompt: str, max_tokens: int = 4000) -> str:
        """
        Validate and truncate prompt if too long.
        
        Args:
            prompt: Prompt to validate
            max_tokens: Maximum token limit (approximate)
            
        Returns:
            Validated prompt
        """
        # Rough approximation: 1 token ≈ 4 characters
        estimated_tokens = len(prompt) // 4
        
        if estimated_tokens <= max_tokens:
            return prompt
        
        logger.warning(f"Prompt too long ({estimated_tokens} tokens), truncating")
        
        # Truncate from the middle (preserve beginning and end)
        char_limit = max_tokens * 4
        if len(prompt) > char_limit:
            truncate_point = char_limit // 2
            truncated = (
                prompt[:truncate_point] + 
                "\n\n[... content truncated ...]\n\n" + 
                prompt[-truncate_point:]
            )
            return truncated
        
        return prompt