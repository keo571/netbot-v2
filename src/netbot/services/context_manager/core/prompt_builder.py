"""
Context-aware prompt construction for LLM interactions.

Builds prompts that incorporate conversation history, user preferences,
and contextual information for improved response quality.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from ....shared import get_logger
from ..models import Session, Message, User, ResponseStyle, ExpertiseLevel


class PromptBuilder:
    """
    Constructs context-aware prompts for LLM interactions.
    
    Integrates conversation history, user preferences, and contextual
    information to create rich, personalized prompts.
    """
    
    def __init__(self):
        """Initialize the prompt builder."""
        self.logger = get_logger(__name__)
        
        # Prompt templates for different scenarios
        self.base_templates = {
            'search_query': self._get_search_template(),
            'explanation': self._get_explanation_template(),
            'follow_up': self._get_follow_up_template(),
            'summary': self._get_summary_template()
        }
    
    def build_context_prompt(self,
                           user_query: str,
                           session: Session,
                           user: User,
                           retrieved_data: List[Dict[str, Any]] = None,
                           conversation_history: List[Message] = None,
                           prompt_type: str = 'search_query') -> str:
        """
        Build a context-aware prompt for LLM interaction.
        
        Args:
            user_query: The user's query
            session: Current session context
            user: User preferences and profile
            retrieved_data: Retrieved search results
            conversation_history: Recent conversation history
            prompt_type: Type of prompt to build
            
        Returns:
            Contextually enriched prompt
        """
        try:
            # Get base template
            template = self.base_templates.get(prompt_type, self.base_templates['search_query'])
            
            # Build context components
            context_components = {
                'user_context': self._build_user_context(user),
                'session_context': self._build_session_context(session),
                'conversation_context': self._build_conversation_context(conversation_history or []),
                'data_context': self._build_data_context(retrieved_data or []),
                'query': user_query,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Apply user preferences to template selection and formatting
            template = self._customize_template_for_user(template, user)
            
            # Fill template with context
            prompt = template.format(**context_components)
            
            # Apply final formatting based on user preferences
            prompt = self._apply_user_formatting(prompt, user)
            
            self.logger.debug(f"Built {prompt_type} prompt with context for user {user.user_id}")
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"Failed to build context prompt: {e}")
            # Return fallback prompt
            return self._build_fallback_prompt(user_query, user)
    
    def build_explanation_prompt(self,
                               nodes: List[Dict[str, Any]],
                               relationships: List[Dict[str, Any]],
                               original_query: str,
                               user: User,
                               session: Session) -> str:
        """
        Build a prompt for generating explanations of graph results.
        
        Args:
            nodes: Graph nodes to explain
            relationships: Graph relationships to explain
            original_query: Original user query
            user: User preferences
            session: Session context
            
        Returns:
            Explanation prompt
        """
        try:
            # Format graph data for explanation
            nodes_text = self._format_nodes_for_explanation(nodes)
            relationships_text = self._format_relationships_for_explanation(relationships)
            
            # Build explanation context
            explanation_context = {
                'original_query': original_query,
                'nodes_text': nodes_text,
                'relationships_text': relationships_text,
                'user_expertise': user.expertise_level.value,
                'response_style': user.response_style.value,
                'diagram_context': session.diagram_id or 'unknown',
                'active_entities': ', '.join(session.active_entities) if session.active_entities else 'none'
            }
            
            # Select explanation template based on user expertise and style
            template = self._get_explanation_template_for_user(user)
            
            return template.format(**explanation_context)
            
        except Exception as e:
            self.logger.error(f"Failed to build explanation prompt: {e}")
            return f"Explain the following graph data in response to: {original_query}\\n\\nNodes: {nodes}\\nRelationships: {relationships}"
    
    def _build_user_context(self, user: User) -> str:
        """Build user context section."""
        context_parts = [
            f"User expertise level: {user.expertise_level.value}",
            f"Preferred response style: {user.response_style.value}"
        ]
        
        if user.topic_interests:
            context_parts.append(f"User interests: {', '.join(user.topic_interests[:5])}")
        
        if user.frequent_entities:
            top_entities = user.get_top_entities(3)
            if top_entities:
                entities_text = ', '.join([entity for entity, _ in top_entities])
                context_parts.append(f"Frequently discussed entities: {entities_text}")
        
        # Add learned preferences
        if user.learned_preferences:
            prefs = []
            for key, value in list(user.learned_preferences.items())[:3]:
                prefs.append(f"{key}: {value}")
            if prefs:
                context_parts.append(f"Learned preferences: {'; '.join(prefs)}")
        
        return "\\n".join(context_parts)
    
    def _build_session_context(self, session: Session) -> str:
        """Build session context section."""
        context_parts = [
            f"Current session: {session.session_id}",
            f"Session duration: {session.duration_minutes:.1f} minutes"
        ]
        
        if session.diagram_id:
            context_parts.append(f"Current diagram: {session.diagram_id}")
        
        if session.active_entities:
            entities_list = ', '.join(list(session.active_entities)[:5])
            context_parts.append(f"Currently discussing: {entities_list}")
        
        if session.conversation_topic:
            context_parts.append(f"Conversation topic: {session.conversation_topic}")
        
        if session.last_query_intent:
            context_parts.append(f"Last query intent: {session.last_query_intent}")
        
        return "\\n".join(context_parts)
    
    def _build_conversation_context(self, messages: List[Message]) -> str:
        """Build conversation history context."""
        if not messages:
            return "No previous conversation history."
        
        # Get recent messages (last 5)
        recent_messages = messages[-5:]
        
        context_parts = ["Recent conversation:"]
        for i, message in enumerate(recent_messages):
            role = "User" if message.message_type.value == "user_query" else "System"
            # Truncate long messages
            content = message.content[:100] + "..." if len(message.content) > 100 else message.content
            context_parts.append(f"{i+1}. {role}: {content}")
        
        return "\\n".join(context_parts)
    
    def _build_data_context(self, retrieved_data: List[Dict[str, Any]]) -> str:
        """Build context from retrieved search results."""
        if not retrieved_data:
            return "No additional data context available."
        
        context_parts = ["Retrieved information:"]
        
        # Limit to most relevant items
        for i, item in enumerate(retrieved_data[:3]):
            if 'label' in item and 'type' in item:
                context_parts.append(f"- {item['label']} ({item['type']})")
            elif 'content' in item:
                content = item['content'][:80] + "..." if len(item['content']) > 80 else item['content']
                context_parts.append(f"- {content}")
        
        return "\\n".join(context_parts)
    
    def _format_nodes_for_explanation(self, nodes: List[Dict[str, Any]]) -> str:
        """Format nodes for explanation prompts."""
        if not nodes:
            return "No nodes to explain."
        
        formatted_nodes = []
        for node in nodes[:10]:  # Limit to avoid token overflow
            if isinstance(node, dict):
                label = node.get('label', 'Unknown')
                node_type = node.get('type', 'Unknown')
                formatted_nodes.append(f"- {label} ({node_type})")
            else:
                # Handle node objects
                formatted_nodes.append(f"- {getattr(node, 'label', 'Unknown')} ({getattr(node, 'type', 'Unknown')})")
        
        return "\\n".join(formatted_nodes)
    
    def _format_relationships_for_explanation(self, relationships: List[Dict[str, Any]]) -> str:
        """Format relationships for explanation prompts."""
        if not relationships:
            return "No relationships to explain."
        
        formatted_rels = []
        for rel in relationships[:10]:  # Limit to avoid token overflow
            if isinstance(rel, dict):
                rel_type = rel.get('type', 'CONNECTED')
                source = rel.get('source_label', 'Unknown')
                target = rel.get('target_label', 'Unknown')
                formatted_rels.append(f"- {source} → {target} ({rel_type})")
            else:
                # Handle relationship objects
                rel_type = getattr(rel, 'type', 'CONNECTED')
                formatted_rels.append(f"- Relationship: {rel_type}")
        
        return "\\n".join(formatted_rels)
    
    def _customize_template_for_user(self, template: str, user: User) -> str:
        """Customize template based on user preferences."""
        # Adjust complexity based on expertise level
        if user.expertise_level == ExpertiseLevel.BEGINNER:
            template = template.replace("{technical_instructions}", 
                                      "Use simple, non-technical language. Explain any technical terms.")
        elif user.expertise_level == ExpertiseLevel.EXPERT:
            template = template.replace("{technical_instructions}", 
                                      "Use precise technical terminology. Provide detailed technical insights.")
        else:
            template = template.replace("{technical_instructions}", 
                                      "Use clear technical language with brief explanations of complex terms.")
        
        return template
    
    def _apply_user_formatting(self, prompt: str, user: User) -> str:
        """Apply final formatting based on user preferences."""
        # Add response style instructions
        if user.response_style == ResponseStyle.BRIEF:
            prompt += "\\n\\nIMPORTANT: Keep your response concise and focused. Aim for 2-3 sentences maximum."
        elif user.response_style == ResponseStyle.DETAILED:
            prompt += "\\n\\nIMPORTANT: Provide a comprehensive, detailed response with examples and context."
        elif user.response_style == ResponseStyle.TECHNICAL:
            prompt += "\\n\\nIMPORTANT: Focus on technical details, specifications, and precise terminology."
        elif user.response_style == ResponseStyle.CONVERSATIONAL:
            prompt += "\\n\\nIMPORTANT: Use a friendly, conversational tone. Relate to previous context naturally."
        
        return prompt
    
    def _get_search_template(self) -> str:
        """Get base search query template."""
        return """You are an expert network diagram analyst helping a user understand and search through technical diagrams.

USER CONTEXT:
{user_context}

SESSION CONTEXT:
{session_context}

CONVERSATION CONTEXT:
{conversation_context}

RETRIEVED DATA:
{data_context}

USER QUERY: {query}

{technical_instructions}

Please provide a helpful response based on the user's query and the available context. If you're explaining technical components, consider the user's expertise level and preferred response style."""
    
    def _get_explanation_template(self) -> str:
        """Get explanation template.""" 
        return """You are explaining network diagram components to a user with {user_expertise} level expertise who prefers {response_style} responses.

QUERY: {original_query}

DIAGRAM CONTEXT: {diagram_context}
CURRENTLY DISCUSSING: {active_entities}

COMPONENTS TO EXPLAIN:
{nodes_text}

CONNECTIONS:
{relationships_text}

Please provide a clear explanation of these components and their relationships, tailored to the user's expertise level and response style preferences."""
    
    def _get_follow_up_template(self) -> str:
        """Get follow-up query template."""
        return """You are continuing a conversation about network diagrams.

PREVIOUS CONTEXT:
{conversation_context}

SESSION CONTEXT:
{session_context}

FOLLOW-UP QUERY: {query}

{technical_instructions}

Please respond to this follow-up query, taking into account the previous conversation and maintaining context continuity."""
    
    def _get_summary_template(self) -> str:
        """Get conversation summary template."""
        return """Summarize the following conversation about network diagrams:

CONVERSATION HISTORY:
{conversation_context}

SESSION CONTEXT:
{session_context}

Please provide a concise summary of what was discussed, key findings, and any important conclusions."""
    
    def _get_explanation_template_for_user(self, user: User) -> str:
        """Get explanation template customized for specific user."""
        base_template = self._get_explanation_template()
        
        # Customize based on expertise and style
        if user.expertise_level == ExpertiseLevel.BEGINNER and user.response_style == ResponseStyle.CONVERSATIONAL:
            return base_template + "\\n\\nUse friendly, simple language. Avoid jargon and explain things step by step."
        elif user.expertise_level == ExpertiseLevel.EXPERT and user.response_style == ResponseStyle.TECHNICAL:
            return base_template + "\\n\\nProvide detailed technical analysis with precise terminology and architectural insights."
        else:
            return base_template + "\\n\\nProvide a clear, informative explanation appropriate for the user's background."
    
    def _build_fallback_prompt(self, user_query: str, user: User) -> str:
        """Build a simple fallback prompt when context building fails."""
        return f"""Please help answer this question: {user_query}

User expertise level: {user.expertise_level.value}
Preferred response style: {user.response_style.value}

Provide a helpful response appropriate for this user's background and preferences."""