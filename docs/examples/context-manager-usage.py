"""
Example usage of the Context Manager system.

This script demonstrates how to set up and use the Context Manager
for a stateful RAG chatbot system.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any

# Import context manager components
from context_manager import (
    ContextManager,
    Config, 
    InMemorySessionStore, InMemoryHistoryStore, InMemoryUserStore,
    UserFeedback,
    generate_session_id
)


def setup_logging():
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_sample_retrieval_results() -> List[Dict[str, Any]]:
    """Create sample retrieval results for demonstration."""
    return [
        {
            "content": "Network switches are devices that connect multiple devices on a computer network by using packet switching to receive, process, and forward data.",
            "source": "Network Basics Guide",
            "score": 0.85,
            "diagram_id": "network_diagram_001"
        },
        {
            "content": "A router is a networking device that forwards data packets between computer networks, typically connecting a local area network to the internet.",
            "source": "Routing Fundamentals",
            "score": 0.78,
            "diagram_id": "network_diagram_001"
        },
        {
            "content": "Firewalls monitor and control incoming and outgoing network traffic based on predetermined security rules.",
            "source": "Security Architecture",
            "score": 0.72,
            "diagram_id": "security_diagram_002"
        }
    ]


def create_sample_graph_data() -> Dict[str, Any]:
    """Create sample graph data for demonstration."""
    return {
        "nodes": [
            {"name": "User Device", "type": "endpoint", "id": "node_001"},
            {"name": "Switch", "type": "network_device", "id": "node_002"},
            {"name": "Router", "type": "network_device", "id": "node_003"},
            {"name": "Firewall", "type": "security_device", "id": "node_004"},
            {"name": "Server", "type": "server", "id": "node_005"}
        ],
        "relationships": [
            {"source": "User Device", "target": "Switch", "type": "connects_to"},
            {"source": "Switch", "target": "Router", "type": "connects_to"},
            {"source": "Router", "target": "Firewall", "type": "routes_through"},
            {"source": "Firewall", "target": "Server", "type": "protects"}
        ]
    }


def demonstrate_basic_workflow():
    """Demonstrate the basic Context Manager workflow."""
    print("=== Context Manager Basic Workflow Demo ===\n")
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration (using in-memory storage for demo)
    config = Config()
    config.storage.use_in_memory = True
    
    # Initialize storage backends (in-memory for demo)
    session_store = InMemorySessionStore()
    history_store = InMemoryHistoryStore() 
    user_store = InMemoryUserStore()
    
    # Create Context Manager
    context_manager = ContextManager(
        session_store=session_store,
        history_store=history_store,
        user_store=user_store,
        session_timeout_seconds=1800
    )
    
    print("✓ Context Manager initialized with in-memory storage")
    
    # Simulate a conversation
    user_id = "demo_user_123"
    
    # First interaction
    print(f"\n--- First Interaction ---")
    query1 = "What is a network switch?"
    
    # Process the query
    query_context = context_manager.process_query(user_id, query1)
    print(f"Original query: {query_context['original_query']}")
    print(f"Rewritten query: {query_context['rewritten_query']}")
    print(f"Query intent: {query_context['query_intent']}")
    print(f"Session ID: {query_context['session_id']}")
    
    # Simulate retrieval and generate response
    retrieval_results = create_sample_retrieval_results()
    filtered_results = context_manager.filter_and_rank_results(
        retrieval_results, user_id, query_context['session_id'], query_context
    )
    
    print(f"Retrieved {len(filtered_results)} filtered results")
    
    # Build context-aware prompt
    graph_data = create_sample_graph_data()
    prompt = context_manager.build_context_aware_prompt(
        query_context, filtered_results, graph_data
    )
    
    print(f"Generated prompt length: {len(prompt)} characters")
    
    # Simulate LLM response
    llm_response = """A network switch is a crucial networking device that connects multiple devices within a local area network (LAN). It operates at the data link layer (Layer 2) of the OSI model and uses MAC addresses to forward data frames between connected devices.
    
Key functions:
• Receives data packets from connected devices
• Learns and stores MAC addresses in a forwarding table
• Forwards packets only to the intended recipient device
• Reduces network collisions compared to older hub technology

In your network diagram, the switch connects user devices to the router, enabling efficient local communication."""
    
    # Update context after response
    context_manager.update_context_after_response(
        user_id=user_id,
        session_id=query_context['session_id'],
        query_context=query_context,
        llm_response=llm_response,
        retrieval_context={"sources": [r["source"] for r in filtered_results]},
        diagram_ids_used=["network_diagram_001"],
        user_feedback=UserFeedback.THUMB_UP
    )
    
    print("✓ Context updated after first interaction")
    
    # Second interaction (follow-up)
    print(f"\n--- Follow-up Interaction ---")
    query2 = "How does it differ from a router?"
    
    # Process follow-up query (should use context)
    query_context2 = context_manager.process_query(
        user_id, query2, query_context['session_id']
    )
    
    print(f"Original query: {query_context2['original_query']}")
    print(f"Rewritten query: {query_context2['rewritten_query']}")  # Should include context
    print(f"Query intent: {query_context2['query_intent']}")
    print(f"Active entities: {query_context2['session_state'].active_entities}")
    
    # Show context summary
    context_summary = context_manager.get_context_summary(user_id, query_context['session_id'])
    print(f"\nContext Summary:")
    print(f"- Session active: {context_summary['session_active']}")
    print(f"- Conversation length: {context_summary['conversation_length']}")
    print(f"- Recent exchanges: {context_summary['recent_exchanges']}")
    
    print("\n✓ Demo completed successfully!")


def demonstrate_user_preferences():
    """Demonstrate user preference management."""
    print("\n=== User Preferences Demo ===\n")
    
    # Setup
    user_store = InMemoryUserStore()
    session_store = InMemorySessionStore()
    history_store = InMemoryHistoryStore()
    
    context_manager = ContextManager(
        session_store=session_store,
        history_store=history_store,
        user_store=user_store
    )
    
    user_id = "pref_demo_user"
    
    # Update user preferences
    success = context_manager.update_user_preferences(
        user_id,
        response_style="detailed",
        expertise_level="expert",
        preferred_formats=["mermaid_diagram", "code_blocks", "bullet_points"],
        topic_interest_profile=["networking", "security", "cloud_architecture"]
    )
    
    print(f"✓ Updated user preferences: {success}")
    
    # Get preferences
    preferences = context_manager.get_user_preferences(user_id)
    print(f"Response style: {preferences.response_style}")
    print(f"Expertise level: {preferences.expertise_level}")
    print(f"Preferred formats: {preferences.preferred_formats}")
    print(f"Topic interests: {preferences.topic_interest_profile}")
    
    # Process a query with these preferences
    query_context = context_manager.process_query(user_id, "Explain network security best practices")
    
    # Build prompt (should reflect user preferences)
    retrieval_results = create_sample_retrieval_results()
    prompt = context_manager.build_context_aware_prompt(
        query_context, retrieval_results
    )
    
    print(f"\n✓ Generated preference-aware prompt")
    print("Prompt includes expert-level instructions and detailed response style")


def demonstrate_analytics():
    """Demonstrate analytics and maintenance features."""
    print("\n=== Analytics and Maintenance Demo ===\n")
    
    from context_manager.utils import ContextAnalytics, ContextMaintenance
    
    # Setup storage
    session_store = InMemorySessionStore()
    history_store = InMemoryHistoryStore()
    user_store = InMemoryUserStore()
    
    # Create some sample data
    context_manager = ContextManager(session_store, history_store, user_store)
    
    user_id = "analytics_user"
    
    # Add some conversation history
    for i in range(5):
        query_context = context_manager.process_query(user_id, f"Query {i+1} about networking")
        context_manager.update_context_after_response(
            user_id=user_id,
            session_id=query_context['session_id'],
            query_context=query_context,
            llm_response=f"Response to query {i+1}",
            retrieval_context={"sources": ["demo_source"]},
            user_feedback=UserFeedback.THUMB_UP if i < 3 else UserFeedback.THUMB_DOWN
        )
    
    # Analytics
    analytics = ContextAnalytics(session_store, history_store, user_store)
    engagement_stats = analytics.get_user_engagement_stats(user_id)
    
    print("User Engagement Statistics:")
    print(f"- Total exchanges: {engagement_stats['total_exchanges']}")
    print(f"- Satisfaction rate: {engagement_stats['satisfaction_rate']:.2f}")
    print(f"- Feedback distribution: {engagement_stats['feedback_distribution']}")
    
    # Maintenance
    maintenance = ContextMaintenance(session_store, history_store, user_store)
    optimized = maintenance.optimize_user_preferences(user_id)
    print(f"✓ Preference optimization attempted: {optimized}")
    
    system_metrics = analytics.get_system_metrics()
    print(f"System status: {system_metrics['status']}")


if __name__ == "__main__":
    """Run all demonstrations."""
    try:
        demonstrate_basic_workflow()
        demonstrate_user_preferences()
        demonstrate_analytics()
        
        print("\n🎉 All demonstrations completed successfully!")
        print("\nTo use the Context Manager in your own application:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure storage backends (Redis, MongoDB, PostgreSQL)")
        print("3. Set environment variables (see create_example_env_file())")
        print("4. Initialize ContextManager with your storage backends")
        print("5. Use process_query() -> filter_and_rank_results() -> build_context_aware_prompt() workflow")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        logging.exception("Demonstration failed")