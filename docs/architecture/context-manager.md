
# NetBot V2 Context Manager

The Context Manager provides stateful conversation capabilities for RAG systems. In NetBot V2, it's implemented as a modular system that can be integrated into any RAG pipeline to provide session management, conversation history, and user preferences.

This document describes the current implementation in the `context_manager/` module and provides usage examples for integrating with NetBot V2.

## NetBot V2 Implementation Structure

The Context Manager in NetBot V2 is organized into modular components:

```
context_manager/
├── __init__.py                 # Module exports
├── client.py                   # Main ContextManager client
├── config.py                   # Configuration management
├── models.py                   # Data models (Session, Message, User)
├── core/                       # Core processing components
│   ├── __init__.py
│   ├── prompt_builder.py       # Context-aware prompt construction
│   ├── query_rewriter.py       # Query enhancement using context
│   └── retrieval_filter.py     # Context-based result filtering
├── storage/                    # Storage backends
│   ├── __init__.py
│   └── backends.py             # In-memory and external storage
└── utils/                      # Helper utilities
    ├── __init__.py
    └── helpers.py              # Common utilities
```

## Core Components

```mermaid
graph TD
    subgraph "NetBot V2 Context Manager"
        CLIENT["ContextManager Client"]
        CORE["Core Processing"]
        STORAGE["Storage Backends"]
        MODELS["Data Models"]
    end

    subgraph "Core Processing"
        PB["PromptBuilder"]
        QR["QueryRewriter"] 
        RF["RetrievalFilter"]
    end

    subgraph "Storage Backends"
        MEM["In-Memory Storage"]
        EXT["External Storage"]
    end

    USER["👤 User"] --> CLIENT
    CLIENT --> CORE
    CLIENT --> STORAGE
    CORE --> MODELS
    STORAGE --> MODELS

    style CLIENT fill:#e3f2fd,stroke:#333,stroke-width:2px
    style CORE fill:#e8f5e9,stroke:#333,stroke-width:2px
    style STORAGE fill:#f3e5f5,stroke:#333,stroke-width:2px
```

### 1. Data Models (`models.py`)

The Context Manager uses structured data models to represent conversation state:

```python
from context_manager.models import Session, Message, User

# Session represents a conversation instance
session = Session(
    session_id="unique_session_id",
    user_id="user_123",
    diagram_id="current_diagram",
    active_entities=["Load Balancer", "Network Router"],
    last_query_intent="find_components",
    created_at=datetime.now(),
    last_activity=datetime.now()
)

# Message represents individual conversation turns
message = Message(
    message_id="msg_001",
    session_id="unique_session_id", 
    content="Find all load balancers",
    message_type="user_query",
    timestamp=datetime.now(),
    metadata={"diagram_context": "network_001"}
)

# User represents user preferences and history
user = User(
    user_id="user_123",
    preferences={
        "response_style": "detailed",
        "expertise_level": "intermediate"
    },
    conversation_history=[],
    topic_interests=["networking", "security"]
)
```

### 2. Core Processing Components

#### PromptBuilder (`core/prompt_builder.py`)
Constructs context-aware prompts for LLM interactions:

```python
from context_manager.core.prompt_builder import PromptBuilder

builder = PromptBuilder()

# Build prompt with session context
prompt = builder.build_context_prompt(
    user_query="What are the security components?",
    session=session,
    retrieved_data=search_results,
    user_preferences=user.preferences
)
```

**Features:**
- Integrates conversation history into prompts
- Applies user preferences (response style, expertise level)
- Includes relevant diagram context
- Handles follow-up questions and pronoun resolution

#### QueryRewriter (`core/query_rewriter.py`)
Enhances user queries using conversation context:

```python
from context_manager.core.query_rewriter import QueryRewriter

rewriter = QueryRewriter()

# Enhance vague queries with context
enhanced_query = rewriter.rewrite_with_context(
    raw_query="Show me that again",
    session=session,
    conversation_history=messages[-3:]  # Last 3 messages
)
# Result: "Show the network security components from diagram network_001"
```

**Capabilities:**
- Resolves pronouns and vague references
- Adds diagram context to queries
- Maintains topic continuity
- Handles follow-up questions

#### RetrievalFilter (`core/retrieval_filter.py`)
Filters and re-ranks search results based on context:

```python
from context_manager.core.retrieval_filter import RetrievalFilter

filter = RetrievalFilter()

# Filter results based on user context
filtered_results = filter.filter_by_context(
    search_results=raw_results,
    user_preferences=user.preferences,
    session_context=session,
    relevance_threshold=0.7
)
```

**Functions:**
- Filters results by user expertise level
- Re-ranks based on conversation context
- Applies user topic preferences
- Removes duplicate or irrelevant results

### 3. Storage Backends (`storage/backends.py`)

The Context Manager supports multiple storage backends for different deployment needs:

```python
from context_manager.storage.backends import InMemoryStorage, ExternalStorage
from context_manager.client import ContextManager

# In-memory storage (for development/testing)
memory_storage = InMemoryStorage()
context_manager = ContextManager(storage_backend=memory_storage)

# External storage (for production)
external_storage = ExternalStorage(
    database_url="postgresql://user:pass@localhost/contextdb"
)
context_manager = ContextManager(storage_backend=external_storage)
```

**Storage Backend Features:**
- **InMemoryStorage**: Fast, ephemeral storage for development
- **ExternalStorage**: Persistent storage with database backing
- **Pluggable Architecture**: Easy to add new storage backends
- **Automatic Serialization**: Handles model serialization/deserialization

### 4. Client Interface (`client.py`)

The main ContextManager client provides a simple interface for integration:

```python
from context_manager.client import ContextManager

# Initialize context manager
context_manager = ContextManager(
    storage_backend=storage,
    enable_learning=True  # Enable implicit preference learning
)

# Start a new session
session = context_manager.start_session(
    user_id="user_123",
    diagram_id="network_001"
)

# Process a query with context
enhanced_query = context_manager.enhance_query(
    session_id=session.session_id,
    raw_query="Show me that firewall"
)

# Update session with results
context_manager.update_session(
    session_id=session.session_id,
    query=enhanced_query,
    response=llm_response,
    retrieved_context=search_results
)
```

---

## Integration with NetBot V2

The Context Manager is designed to integrate seamlessly with NetBot V2's RAG pipeline:

```python
from client import NetBot
from context_manager.client import ContextManager

# Initialize NetBot with Context Manager
context_manager = ContextManager()
netbot = NetBot(context_manager=context_manager)

# Start a contextual session
session = context_manager.start_session(
    user_id="user_123",
    diagram_id="network_001"
)

# Process queries with context
enhanced_query = context_manager.enhance_query(
    session_id=session.session_id,
    raw_query="Show me the firewalls"
)

# NetBot processes with enhanced context
results = netbot.search(enhanced_query, "network_001")

# Update context with results
context_manager.update_session(
    session_id=session.session_id,
    query=enhanced_query,
    response=results,
    retrieved_context=results.get('nodes', [])
)
```

### Workflow with Context

```mermaid
sequenceDiagram
    participant User
    participant ContextMgr as Context Manager
    participant NetBot
    participant GraphRAG
    participant Neo4j

    User->>+ContextMgr: "What about those routers?"
    ContextMgr->>ContextMgr: 1. Enhance Query with Context
    Note right of ContextMgr: Enhanced: "Find network routers in diagram network_001"
    
    ContextMgr->>+NetBot: 2. Process Enhanced Query
    NetBot->>+GraphRAG: 3. Search with Context
    GraphRAG->>+Neo4j: 4. Execute Contextual Search
    Neo4j-->>-GraphRAG: Returns relevant nodes/relationships
    GraphRAG-->>-NetBot: Assembled subgraph + explanation
    NetBot-->>-ContextMgr: Search results + visualization
    
    ContextMgr->>ContextMgr: 5. Update Session Context
    Note right of ContextMgr: Update active entities, conversation history
    
    ContextMgr->>-User: 6. Contextual Response
```

## Configuration (`config.py`)

The Context Manager supports flexible configuration:

```python
from context_manager.config import ContextConfig

config = ContextConfig(
    session_timeout_minutes=30,
    max_conversation_history=100,
    enable_implicit_learning=True,
    storage_backend_type="memory",  # or "external"
    context_window_size=4000,
    relevance_threshold=0.6
)

context_manager = ContextManager(config=config)
```

## Usage Examples

See `docs/examples/context-manager-usage.py` for complete integration examples with NetBot V2.

## Current Status

The Context Manager is implemented as a foundational component ready for integration. Key features include:

- **Modular Design**: Easy to integrate with existing RAG pipelines
- **Pluggable Storage**: Support for different storage backends
- **Context Processing**: Query enhancement, prompt building, result filtering
- **Session Management**: Automatic session creation and cleanup
- **Configuration**: Flexible configuration for different deployment scenarios

## Future Enhancements

Planned improvements include:

- **Advanced Learning**: Implicit user preference learning from interaction patterns
- **Multi-Diagram Context**: Cross-diagram conversation context
- **Conversation Summarization**: Automatic summarization for long conversations
- **Advanced Filtering**: Semantic relevance filtering for conversation history
- **Real-time Analytics**: Context usage analytics and optimization insights

