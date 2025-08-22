# NetBot v2 🤖

**AI-powered network diagram processing and GraphRAG system for transforming network diagrams into knowledge graphs**

NetBot v2 is a sophisticated system that converts network diagrams and flowcharts into structured knowledge graphs, enabling powerful semantic search and analysis through advanced Graph-based Retrieval Augmented Generation (GraphRAG) capabilities.

## 🚀 Features

- **3-Phase Processing Pipeline**: OCR → Relationship Generation → Export & Storage
- **Advanced GraphRAG**: Semantic search with embeddings and vector storage
- **Multiple Diagram Types**: Network diagrams, flowcharts, and mixed architectures
- **Knowledge Graph Generation**: Automatic Neo4j graph database population
- **Visual Analysis**: AI-powered relationship extraction using Gemini 2.5 Pro
- **Modular Architecture**: Pluggable components with unified CLI interfaces
- **Context Management**: Session handling and conversation history
- **Rich Visualization**: Multiple graph visualization backends

## 📋 Quick Start

### Installation

```bash
# Basic installation
pip install -e .

# With optional features
pip install -e ".[vision,storage,nlp,vector]"

# Development setup
pip install -e ".[dev]"

# Install everything
pip install -e ".[all]"
```

### Environment Setup

Create a `.env` file:

```bash
# Required
GEMINI_API_KEY="your-gemini-api-key"
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="your-password"

# Optional
GOOGLE_APPLICATION_CREDENTIALS="path/to/google-credentials.json"
```

### Quick Example

```bash
# Complete workflow (recommended)
python cli.py quickstart data/examples/network_diagram.png diagram_001 "find load balancers"

# Step-by-step workflow
python cli.py process-and-search data/examples/network_diagram.png diagram_001 "find servers" --visualize --explain
```

## 🏗️ Architecture

### Core Components

- **`diagram_processing/`** - 3-phase image-to-graph pipeline
- **`graph_rag/`** - Advanced retrieval system with embeddings
- **`embeddings/`** - Vector encoding and semantic search
- **`context_manager/`** - Session and conversation management
- **`interfaces/`** - Unified CLI system

### System Architecture Overview

```mermaid
graph TB
    subgraph "Input Layer"
        A[Network Diagrams]
        B[Flowcharts]
        C[Mixed Architectures]
    end
    
    subgraph "Processing Layer"
        D[diagram_processing]
        E[embeddings]
        F[graph_rag]
    end
    
    subgraph "Storage Layer"
        G[Neo4j Graph DB]
        H[Vector Store]
        I[CSV Export]
    end
    
    subgraph "Interface Layer"
        J[CLI Interface]
        K[Python API]
        L[Visualization]
    end
    
    A --> D
    B --> D
    C --> D
    D --> G
    D --> I
    D --> E
    E --> H
    G --> F
    H --> F
    F --> J
    F --> K
    F --> L
```

## 🔄 Core System Components

### 1. Diagram Processing Pipeline

The `diagram_processing` module transforms images into structured knowledge graphs through a 3-phase pipeline:

```mermaid
graph TB
    subgraph "Phase 1: OCR & Preprocessing"
        A1[Image Input] --> B1[Google Cloud Vision OCR]
        A1 --> C1[OpenCV Shape Detection]
        B1 --> D1[Text Extraction]
        C1 --> E1[Geometric Analysis]
        D1 --> F1[Diagram Type Detection]
        E1 --> F1
    end
    
    subgraph "Phase 2: AI Relationship Generation"
        F1 --> G2[Gemini 2.5 Pro Analysis]
        G2 --> H2[Visual Reasoning]
        H2 --> I2[Relationship Extraction]
        I2 --> J2[JSON Structure Generation]
    end
    
    subgraph "Phase 3: Export & Storage"
        J2 --> K3[Data Validation]
        K3 --> L3[CSV Generation]
        K3 --> M3[Neo4j Storage]
        L3 --> N3[nodes.csv + relationships.csv]
        M3 --> O3[Typed Graph Database]
    end
    
    style A1 fill:#e1f5fe
    style G2 fill:#f3e5f5
    style N3 fill:#e8f5e8
    style O3 fill:#e8f5e8
```

**Key Features:**
- **Smart OCR**: Extracts text with positional awareness
- **Shape Recognition**: Identifies boxes, circles, diamonds, arrows
- **AI-Powered Analysis**: Uses Gemini 2.5 Pro for visual understanding
- **Robust Parsing**: Multi-tiered JSON extraction with fallbacks
- **Typed Storage**: Creates strongly-typed Neo4j graph structures

### 2. Embeddings System

The `embeddings` module provides semantic understanding and vector search capabilities:

```mermaid
graph TB
    subgraph "Embedding Generation"
        A2[Graph Nodes & Relationships] --> B2[Text Preprocessing]
        B2 --> C2[Sentence Transformers]
        C2 --> D2[Vector Embeddings]
        D2 --> E2[Dimension: 384/768/1024]
    end
    
    subgraph "Vector Storage"
        E2 --> F2[ChromaDB Store]
        E2 --> G2[In-Memory Cache]
        F2 --> H2[Persistent Storage]
        G2 --> I2[Fast Retrieval]
    end
    
    subgraph "Semantic Search"
        J2[Natural Language Query] --> K2[Query Embedding]
        K2 --> L2[Similarity Search]
        I2 --> L2
        H2 --> L2
        L2 --> M2[Ranked Results]
    end
    
    subgraph "Hybrid Manager"
        M2 --> N2[Traditional Graph Search]
        M2 --> O2[Vector Similarity]
        N2 --> P2[Combined Ranking]
        O2 --> P2
        P2 --> Q2[Final Results]
    end
    
    style C2 fill:#fff3e0
    style F2 fill:#e8f5e8
    style L2 fill:#f3e5f5
    style P2 fill:#e1f5fe
```

**Key Features:**
- **Multi-Model Support**: Various Sentence Transformer models
- **Hybrid Search**: Combines semantic and structural search
- **Caching Strategy**: Intelligent embedding cache management
- **Chunking Support**: Advanced document chunking for large diagrams
- **Vector Stores**: ChromaDB integration with extensible backends

### 3. GraphRAG System

The `graph_rag` module provides intelligent retrieval and reasoning over knowledge graphs:

```mermaid
graph TB
    subgraph "Query Processing"
        A3[Natural Language Query] --> B3[Query Analysis]
        B3 --> C3[Intent Classification]
        C3 --> D3[Query Rewriting]
    end
    
    subgraph "Two-Phase Retrieval"
        D3 --> E3[Phase 1: Structural Search]
        D3 --> F3[Phase 2: Semantic Search]
        
        E3 --> G3[Cypher Query Generation]
        G3 --> H3[Neo4j Graph Traversal]
        
        F3 --> I3[Vector Similarity Search]
        I3 --> J3[Embedding Matching]
    end
    
    subgraph "Result Fusion"
        H3 --> K3[Structural Results]
        J3 --> L3[Semantic Results]
        K3 --> M3[Result Ranking]
        L3 --> M3
        M3 --> N3[Context Assembly]
    end
    
    subgraph "Response Generation"
        N3 --> O3[Answer Generation]
        O3 --> P3[Explanation Building]
        P3 --> Q3[Visualization Data]
        Q3 --> R3[Final Response]
    end
    
    subgraph "Visualization Engine"
        R3 --> S3[NetworkX Layout]
        R3 --> T3[PyVis Interactive]
        R3 --> U3[Graphviz Diagrams]
        S3 --> V3[Static Graphs]
        T3 --> W3[Interactive HTML]
        U3 --> X3[Publication Quality]
    end
    
    style G3 fill:#e3f2fd
    style I3 fill:#fff3e0
    style M3 fill:#f3e5f5
    style O3 fill:#e8f5e8
```

**Key Features:**
- **Intelligent Querying**: Context-aware query understanding
- **Dual Retrieval**: Combines graph traversal with vector search
- **Result Fusion**: Smart ranking and deduplication
- **Rich Visualization**: Multiple rendering backends
- **Explainable AI**: Provides reasoning traces for results

## 🔧 Usage

### Command Line Interface

#### Main CLI (Orchestration)
```bash
# Complete automated workflow
python cli.py quickstart image.png diagram_id "query"

# Advanced processing with visualization
python cli.py process-and-search image.png diagram_id "query" --visualize --explain
```

#### Specialized Module CLIs
```bash
# Diagram processing
python -m diagram_processing process image.png diagram_id

# GraphRAG operations
python -m graph_rag search "find load balancers" diagram_id
python -m graph_rag visualize "show topology" diagram_id --backend graphviz

# Embeddings management
python -m embeddings add diagram_id
```

### Python API

#### Quick Processing
```python
from diagram_processing import process_diagram

# Process single diagram
result = process_diagram("image.png", "diagram_001")
print(f"Found {len(result.nodes)} nodes and {len(result.relationships)} relationships")
```

#### Full GraphRAG System
```python
from graph_rag import GraphRAG

# Initialize system
rag = GraphRAG()

# Process and add embeddings
result = rag.process_diagram("image.png", "diagram_001")
rag.add_embeddings("diagram_001")

# Semantic search
results = rag.search(
    query="Find all load balancers and their connections",
    diagram_id="diagram_001"
)

# Visualize results
rag.visualize(results, "results.html")
rag.close()
```

## 📊 Output Structure

```
data/processed/
├── [image_name]/
│   ├── nodes.csv               # Extracted nodes with properties
│   ├── relationships.csv       # Relationships between nodes
│   └── pipeline_metadata.json  # Processing metadata
```

## 🎯 Key Features

### Diagram Processing
- **OCR Integration**: Google Cloud Vision API for text extraction
- **Shape Detection**: OpenCV-based geometric analysis
- **AI Relationship Extraction**: Gemini 2.5 Pro visual reasoning
- **Multi-format Export**: CSV, JSON, Neo4j direct storage

### GraphRAG System
- **Two-Phase Retrieval**: Structural + semantic search
- **Vector Embeddings**: Sentence Transformers integration
- **Caching**: Intelligent embedding and query caching
- **Visualization**: NetworkX, PyVis, Graphviz backends

### Context Management
- **Session Handling**: Stateful conversation management
- **Storage Backends**: In-memory, Redis, MongoDB, PostgreSQL
- **Query Rewriting**: Intelligent context-aware query enhancement
- **Analytics**: Usage tracking and performance monitoring

## 🗂️ Repository Structure

```
netbot-v2/
├── cli.py                      # Main orchestration CLI
├── diagram_processing/         # Core processing pipeline
│   ├── pipeline/               # 3-phase processing
│   ├── models/                 # Data models
│   └── utils/                  # OCR, shape detection, JSON parsing
├── graph_rag/                  # GraphRAG system
│   ├── retrieval/              # Two-phase retrieval
│   ├── search/                 # Vector search & caching
│   └── visualization/          # Graph visualization
├── embeddings/                 # Embedding management
│   ├── advanced/               # Hybrid RAG features
│   └── vector_stores/          # Vector database support
├── context_manager/            # Session management
│   ├── core/                   # Processing components
│   ├── storage/                # Storage backends
│   └── utils/                  # Analytics & maintenance
└── docs/                       # Architecture & guides
```

## 🔧 Dependencies

### Core Requirements
- **Neo4j** - Graph database storage
- **Google Gemini API** - Visual reasoning and relationship extraction
- **Sentence Transformers** - Semantic embeddings
- **OpenCV** - Image processing and shape detection
- **Pillow** - Image manipulation

### Optional Features
- **Google Cloud Vision** - Advanced OCR capabilities
- **ChromaDB** - Vector storage for semantic search
- **Redis/MongoDB/PostgreSQL** - External storage backends
- **Graphviz** - Advanced graph visualization

## 📖 Documentation

- **Architecture Guides**: `docs/architecture/` - Detailed system design
- **Examples**: `docs/examples/` - Usage examples and patterns
- **API Reference**: Module docstrings and type hints

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest`
5. Format code: `black . && isort .`
6. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built with Google Gemini 2.5 Pro for visual AI reasoning
- Powered by Neo4j for graph storage and querying
- Uses Sentence Transformers for semantic embeddings
- Visualization powered by NetworkX, PyVis, and Graphviz

---

**NetBot v2** - Transforming visual diagrams into intelligent knowledge graphs 🚀