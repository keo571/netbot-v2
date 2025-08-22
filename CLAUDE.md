# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains `netbot-v2`, a sophisticated AI-powered system for transforming network diagrams and flowcharts into knowledge graphs. The project consists of multiple integrated components:

1. **diagram_processing**: Core pipeline for converting images to structured graph data (Phase 1-3 pipeline)
2. **graph_rag**: Advanced graph-based retrieval system with embedding support
3. **interfaces**: Unified CLI system for all operations
4. **Architecture Documentation**: Detailed design documents for hybrid RAG systems and context management

## Core Commands

### Environment Setup
```bash
# Install Python dependencies (basic installation)
pip install -r requirements.txt

# Or install as a package with optional dependencies
pip install -e .                          # Core functionality
pip install -e ".[vision,storage,nlp]"    # With optional features
pip install -e ".[dev]"                   # Development environment

# Required environment variables (use .env file)
export GEMINI_API_KEY="your-gemini-api-key"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/google-credentials.json"  # Optional for OCR
```

### Primary CLI Interface (Recommended)
```bash
# Complete automated workflow (quickstart)
python cli.py quickstart data/examples/network_diagram.png diagram_001 "find load balancers"

# Advanced workflow with more control
python cli.py process-and-search data/examples/network_diagram.png diagram_001 "find servers" --visualize --explain
```

### Specialized Module CLIs
```bash
# GraphRAG operations
python -m graph_rag search "find load balancers" diagram_001
python -m graph_rag visualize "show topology" diagram_001 --backend graphviz --explain
python -m graph_rag explain "network architecture" diagram_001 --detailed

# Diagram processing
python -m diagram_processing process data/examples/diagram.png diagram_001

# Embeddings management
python -m embeddings add diagram_001
```

### Direct Pipeline Access
```bash
# Process single image (direct access)
python diagram_processing/main.py data/examples/network_diagram.png

# Batch process multiple images
python diagram_processing/batch_process.py data/examples/ data/processed/

# Using pipeline CLI directly
python diagram_processing/pipeline/cli/main.py data/examples/diagram.png --output-dir data/processed/
```

### Legacy Tools
```bash
# Standalone OCR tool (legacy)
python diagram_processing/ocr_tool.py image.png
```

### Testing and Development
The project uses Python's standard practices. Check for test files or CI configuration in the diagram_processing directory for specific testing commands.

## Architecture Overview

### System Components

The repository is organized into distinct, modular components:

```
netbot-v2/
├── cli.py                      # Main orchestration CLI
├── diagram_processing/         # Core image-to-graph pipeline
│   ├── pipeline/               # 3-phase processing pipeline
│   ├── models/                 # Data models (Node, Relationship, Shape)
│   ├── data/                   # All data files
│   │   ├── examples/           # Sample images
│   │   ├── processed/          # Generated CSV/JSON results
│   │   └── visualizations/     # Charts and graphs
│   ├── cli.py                  # Specialized processing CLI
│   ├── main.py                 # Direct entry point
│   └── batch_process.py        # Batch processing
├── graph_rag/                  # Advanced retrieval system
│   ├── cli.py                  # Specialized GraphRAG CLI
│   ├── retrieval/              # Two-phase retrieval with embeddings
│   ├── search/                 # Vector search and caching
│   └── database/               # Neo4j integration
├── embeddings/                 # Embedding management
│   ├── cli.py                  # Specialized embeddings CLI
│   ├── client.py               # Core embeddings client
│   ├── embedding_encoder.py    # Core encoder functionality
│   └── advanced/               # Advanced/experimental features
│       ├── hybrid_manager.py   # Hybrid RAG manager
│       ├── chunking/           # Document chunking
│       └── vector_stores/      # Vector database support\n├── graph_database/             # Database operations
└── context_manager/            # Session and context management
    ├── core/                   # Core processing components
    ├── storage/                # Storage backends (in-memory and external)
    └── utils/                  # Analytics and maintenance tools
```

### diagram_processing Pipeline (Phase 1-3)
The core pipeline consists of 3 phases:

1. **Phase 1: OCR & Preprocessing** (`pipeline/core/preprocessor.py`, `pipeline/utils/ocr.py`)
   - Google Cloud Vision API for text extraction
   - OpenCV shape detection (`pipeline/utils/shape_detection.py`)
   - Automatic diagram type detection (network/flowchart/mixed)

2. **Phase 2: Relationship Generation** (`pipeline/core/gemini_generator.py`)
   - Gemini 2.5 Pro for visual reasoning and relationship extraction
   - Advanced prompting for different diagram types
   - Robust JSON parsing with fallback strategies (`pipeline/utils/json_utils.py`)

3. **Phase 3: Export & Storage** (`pipeline/core/exporter.py`)
   - CSV file generation (nodes.csv, relationships.csv)
   - Neo4j database storage with typed nodes and relationships
   - Complete pipeline results saved as JSON

### Key Components

- **Pipeline Orchestrator**: `diagram_processing/pipeline/core/pipeline.py` - Main `KnowledgeGraphPipeline` class
- **Main CLI**: `cli.py` - Orchestration interface for complete workflows
- **Specialized CLIs**: `graph_rag/cli.py`, `diagram_processing/cli.py`, `embeddings/cli.py` - Module-specific operations
- **Data Models**: `diagram_processing/models/graph_models.py` - Node, Relationship, and Shape classes
- **Graph RAG**: `graph_rag/main_querier.py` - Advanced retrieval with embeddings
- **Utilities**: `diagram_processing/pipeline/utils/` - JSON parsing, text processing, shape detection

### Output Structure
```
data/processed/
├── [image_name]/
│   ├── nodes.csv               # Extracted nodes with properties
│   ├── relationships.csv       # Relationships between nodes
│   └── pipeline_metadata.json  # Processing metadata and file references
```

## Programming Usage

### Diagram Processing (Direct)
```python
from diagram_processing import DiagramProcessor
# OR for quick processing
from diagram_processing import process_diagram

# Method 1: Using DiagramProcessor class
processor = DiagramProcessor(gemini_api_key="your-key")
result = processor.process(
    image_path="data/examples/network_diagram.png",
    diagram_id="diagram_001",
    output_dir="data/processed/diagrams"
)
processor.close()

# Method 2: Quick function
result = process_diagram("data/examples/network_diagram.png", "diagram_001")
```

### GraphRAG (Full System - Recommended)
```python
from graph_rag import GraphRAG

# Initialize the complete system
rag = GraphRAG()

# Process diagram
result = rag.process_diagram(
    image_path="data/examples/network_diagram.png",
    diagram_id="diagram_001"
)

# Add embeddings for semantic search
rag.add_embeddings("diagram_001")

# Query with natural language
results = rag.search(
    query="Find all load balancers and their connections",
    diagram_id="diagram_001"
)

# Visualize results
rag.visualize(results, "results.html")

rag.close()
```

### Command Line Usage
```bash
# Process single diagram
python -m diagram_processing process data/examples/diagram.png diagram_001

# Process multiple diagrams
python -m diagram_processing batch data/examples/ data/processed/diagrams/

# GraphRAG workflow
python -m graph_rag process data/examples/diagram.png diagram_001
python -m graph_rag embeddings add diagram_001
python -m graph_rag search "find load balancers" diagram_001
```

## Important Design Patterns

### Diagram ID Pattern
All diagram processing uses unique diagram IDs for data partitioning in Neo4j. This enables:
- Multiple diagrams in a single database
- Context linking between text and diagrams
- Efficient retrieval of specific diagram data

### Error Resilience
The pipeline includes multi-tiered error handling:
- Robust JSON parsing for LLM responses
- Fallback extraction strategies
- Comprehensive logging throughout phases

### Neo4j Storage Pattern
```cypher
// All nodes and relationships tagged with diagram_id
(n:Process {name: 'Submit Request', diagram_id: 'diagram_001'})
(d:Decision {name: 'Approved?', diagram_id: 'diagram_001'})
(n)-[:FLOWS_TO {diagram_id: 'diagram_001'}]->(d)

// Efficient diagram retrieval
MATCH (n {diagram_id: $diagram_id})
OPTIONAL MATCH (n)-[r {diagram_id: $diagram_id}]->(m)
RETURN n, r, m
```

## Architecture Documentation

This repository contains comprehensive design documents for building RAG systems:

- `docs/architecture/ai-architecture.md`: Detailed hybrid RAG architecture for chatbot systems using the diagram-to-graph pipeline
- `docs/architecture/context-manager.md`: Implementation guide for stateful RAG systems with session management, conversation history, and user preferences

These documents provide production-ready architectural patterns for integrating the diagram-to-graph pipeline into larger AI systems.

## Dependencies

Key external dependencies:
- **Gemini 2.5 Pro API** (relationship generation and visual reasoning)
- **Neo4j** (graph database storage and retrieval)
- **Google Cloud Vision API** (OCR and text extraction - optional)
- **OpenCV** (shape detection and image processing)
- **Sentence Transformers** (embedding generation for semantic search)
- **Standard Python data science stack** (pandas, numpy, Pillow, matplotlib)
- **Visualization tools** (networkx, pyvis, graphviz for graph visualization)

## Current Repository Structure

The latest repository includes these main components:
- `diagram_processing/` - Core 3-phase pipeline with modular architecture
- `graph_rag/` - Advanced retrieval system with embeddings and two-phase search
- `interfaces/` - Unified CLI system for all operations
- `embeddings/` - Embedding management and encoding
- `graph_database/` - Database connection and query execution
- `context_manager/` - Session management and conversation context with modular architecture:
  - `core/` - Core processing (PromptBuilder, QueryRewriter, RetrievalFilter)
  - `storage/` - Storage backends (in-memory and external database support)
  - `utils/` - Analytics, maintenance, and migration tools
- `visualization/` - Graph visualization tools
- Architecture documentation files for production RAG systems