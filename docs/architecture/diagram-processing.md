# NetBot V2 - Diagram Processing Module

A robust, modular pipeline to transform network diagrams and flowcharts into knowledge graphs using OCR, Gemini 2.5 Pro, and Neo4j storage. This module is part of the larger NetBot V2 ecosystem.

## 📁 Project Structure

```
netbot-v2/
├── diagram_processing/         # Core image-to-graph pipeline
│   ├── __init__.py            # Module exports
│   ├── __main__.py            # Module CLI entry point
│   ├── cli.py                 # Specialized processing CLI
│   ├── client.py              # DiagramProcessor client
│   ├── core/                  # Core pipeline components
│   │   ├── __init__.py        # Core exports
│   │   ├── preprocessor.py    # Phase 1: Image preprocessing & OCR
│   │   ├── generator.py       # Phase 2: Gemini relationship generation
│   │   ├── exporter.py        # Phase 3: CSV & Neo4j export
│   │   └── pipeline.py        # Main pipeline orchestrator
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py        # Utilities exports
│   │   ├── ocr.py             # Google Cloud Vision OCR
│   │   ├── json_utils.py      # Robust LLM JSON parsing utilities
│   │   ├── text_processing.py # Text categorization utilities
│   │   └── shape_detection.py # OpenCV shape detection
│   ├── tools/                 # Standalone tools
│   │   ├── __init__.py        # Tools exports
│   │   └── ocr_tool.py        # Standalone OCR tool (legacy)
│   └── credentials/           # Google Cloud credentials (gitignored)
│
├── models/                    # Shared data models
│   ├── __init__.py            # Models exports
│   └── graph_models.py        # GraphNode, GraphRelationship, and Shape classes
│
├── cli.py                     # Main orchestration CLI
├── client.py                  # Root NetBot orchestration client
└── requirements.txt           # Dependencies
```

## ✨ Key Features

- **🧠 Advanced LLM Integration**: Gemini 2.5 Pro with robust JSON parsing
- **🔧 Modular Architecture**: Clean separation of concerns with reusable utilities
- **🛡️ Error Resilience**: Multi-tiered JSON parsing with fallback strategies
- **📊 Multiple Output Formats**: CSV files, Neo4j database, and structured JSON
- **🔍 Comprehensive OCR**: Google Cloud Vision with shape detection
- **🎯 Production Ready**: Proper error handling and logging throughout

## 🚀 Quick Start

```bash
# 1. Install dependencies (from project root)
pip install -r requirements.txt

# 2. Set up environment variables (create .env file in project root)
export GEMINI_API_KEY="your-gemini-api-key"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/google-credentials.json"  # Optional

# 3. Run via main orchestration CLI
python cli.py process-diagram data/examples/network_diagram.png

# 4. Or run via specialized module CLI
python -m diagram_processing process data/examples/network_diagram.png diagram_001
```

## 📋 3-Phase Pipeline

### Phase 1: Image Preprocessing & OCR (`core/preprocessor.py`)
- **Google Cloud Vision API** for accurate text extraction (`utils/ocr.py`)
- **OpenCV** for shape detection (rectangles, circles, lines) (`utils/shape_detection.py`)
- **Automatic diagram type detection** (network vs flowchart vs mixed)
- **Spatial analysis** with bounding boxes and coordinates

### Phase 2: Gemini 2.5 Pro Relationship Generation (`core/generator.py`)
- **Advanced prompts** for network diagrams and flowcharts
- **Visual reasoning** to understand spatial relationships
- **Structured graph extraction** of nodes and relationships with confidence scores
- **Robust JSON parsing** with multi-tiered error handling and fallback strategies (`utils/json_utils.py`)
- **Clean data output** separating nodes and relationships for maximum data purity

### Phase 3: Export & Storage (`core/exporter.py`)
- **CSV generation** (`nodes.csv` and `relationships.csv`)
- **Neo4j database storage** with typed nodes and relationships
- **Schema validation** and node type classification
- **Full pipeline results** saved as comprehensive JSON metadata

## 🛠️ Setup Instructions

(Setup instructions for Google Cloud, Gemini, and Neo4j remain the same)

## 📊 Usage Examples

### Main CLI (Recommended)
```bash
# Complete automated workflow
python cli.py quickstart data/examples/network_diagram.png diagram_001 "find load balancers"

# Process diagram and store in Neo4j
python cli.py process-diagram data/examples/network_diagram.png

# Advanced workflow with more control
python cli.py process-and-search data/examples/network_diagram.png diagram_001 "find servers" --visualize --explain
```

### Specialized Module CLI
```bash
# Process single diagram
python -m diagram_processing process data/examples/diagram.png diagram_001

# Using client directly
python diagram_processing/cli.py process data/examples/diagram.png diagram_001
```

### Programmatic Usage (via NetBot client)
```python
from client import NetBot

# Initialize NetBot (includes diagram processing)
netbot = NetBot()

# Process diagram
result = netbot.process_diagram(
    image_path="data/examples/network_diagram.png",
    output_dir="data/processed",
    force_reprocess=False
)

if result['status'] == 'success':
    print(f"Generated {len(result['nodes'])} nodes and {len(result['relationships'])} relationships")
    print(f"Neo4j stored: {result['neo4j_stored']}")

# Close connections
netbot.close()
```

### Direct DiagramProcessor Usage
```python
from diagram_processing.client import DiagramProcessor

# Initialize processor
processor = DiagramProcessor(gemini_api_key="your-key")

# Process image with diagram ID
result = processor.process(
    image_path="data/examples/network_diagram.png",
    diagram_id="diagram_001",
    output_dir="data/processed/diagrams"
)

if result['status'] == 'success':
    # Access nodes and relationships directly
    for node in result.get('nodes', []):
        print(f"Node: {node.label} ({node.type})")
    
    for rel in result.get('relationships', []):
        print(f"Relationship: {rel.source_id} -[{rel.type}]-> {rel.target_id}")

processor.close()
```

## 📁 Output Structure

```
data/processed/
├── [image_name]/                    # Directory named after input image
│   ├── nodes.csv                    # All unique nodes with their properties
│   ├── relationships.csv            # All relationships with their properties
│   └── pipeline_metadata.json      # Complete pipeline results and metadata
├── diagrams/                        # Alternative output location for some workflows
│   └── [image_name]/
│       ├── nodes.csv
│       ├── relationships.csv
│       └── pipeline_metadata.json
└── visualizations/                  # Generated visualization files
    └── [diagram_id]_subgraph_[hash]_[timestamp].png
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Phase 1: OCR   │───▶│ Phase 2: Gemini  │───▶│ Phase 3: Export │
│                 │    │                  │    │                 │
│ • Vision API    │    │ • Relationship Gen│    │ • CSV Files     │
│ • Shape Detect  │    │ • Prompt Eng     │    │ • Neo4j Store   │
│ • Text Categorize│    │ • Visual Reason  │    │ • Validation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 Modular Utilities

### JSON Parsing (`utils/json_utils.py`)
The pipeline includes robust JSON parsing utilities designed specifically for handling imperfect LLM responses:

```python
from diagram_processing.utils.json_utils import LLMJsonParser

# Parse any LLM JSON response with automatic error recovery
data = LLMJsonParser.parse_llm_response(gemini_response)
```

**Features:**
- **Multi-tiered parsing**: Raw JSON → Gentle cleaning → Aggressive fixes
- **String-aware cleaning**: Preserves URLs and content inside JSON strings
- **Control character handling**: Removes invalid characters that break parsing
- **Automatic structure repair**: Fixes missing braces, trailing commas, etc.
- **Fallback strategies**: Regex extraction when JSON parsing completely fails

This makes the pipeline extremely resilient to the common JSON formatting issues that occur with LLM responses.

### Data Models (`models/graph_models.py`)
The pipeline uses standardized data classes for all graph components (shared across the NetBot ecosystem):

```python
from models.graph_models import GraphNode, GraphRelationship, Shape

# Standard node representation
node = GraphNode(id="node_1", label="Load Balancer", type="Network", properties={...})

# Standard relationship representation  
rel = GraphRelationship(id="rel_1", source_id="node_1", target_id="node_2", type="CONNECTS_TO")
```

## 🚨 Troubleshooting

**"No graph data generated"**
- Check image quality and text clarity
- Verify diagram contains recognizable elements  
- Review the Gemini response in `../data/processed/diagrams/[image_name]/pipeline_metadata.json`
- Ensure GEMINI_API_KEY is properly set

**"JSON parsing errors"**
- The pipeline automatically handles most JSON parsing issues
- Check the console output for detailed parsing error information
- If issues persist, the fallback extraction will attempt to recover data
- Consider adjusting the Gemini prompt if responses are consistently malformed

**"Import errors"**
- Ensure you're running from the project root directory
- Input images should be in data/examples/
- Output will be in data/processed/ (consolidated with other outputs)
- Install dependencies from project root: `pip install -r requirements.txt`
- Check Python path and virtual environment setup
- Use `python -m diagram_processing` for module-based execution

**"Neo4j connection issues"**
- Verify Neo4j is running and accessible
- Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD environment variables
- Use `--no-neo4j` flag to skip Neo4j storage for testing
