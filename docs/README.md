# NetBot-v2 Documentation

Welcome to the NetBot-v2 documentation! This comprehensive guide covers all aspects of the AI-powered network diagram analysis and GraphRAG system.

## 📚 Documentation Structure

### 🏗️ [Architecture](./architecture/)
Comprehensive design documents and architectural patterns:
- **[AI Architecture](./architecture/ai-architecture.md)** - Detailed hybrid RAG architecture for chatbot systems using the diagram-to-graph pipeline
- **[Context Manager](./architecture/context-manager.md)** - Implementation guide for stateful RAG systems with session management, conversation history, and user preferences
- **[Diagram Processing](./architecture/diagram-processing.md)** - Core pipeline architecture for converting images to knowledge graphs

### 📖 [API Reference](./api/)
*Coming soon* - Detailed API documentation for all modules and classes

### 🚀 [User Guides](./guides/)
*Coming soon* - Step-by-step guides for common workflows:
- Getting started with diagram processing
- Setting up the GraphRAG system
- Advanced querying techniques
- Visualization and reporting

### 💡 [Examples](./examples/)
Practical examples and use cases:
- **[Context Manager Usage](./examples/context-manager-usage.py)** - Complete example of setting up stateful RAG system
- Processing network diagrams *(coming soon)*
- Building chatbots with GraphRAG *(coming soon)*
- Custom integrations *(coming soon)*

## 🔗 Quick Links

### Main Documentation
- **[Project README](../README_NEW_STRUCTURE.md)** - Project overview and setup
- **[CLAUDE.md](../CLAUDE.md)** - Development guidance for Claude Code

### Module Documentation
- **[Diagram Processing](./architecture/diagram-processing.md)** - Core pipeline documentation
- **[Graph RAG Module](../graph_rag/)** - Advanced retrieval system

## 🛠️ Development

### Installation
```bash
# Install the package
pip install -e .

# Install with optional dependencies
pip install -e ".[dev,vision,visualization,storage,nlp]"
```

### Usage
```bash
# Process diagrams
python -m diagram_processing process data/examples/diagram.png diagram_001

# Query with GraphRAG
python -m graph_rag search "find load balancers" diagram_001
```

### Programmatic Usage
```python
from diagram_processing import DiagramProcessor
from graph_rag import GraphRAG

# Process diagram
processor = DiagramProcessor()
result = processor.process("data/examples/diagram.png", "diagram_001")

# Query with GraphRAG
rag = GraphRAG()
results = rag.search("find database connections", "diagram_001")
```

## 🤝 Contributing

We welcome contributions! Please see our development guidelines:
- Follow Python best practices
- Update documentation for new features
- Include tests for new functionality
- Follow the existing code style

## 📞 Support

For questions and support:
- Check the documentation first
- Review the architecture documents for design patterns
- Look at the examples for common use cases

---

*This documentation is organized to help you quickly find the information you need, whether you're getting started, diving deep into the architecture, or looking for specific examples.*