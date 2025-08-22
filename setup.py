"""
Setup script for netbot-v2: AI-powered network diagram analysis and GraphRAG system
"""

from setuptools import setup, find_packages

setup(
    name="netbot-v2",
    version="1.0.0",
    description="AI-powered network diagram processing and GraphRAG system",
    long_description="AI-powered network diagram processing and GraphRAG system for transforming network diagrams into knowledge graphs",
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "neo4j>=5.13.0",
        "google-generativeai>=0.8.5", 
        "sentence-transformers>=3.0.0",
        "python-dotenv>=1.0.0",
        "python-dateutil>=2.8.0",
        
        # Data processing
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        
        # Image processing
        "pillow>=9.0.0",
        "opencv-python>=4.8.0",
        
        # Visualization
        "networkx>=3.0",
        "matplotlib>=3.6.0",
        "pyvis>=0.3.2",
    ],
    extras_require={
        "vision": ["google-cloud-vision>=3.4.0"],
        "visualization": ["graphviz>=0.20.0"],
        "storage": [
            "redis>=4.0.0",
            "pymongo>=4.0.0", 
            "psycopg2-binary>=2.9.0"
        ],
        "nlp": [
            "regex>=2022.0.0",
            "nltk>=3.8.0"
        ],
        "vector": [
            "chromadb>=0.4.0"
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0", 
            "pytest-mock>=3.10.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0", 
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
            "ipython>=8.0.0"
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0"
        ],
        "all": [
            "google-cloud-vision>=3.4.0",
            "graphviz>=0.20.0",
            "redis>=4.0.0",
            "pymongo>=4.0.0", 
            "psycopg2-binary>=2.9.0",
            "regex>=2022.0.0",
            "nltk>=3.8.0",
            "chromadb>=0.4.0"
        ],
    },
    entry_points={
        "console_scripts": [
            "netbot=cli:main",
            "diagram-processing=diagram_processing.cli:main",
            "graph-rag=graph_rag.cli:main",
            "netbot-embeddings=embeddings.cli:main",
        ],
    },
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
    include_package_data=True,
    author="NetBot Team",
    author_email="team@netbot.ai",
    url="https://github.com/keo571/netbot-v2",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="ai graph-rag knowledge-graph diagram-processing network-analysis",
)