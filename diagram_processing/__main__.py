#!/usr/bin/env python3
"""
Enable running diagram_processing as a module: python -m diagram_processing

Usage:
    python -m diagram_processing --help
    python -m diagram_processing process image.png diagram_001
"""

from .cli import main

if __name__ == "__main__":
    main()