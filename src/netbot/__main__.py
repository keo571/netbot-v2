"""
Main entry point for NetBot V2.

Provides CLI commands and API server startup.
"""

import sys
import uvicorn
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from netbot.api import create_app


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        # Start API server
        app = create_app()
        
        # Development server settings
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    else:
        print("NetBot V2 - AI-Powered Network Diagram Analysis")
        print("")
        print("Usage:")
        print("  python -m netbot serve    # Start API server")
        print("  python -m netbot --help   # Show help")
        print("")
        print("API Documentation:")
        print("  http://localhost:8000/docs    # Swagger UI")
        print("  http://localhost:8000/redoc   # ReDoc")


if __name__ == "__main__":
    main()