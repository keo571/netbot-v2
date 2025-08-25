#!/usr/bin/env python3
"""
Simple script to run the NetBot-v2 API server
"""

import os
import subprocess
import sys
from pathlib import Path

def check_dependencies():
    """Check if API dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import jwt
        print("✅ API dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install with: pip install -r requirements-api.txt")
        return False

def check_env_file():
    """Check if .env file exists"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("Copy .env.example to .env and configure your API keys:")
        print("  cp .env.example .env")
        return False
    
    # Check for required variables
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['GEMINI_API_KEY', 'NEO4J_PASSWORD']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please configure them in your .env file")
        return False
    
    print("✅ Environment configuration found")
    return True

def main():
    print("🚀 NetBot-v2 API Server")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check environment
    if not check_env_file():
        return 1
    
    # Start server
    print("\n🌟 Starting API server...")
    print("📖 API docs: http://localhost:8000/docs")
    print("🔍 Public endpoints: /chat, /diagrams")
    print("🔒 Admin endpoints: /admin/* (requires authentication)")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        os.system("python api_server.py")
    except KeyboardInterrupt:
        print("\n👋 API server stopped")
        return 0

if __name__ == "__main__":
    sys.exit(main())