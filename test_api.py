#!/usr/bin/env python3
"""
Test script for NetBot-v2 API endpoints
"""

import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test basic API health"""
    print("🔍 Testing API health...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.ConnectionError:
        print("❌ Cannot connect to API. Is the server running?")
        return False

def test_list_diagrams():
    """Test listing available diagrams"""
    print("\n📋 Testing diagram listing...")
    try:
        response = requests.get(f"{BASE_URL}/diagrams")
        if response.status_code == 200:
            diagrams = response.json()
            print(f"✅ Found {len(diagrams)} diagrams")
            for diagram in diagrams[:3]:  # Show first 3
                print(f"  • {diagram['diagram_id']} ({diagram['node_count']} nodes)")
            return diagrams
        else:
            print(f"❌ Failed to list diagrams: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error listing diagrams: {e}")
        return []

def test_admin_login():
    """Test admin authentication"""
    print("\n🔐 Testing admin login...")
    try:
        # Use default admin key from .env.example
        login_data = {"api_key": "your-admin-api-key-change-in-production"}
        
        response = requests.post(f"{BASE_URL}/admin/login", json=login_data)
        if response.status_code == 200:
            token_data = response.json()
            print("✅ Admin login successful")
            return token_data["access_token"]
        else:
            print(f"❌ Admin login failed: {response.status_code}")
            print("Make sure ADMIN_API_KEY is set in your .env file")
            return None
    except Exception as e:
        print(f"❌ Admin login error: {e}")
        return None

def test_chat_search(diagrams):
    """Test chat search functionality"""
    if not diagrams:
        print("\n⚠️  No diagrams available for search test")
        return
    
    print("\n💬 Testing chat search...")
    diagram_id = diagrams[0]['diagram_id']
    
    try:
        chat_data = {
            "message": "find servers",
            "diagram_id": diagram_id
        }
        
        response = requests.post(f"{BASE_URL}/chat", json=chat_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Search successful: {result['response']}")
            if result.get('results'):
                node_count = len(result['results'].get('nodes', []))
                print(f"  Found {node_count} matching nodes")
        else:
            print(f"❌ Chat search failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Chat search error: {e}")

def test_admin_upload(token):
    """Test admin file upload (if token available)"""
    if not token:
        print("\n⚠️  Skipping upload test (no admin token)")
        return
    
    print("\n📤 Testing admin upload...")
    
    # Check for example images
    example_images = list(Path("data/examples").glob("*.png")) if Path("data/examples").exists() else []
    if not example_images:
        print("⚠️  No example images found for upload test")
        return
    
    try:
        image_path = example_images[0]
        headers = {"Authorization": f"Bearer {token}"}
        
        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/png")}
            response = requests.post(f"{BASE_URL}/admin/upload-diagram", 
                                   files=files, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful: {result['message']}")
            print(f"  Diagram ID: {result['diagram_id']}")
            print(f"  Nodes: {result['node_count']}")
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Upload test error: {e}")

def main():
    """Run all API tests"""
    print("🧪 NetBot-v2 API Test Suite")
    print("=" * 40)
    
    # Test 1: Health check
    if not test_health_check():
        print("\n❌ API server is not running. Start it with: python run_api.py")
        return 1
    
    # Test 2: List diagrams
    diagrams = test_list_diagrams()
    
    # Test 3: Admin login
    admin_token = test_admin_login()
    
    # Test 4: Chat search
    test_chat_search(diagrams)
    
    # Test 5: Admin upload
    test_admin_upload(admin_token)
    
    print("\n🎉 API test suite completed!")
    print("\n💡 To interact with the API:")
    print(f"  • View docs: {BASE_URL}/docs")
    print(f"  • Test frontend: Connect React app to {BASE_URL}")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())