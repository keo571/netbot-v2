#!/usr/bin/env python3
"""
Standalone OCR Tool - Extract text from diagrams without full pipeline

Usage:
    python ocr_tool.py network_diagram.png
    python ocr_tool.py diagram.png results.json
"""

import os
import sys
import json
from ..utils.ocr import NetworkGraphOCR


def save_results(results: dict, output_file: str = 'ocr_results.json'):
    """Save OCR results to JSON file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def main():
    """Main function to run OCR on command line"""
    if len(sys.argv) < 2:
        print("Usage: python ocr_tool.py <image_path> [output_file]")
        print("Example: python ocr_tool.py network_diagram.png results.json")
        return
    
    image_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'ocr_results.json'
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image file not found: {image_path}")
        return
    
    print(f"🔍 Processing image: {image_path}")
    
    try:
        # Initialize OCR
        ocr = NetworkGraphOCR()
        
        # Extract text
        results = ocr.extract_diagram_elements(image_path)
        
        if results['status'] == 'success':
            print(f"✅ Text extraction completed!")
            print(f"🎯 Detected diagram type: {results['diagram_type'].upper()}")
            print(f"📊 Found {results['total_elements']} text elements")
            
            # Display categorized results based on diagram type
            if results['diagram_type'] == 'mixed':
                # For mixed diagrams, show both categorizations
                print(f"\n🔸 Network Elements:")
                for category, items in results['categorized_elements']['network_elements'].items():
                    if items:
                        print(f"  📋 {category.replace('_', ' ').title()}: {len(items)} items")
                        for item in items[:2]:  # Show first 2 items
                            print(f"    - {item['text']}")
                        if len(items) > 2:
                            print(f"    ... and {len(items) - 2} more")
                
                print(f"\n🔸 Flowchart Elements:")
                for category, items in results['categorized_elements']['flowchart_elements'].items():
                    if items:
                        print(f"  📋 {category.replace('_', ' ').title()}: {len(items)} items")
                        for item in items[:2]:  # Show first 2 items
                            print(f"    - {item['text']}")
                        if len(items) > 2:
                            print(f"    ... and {len(items) - 2} more")
            else:
                # For single type diagrams
                for category, items in results['categorized_elements'].items():
                    if items:
                        print(f"\n📋 {category.replace('_', ' ').title()}: {len(items)} items")
                        for item in items[:3]:  # Show first 3 items
                            print(f"  - {item['text']}")
                        if len(items) > 3:
                            print(f"  ... and {len(items) - 3} more")
            
            # Save results
            save_results(results, output_file)
            
        else:
            print(f"❌ Error: {results.get('message', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main() 