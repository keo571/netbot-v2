"""
Google Cloud Vision OCR functionality for diagram text extraction.
"""

import json
from google.cloud import vision

from ..utils.text_processing import TextCategorizer


class NetworkGraphOCR:
    """Google Cloud Vision OCR with intelligent text categorization"""
    
    def __init__(self):
        """Initialize Google Cloud Vision client"""
        try:
            self.client = vision.ImageAnnotatorClient()
            self.text_categorizer = TextCategorizer()
            self._initialized = True
        except Exception as e:
            print(f"❌ Error initializing Google Cloud Vision: {e}")
            print("Make sure you have set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            raise
    
    def extract_text_from_image(self, image_path: str):
        """
        Extract text from image using Google Cloud Vision
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            dict: Extracted text with bounding box coordinates and confidence scores
        """
        # Print initialization message on first OCR call
        if hasattr(self, '_initialized') and self._initialized:
            print("Google Cloud Vision client initialized successfully")
            self._initialized = False  # Only show once
            
        try:
            # Read the image file
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            # Create image object
            image = vision.Image(content=content)
            
            # Perform text detection
            response = self.client.text_detection(image=image)
            texts = response.text_annotations
            
            if response.error.message:
                raise Exception(f'{response.error.message}')
            
            if not texts:
                return {'status': 'no_text', 'texts': []}
            
            # Process results
            result = {
                'status': 'success',
                'full_text': texts[0].description if texts else '',
                'individual_texts': []
            }
            
            # Skip the first element (full text) and process individual text elements
            for text in texts[1:]:
                vertices = text.bounding_poly.vertices
                bbox = {
                    'x1': vertices[0].x,
                    'y1': vertices[0].y,
                    'x2': vertices[2].x,
                    'y2': vertices[2].y
                }
                
                text_info = {
                    'text': text.description,
                    'bbox': bbox
                }
                
                result['individual_texts'].append(text_info)
            
            return result
            
        except FileNotFoundError:
            return {'status': 'error', 'message': f'Image file not found: {image_path}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def extract_diagram_elements(self, image_path: str):
        """
        Extract and categorize text elements for both network graphs and flowcharts
        
        Returns:
            dict: Categorized diagram elements with detected type
        """
        result = self.extract_text_from_image(image_path)
        
        if result['status'] != 'success':
            return result
        
        # Detect diagram type and categorize elements
        diagram_type = self.text_categorizer.detect_diagram_type(result['individual_texts'])
        categorized_elements = self.text_categorizer.categorize_elements(
            result['individual_texts'], 
            diagram_type
        )
        
        return {
            'status': 'success',
            'diagram_type': diagram_type,
            'full_text': result['full_text'],
            'categorized_elements': categorized_elements,
            'total_elements': len(result['individual_texts'])
        }
 