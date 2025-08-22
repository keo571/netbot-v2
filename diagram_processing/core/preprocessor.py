"""
Phase 1: Image preprocessing and OCR text extraction.
"""

from typing import Dict
from ..utils.ocr import NetworkGraphOCR
from ..utils.shape_detection import ShapeDetector


class ImagePreprocessor:
    """Phase 1: Enhanced OCR with shape detection"""
    
    def __init__(self):
        self.ocr = NetworkGraphOCR()
        self.shape_detector = ShapeDetector()
    
    def process_image(self, image_path: str) -> Dict:
        """
        Complete Phase 1 processing: OCR + Shape Detection
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict: Combined OCR and shape detection results
        """
        print("Phase 1: Image preprocessing & OCR...")
        
        # OCR text extraction
        ocr_result = self.ocr.extract_diagram_elements(image_path)
        
        if ocr_result['status'] != 'success':
            return ocr_result
        
        # Shape detection
        shapes = self.shape_detector.detect_shapes(image_path)
        
        return {
            'status': 'success',
            'diagram_type': ocr_result['diagram_type'],
            'text_elements': ocr_result['categorized_elements'],
            'shapes': shapes,
            'total_text_elements': ocr_result['total_elements'],
            'total_shapes': len(shapes),
            'full_text': ocr_result['full_text']
        } 