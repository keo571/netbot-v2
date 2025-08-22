"""
Shape detection utilities using OpenCV.
"""

import cv2
import numpy as np
from typing import List

from models.graph_models import Shape


class ShapeDetector:
    """Detects basic shapes (rectangles, circles, lines) in images using OpenCV"""
    
    def detect_shapes(self, image_path: str) -> List[Shape]:
        """
        Detect basic shapes using OpenCV
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of detected shapes
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return []
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            shapes = []
            
            # Detect rectangles/squares
            shapes.extend(self._detect_rectangles(gray))
            
            # Detect lines
            shapes.extend(self._detect_lines(gray))
            
            return shapes
            
        except Exception as e:
            print(f"❌ Error in shape detection: {e}")
            return []
    
    def _detect_rectangles(self, gray_image) -> List[Shape]:
        """Detect rectangular shapes"""
        shapes = []
        
        # Find contours
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip very small shapes
            if w < 20 or h < 20:
                continue
            
            # Classify shape based on number of vertices
            if len(approx) == 4:
                # Rectangle or square
                aspect_ratio = w / h
                shape_type = 'square' if 0.8 <= aspect_ratio <= 1.2 else 'rectangle'
                shapes.append(Shape(
                    type=shape_type,
                    bbox=(x, y, x + w, y + h),
                    properties={'width': str(w), 'height': str(h), 'aspect_ratio': str(aspect_ratio)}
                ))
            elif len(approx) > 6:
                # Likely a circle/ellipse
                shapes.append(Shape(
                    type='circle',
                    bbox=(x, y, x + w, y + h),
                    properties={'radius': str(min(w, h) // 2)}
                ))
        
        return shapes
    
    def _detect_lines(self, gray_image) -> List[Shape]:
        """Detect line shapes using Hough transform"""
        shapes = []
        
        # Detect edges
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        
        # Detect lines using HoughLines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            for line in lines[:50]:  # Limit to 50 lines
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                
                # Calculate line length and angle
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                angle_degrees = theta * 180 / np.pi
                
                shapes.append(Shape(
                    type='line',
                    bbox=(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
                    properties={
                        'angle': str(angle_degrees),
                        'length': str(int(length)),
                        'rho': str(rho)
                    }
                ))
        
        return shapes 