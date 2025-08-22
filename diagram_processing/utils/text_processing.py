"""
Text processing and categorization utilities.
"""

import re
from typing import List, Dict, Any


class TextCategorizer:
    """Categorizes extracted text based on diagram type and content patterns"""
    
    def detect_diagram_type(self, text_elements: List[Dict[str, Any]]) -> str:
        """
        Detect whether the diagram is a network graph or flowchart
        
        Args:
            text_elements (list): List of extracted text elements
        
        Returns:
            str: 'network', 'flowchart', or 'mixed'
        """
        network_indicators = 0
        flowchart_indicators = 0
        
        # Common network terms
        network_terms = [
            r'\b(router|switch|firewall|gateway|hub|bridge|access point|ap)\b',
            r'\b(eth|gi|fa|se|lo|en|wlan)\d+',
            r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',  # IP addresses
            r'\b(vlan|subnet|network|interface|port)\b',
            r'\b(cisco|juniper|netgear|linksys)\b'
        ]
        
        # Common flowchart terms
        flowchart_terms = [
            r'\b(start|end|begin|finish|stop)\b',
            r'\b(yes|no|true|false)\b',
            r'\b(if|then|else|while|for|do)\b',
            r'\b(process|decision|input|output)\b',
            r'\b(step \d+|phase \d+)\b',
            r'\?\s*$',  # Questions ending with ?
            r'^(check|verify|validate|confirm)',
            r'\b(approve|reject|submit|cancel)\b'
        ]
        
        all_text = ' '.join([item['text'].lower() for item in text_elements])
        
        # Count indicators
        for pattern in network_terms:
            if re.search(pattern, all_text, re.IGNORECASE):
                network_indicators += len(re.findall(pattern, all_text, re.IGNORECASE))
        
        for pattern in flowchart_terms:
            if re.search(pattern, all_text, re.IGNORECASE):
                flowchart_indicators += len(re.findall(pattern, all_text, re.IGNORECASE))
        
        # Determine type
        if network_indicators > flowchart_indicators * 1.5:
            return 'network'
        elif flowchart_indicators > network_indicators * 1.5:
            return 'flowchart'
        else:
            return 'mixed'
    
    def categorize_elements(self, text_elements: List[Dict[str, Any]], diagram_type: str) -> Dict:
        """
        Categorize text elements based on diagram type
        
        Args:
            text_elements: List of text elements with metadata
            diagram_type: Type of diagram ('network', 'flowchart', 'mixed')
            
        Returns:
            dict: Categorized elements
        """
        if diagram_type == 'network':
            return self.categorize_network_elements(text_elements)
        elif diagram_type == 'flowchart':
            return self.categorize_flowchart_elements(text_elements)
        else:  # mixed
            return {
                'network_elements': self.categorize_network_elements(text_elements),
                'flowchart_elements': self.categorize_flowchart_elements(text_elements)
            }
    
    def categorize_network_elements(self, text_elements: List[Dict[str, Any]]) -> Dict:
        """Categorize elements for network diagrams"""
        categories = {
            'node_labels': [],
            'ip_addresses': [],
            'domain_names': [],
            'interface_names': [],
            'network_terms': [],
            'other_text': []
        }
        
        for text_item in text_elements:
            text = text_item['text'].strip()
            
            # IP address pattern
            ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            
            # Domain name pattern
            domain_pattern = r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)*\b'
            
            # Interface pattern (eth0, gi0/1, etc.)
            interface_pattern = r'\b(eth|gi|fa|se|lo|en|wlan)\d+(/\d+)*\b'
            
            # Network device terms
            network_device_pattern = r'\b(router|switch|firewall|gateway|hub|bridge|access point|ap|server|client)\b'
            
            if re.match(ip_pattern, text):
                categories['ip_addresses'].append(text_item)
            elif re.match(interface_pattern, text, re.IGNORECASE):
                categories['interface_names'].append(text_item)
            elif re.match(domain_pattern, text) and '.' in text:
                categories['domain_names'].append(text_item)
            elif re.search(network_device_pattern, text, re.IGNORECASE):
                categories['network_terms'].append(text_item)
            elif len(text) > 1:  # Likely node labels
                categories['node_labels'].append(text_item)
            else:
                categories['other_text'].append(text_item)
        
        return categories
    
    def categorize_flowchart_elements(self, text_elements: List[Dict[str, Any]]) -> Dict:
        """Categorize elements for flowcharts"""
        categories = {
            'start_end': [],
            'processes': [],
            'decisions': [],
            'inputs_outputs': [],
            'conditions': [],
            'connectors': [],
            'other_text': []
        }
        
        for text_item in text_elements:
            text = text_item['text'].strip()
            text_lower = text.lower()
            
            # Start/End indicators
            if re.search(r'\b(start|end|begin|finish|stop)\b', text_lower):
                categories['start_end'].append(text_item)
            
            # Decision indicators (questions, yes/no)
            elif (text.endswith('?') or 
                  re.search(r'\b(yes|no|true|false)\b', text_lower) or
                  re.search(r'\b(if|check|verify|validate|confirm)\b', text_lower)):
                categories['decisions'].append(text_item)
            
            # Input/Output indicators
            elif re.search(r'\b(input|output|enter|display|print|read|write|save)\b', text_lower):
                categories['inputs_outputs'].append(text_item)
            
            # Process indicators (action verbs)
            elif (re.search(r'\b(process|calculate|compute|execute|perform|run|do)\b', text_lower) or
                  re.search(r'^(create|update|delete|add|remove|modify)\b', text_lower)):
                categories['processes'].append(text_item)
            
            # Condition/connector text (short phrases)
            elif (len(text.split()) <= 3 and 
                  re.search(r'\b(then|else|next|goto|continue|repeat|loop)\b', text_lower)):
                categories['connectors'].append(text_item)
            
            # Generic conditions
            elif len(text.split()) <= 5:  # Short phrases likely conditions
                categories['conditions'].append(text_item)
            
            else:
                categories['other_text'].append(text_item)
        
        return categories
    
 