"""
Centralized model client for AI operations.

Manages connections to various AI models (Gemini, etc.) with
caching, rate limiting, and error handling.
"""

import threading
import time
from typing import Dict, Any, Optional, List, Union
from functools import lru_cache

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from ...config.settings import get_settings
from ...exceptions import AIError, ConfigurationError
from ..cache.cache_manager import get_cache_manager


class ModelClient:
    """
    Centralized client for AI model operations.
    
    Provides unified interface for different AI models with built-in
    caching, error handling, and performance monitoring.
    """
    
    _instance: Optional["ModelClient"] = None
    _lock = threading.RLock()
    _initialized = False
    
    def __new__(cls) -> "ModelClient":
        """Singleton pattern for model client."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize model client."""
        if self._initialized:
            return
        
        self.settings = get_settings()
        self.cache = get_cache_manager()
        self._models: Dict[str, Any] = {}
        self._request_counts: Dict[str, int] = {}
        self._last_request_times: Dict[str, float] = {}
        
        # Initialize Gemini if API key available
        if self.settings.gemini_api_key:
            self._init_gemini()
        
        self._initialized = True
    
    def _init_gemini(self):
        """Initialize Gemini AI client."""
        try:
            genai.configure(api_key=self.settings.gemini_api_key)
            
            # Test connection
            models = genai.list_models()
            available_models = [m.name for m in models]
            print(f"✅ Gemini API connected. Available models: {len(available_models)}")
            
            # Cache the client
            self.cache.cache_model("gemini_client", genai)
            
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Gemini: {e}")
    
    def get_gemini_model(self, model_name: str = "gemini-2.0-flash-exp") -> Any:
        """
        Get or create Gemini model instance.
        
        Args:
            model_name: Gemini model name
            
        Returns:
            Gemini model instance
        """
        cache_key = f"gemini_model_{model_name}"
        cached_model = self.cache.get_model(cache_key)
        
        if cached_model is not None:
            return cached_model
        
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            # Cache the model
            self.cache.cache_model(cache_key, model)
            print(f"✅ Cached Gemini model: {model_name}")
            
            return model
            
        except Exception as e:
            raise AIError(f"Failed to create Gemini model {model_name}: {e}")
    
    def generate_text(self, 
                     prompt: str, 
                     model_name: str = "gemini-2.0-flash-exp",
                     max_tokens: int = 8192,
                     temperature: float = 0.1,
                     cache_response: bool = True) -> str:
        """
        Generate text using specified model.
        
        Args:
            prompt: Text prompt
            model_name: Model to use
            max_tokens: Maximum response tokens
            temperature: Generation temperature
            cache_response: Whether to cache the response
            
        Returns:
            Generated text
        """
        # Check cache first if enabled
        if cache_response:
            cache_key = f"text_gen_{hash(prompt)}_{model_name}_{temperature}"
            cached_response = self.cache.get(cache_key, namespace='ai_responses')
            if cached_response:
                return cached_response
        
        # Rate limiting check
        self._check_rate_limit(model_name)
        
        try:
            model = self.get_gemini_model(model_name)
            
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            )
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if not response.text:
                raise AIError("Empty response from model")
            
            result = response.text.strip()
            
            # Cache response if enabled
            if cache_response:
                cache_key = f"text_gen_{hash(prompt)}_{model_name}_{temperature}"
                self.cache.set(cache_key, result, namespace='ai_responses', ttl_seconds=3600)
            
            # Update request tracking
            self._update_request_stats(model_name)
            
            return result
            
        except Exception as e:
            raise AIError(f"Text generation failed: {e}")
    
    def generate_json(self,
                     prompt: str,
                     model_name: str = "gemini-2.0-flash-exp",
                     max_tokens: int = 8192,
                     temperature: float = 0.1) -> Dict[str, Any]:
        """
        Generate JSON response using specified model.
        
        Args:
            prompt: Text prompt (should request JSON format)
            model_name: Model to use
            max_tokens: Maximum response tokens
            temperature: Generation temperature
            
        Returns:
            Parsed JSON response
        """
        # Ensure prompt asks for JSON
        if "JSON" not in prompt.upper():
            prompt += "\n\nPlease respond with valid JSON only."
        
        try:
            model = self.get_gemini_model(model_name)
            
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                response_mime_type="application/json"
            )
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if not response.text:
                raise AIError("Empty response from model")
            
            import json
            try:
                result = json.loads(response.text)
            except json.JSONDecodeError as e:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise AIError(f"Invalid JSON response: {e}")
            
            self._update_request_stats(model_name)
            return result
            
        except Exception as e:
            raise AIError(f"JSON generation failed: {e}")
    
    def analyze_image(self,
                     image_data: bytes,
                     prompt: str,
                     model_name: str = "gemini-2.0-flash-exp") -> str:
        """
        Analyze image with text prompt.
        
        Args:
            image_data: Binary image data
            prompt: Analysis prompt
            model_name: Model to use
            
        Returns:
            Analysis result
        """
        try:
            import PIL.Image
            import io
            
            # Convert bytes to PIL Image
            image = PIL.Image.open(io.BytesIO(image_data))
            
            model = self.get_gemini_model(model_name)
            response = model.generate_content([prompt, image])
            
            if not response.text:
                raise AIError("Empty response from model")
            
            self._update_request_stats(model_name)
            return response.text.strip()
            
        except Exception as e:
            raise AIError(f"Image analysis failed: {e}")
    
    def _check_rate_limit(self, model_name: str):
        """Check rate limiting for model."""
        current_time = time.time()
        last_request = self._last_request_times.get(model_name, 0)
        
        # Simple rate limiting - minimum 1 second between requests
        min_interval = 1.0
        if current_time - last_request < min_interval:
            time.sleep(min_interval - (current_time - last_request))
    
    def _update_request_stats(self, model_name: str):
        """Update request statistics."""
        self._request_counts[model_name] = self._request_counts.get(model_name, 0) + 1
        self._last_request_times[model_name] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            'models_cached': len([k for k in self.cache.get_stats()['memory_backend'] if 'model' in str(k)]),
            'request_counts': self._request_counts.copy(),
            'last_request_times': {
                k: time.time() - v for k, v in self._last_request_times.items()
            }
        }


# Global instance
_model_client = None
_client_lock = threading.Lock()


@lru_cache()
def get_model_client() -> ModelClient:
    """
    Get the global model client instance.
    
    Returns:
        ModelClient singleton instance
    """
    global _model_client
    
    if _model_client is None:
        with _client_lock:
            if _model_client is None:
                _model_client = ModelClient()
    
    return _model_client