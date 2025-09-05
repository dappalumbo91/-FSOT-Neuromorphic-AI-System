#!/usr/bin/env python3
"""
ENHANCED FSOT 2.0 API MANAGER
============================

Integrated API access system for enhanced neuromorphic capabilities.
Supports OpenAI, GitHub, Wolfram Alpha, and HuggingFace APIs with
intelligent rate limiting and brain module coordination.

Author: GitHub Copilot
"""

import os
import json
import time
import requests
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RateLimiter:
    """Intelligent rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int, calls_per_hour: Optional[int] = None):
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour or calls_per_minute * 60
        self.minute_calls = []
        self.hour_calls = []
        self.lock = threading.Lock()
    
    def can_make_call(self) -> bool:
        """Check if API call is allowed under rate limits"""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        with self.lock:
            # Clean old calls
            self.minute_calls = [call_time for call_time in self.minute_calls if call_time > minute_ago]
            self.hour_calls = [call_time for call_time in self.hour_calls if call_time > hour_ago]
            
            # Check limits
            if len(self.minute_calls) >= self.calls_per_minute:
                return False
            if len(self.hour_calls) >= self.calls_per_hour:
                return False
            
            return True
    
    def record_call(self):
        """Record an API call"""
        now = datetime.now()
        with self.lock:
            self.minute_calls.append(now)
            self.hour_calls.append(now)

class APIManager:
    """Enhanced API manager for FSOT 2.0 neuromorphic system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/api_config.json"
        self.config = self._load_config()
        self.rate_limiters = self._setup_rate_limiters()
        self.session = requests.Session()
        
        # Brain module coordination
        self.brain_context = None
        
        # Setup free API fallback
        self._setup_free_api_fallback()
    
    def _setup_free_api_fallback(self):
        """Setup free API manager as fallback"""
        try:
            from .free_api_manager import get_free_api_manager
            self.free_api_manager = get_free_api_manager()
            logger.info("Free API manager configured as fallback")
        except ImportError as e:
            logger.warning(f"Free API manager not available: {e}")
            self.free_api_manager = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load API configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"API config not found at {self.config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default API configuration"""
        return {
            "openai": {
                "enabled": False,
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "rate_limit": {"per_minute": 60, "per_hour": 3600},
                "model": "gpt-4"
            },
            "github": {
                "enabled": True,
                "api_key": "",
                "base_url": "https://api.github.com",
                "rate_limit": {"per_minute": 83, "per_hour": 5000}
            },
            "wolfram": {
                "enabled": False,
                "api_key": "",
                "base_url": "http://api.wolframalpha.com/v2",
                "rate_limit": {"per_minute": 33, "per_hour": 2000}
            },
            "huggingface": {
                "enabled": False,
                "api_key": "",
                "base_url": "https://api-inference.huggingface.co",
                "rate_limit": {"per_minute": 16, "per_hour": 1000}
            }
        }
    
    def _setup_rate_limiters(self) -> Dict[str, RateLimiter]:
        """Setup rate limiters for each API"""
        limiters = {}
        for api_name, api_config in self.config.items():
            if isinstance(api_config, dict) and "rate_limit" in api_config:
                limits = api_config["rate_limit"]
                limiters[api_name] = RateLimiter(
                    calls_per_minute=limits["per_minute"],
                    calls_per_hour=limits["per_hour"]
                )
        return limiters
    
    def set_brain_context(self, brain_orchestrator):
        """Set brain context for enhanced API coordination"""
        self.brain_context = brain_orchestrator
    
    def is_api_enabled(self, api_name: str) -> bool:
        """Check if API is enabled and configured"""
        api_config = self.config.get(api_name, {})
        return api_config.get("enabled", False) and api_config.get("api_key", "")
    
    def github_search_repositories(self, query: str, per_page: int = 10) -> Optional[Dict]:
        """Search GitHub repositories using the GitHub API"""
        if not self.is_api_enabled("github"):
            return None
        
        endpoint = "search/repositories"
        params = {
            "q": query, 
            "sort": "stars", 
            "order": "desc",
            "per_page": per_page
        }
        
        return self.make_api_call("github", endpoint, "GET", params, brain_module="temporal_lobe")
    
    def search_repositories(self, query: str, per_page: int = 10) -> Optional[Dict]:
        """Search repositories using GitHub API or free alternative"""
        # Try GitHub API first if available
        if self.is_api_enabled("github"):
            return self.github_search_repositories(query, per_page)
        
        # Fallback to free API manager
        elif self.free_api_manager:
            logger.info("Using free API manager for repository search")
            return self.free_api_manager.github_search_repositories(query, per_page=per_page)
        
        logger.warning("No API available for repository search")
        return None
    
    def get_learning_data(self, domain: str) -> Optional[Dict]:
        """Get learning data using free APIs"""
        if self.free_api_manager:
            return self.free_api_manager.get_learning_data(domain)
        
        logger.warning("No free API manager available for learning data")
        return None
    
    def search_and_learn(self, topic: str) -> Optional[Dict]:
        """Search and learn using available APIs"""
        if self.free_api_manager:
            return self.free_api_manager.search_and_learn(topic)
        
        logger.warning("No API available for search and learn")
        return None
    
    def make_api_call(self, api_name: str, endpoint: str, method: str = "GET", 
                     data: Optional[Dict] = None, headers: Optional[Dict] = None, brain_module: Optional[str] = None) -> Optional[Dict]:
        """Make rate-limited API call with brain module coordination"""
        
        if not self.is_api_enabled(api_name):
            logger.warning(f"API {api_name} is not enabled or configured")
            return None
        
        # Check rate limits
        rate_limiter = self.rate_limiters.get(api_name)
        if rate_limiter and not rate_limiter.can_make_call():
            logger.warning(f"Rate limit exceeded for {api_name}")
            return None
        
        # Prepare request
        api_config = self.config[api_name]
        base_url = api_config["base_url"]
        url = f"{base_url}/{endpoint.lstrip('/')}"
        
        # Setup headers
        request_headers = headers or {}
        if api_name == "openai":
            request_headers["Authorization"] = f"Bearer {api_config['api_key']}"
        elif api_name == "github":
            request_headers["Authorization"] = f"token {api_config['api_key']}"
        elif api_name == "huggingface":
            request_headers["Authorization"] = f"Bearer {api_config['api_key']}"
        
        # Brain module coordination
        if self.brain_context and brain_module:
            signal = {
                "type": "api_request",
                "api": api_name,
                "endpoint": endpoint,
                "module": brain_module,
                "timestamp": datetime.now().isoformat()
            }
            self.brain_context.send_signal("thalamus", signal)
        
        try:
            # Make request
            if method.upper() == "GET":
                response = self.session.get(url, headers=request_headers, params=data)
            elif method.upper() == "POST":
                response = self.session.post(url, headers=request_headers, json=data)
            else:
                response = self.session.request(method, url, headers=request_headers, json=data)
            
            # Record successful call
            if rate_limiter:
                rate_limiter.record_call()
            
            # Process response
            if response.status_code == 200:
                result = response.json() if response.content else {}
                
                # Brain module coordination
                if self.brain_context and brain_module:
                    signal = {
                        "type": "api_response",
                        "api": api_name,
                        "endpoint": endpoint,
                        "module": brain_module,
                        "success": True,
                        "timestamp": datetime.now().isoformat()
                    }
                    self.brain_context.send_signal("thalamus", signal)
                
                return result
            else:
                logger.error(f"API call failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"API call exception: {e}")
            return None
    
    # Specific API methods
    def openai_chat_completion(self, messages: List[Dict], model: Optional[str] = None, brain_module: str = "frontal_cortex") -> Optional[str]:
        """OpenAI chat completion with brain module coordination"""
        if not self.is_api_enabled("openai"):
            return None
        
        model = model or self.config["openai"]["model"]
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = self.make_api_call("openai", "chat/completions", "POST", data, brain_module=brain_module)
        if response and "choices" in response:
            return response["choices"][0]["message"]["content"]
        return None
    
    def wolfram_query(self, query: str, brain_module: str = "frontal_cortex") -> Optional[Dict]:
        """Wolfram Alpha query with brain module coordination"""
        if not self.is_api_enabled("wolfram"):
            return None
        
        params = {
            "input": query,
            "appid": self.config["wolfram"]["api_key"],
            "output": "json"
        }
        
        return self.make_api_call("wolfram", "query", "GET", params, brain_module=brain_module)
    
    def huggingface_inference(self, model: str, inputs: Dict, brain_module: str = "temporal_lobe") -> Optional[Dict]:
        """HuggingFace model inference with brain module coordination"""
        if not self.is_api_enabled("huggingface"):
            return None
        
        endpoint = f"models/{model}"
        return self.make_api_call("huggingface", endpoint, "POST", inputs, brain_module=brain_module)
    
    def github_search(self, query: str, search_type: str = "repositories", brain_module: str = "temporal_lobe") -> Optional[Dict]:
        """GitHub search with brain module coordination"""
        if not self.is_api_enabled("github"):
            return None
        
        endpoint = f"search/{search_type}"
        params = {"q": query, "sort": "stars", "order": "desc"}
        
        return self.make_api_call("github", endpoint, "GET", params, brain_module=brain_module)
    
    def get_api_status(self) -> Dict[str, Any]:
        """Get status of all APIs"""
        status = {}
        for api_name in self.config.keys():
            rate_limiter = self.rate_limiters.get(api_name)
            status[api_name] = {
                "enabled": self.is_api_enabled(api_name),
                "configured": bool(self.config.get(api_name, {}).get("api_key")),
                "rate_limit_status": {
                    "minute_calls": len(rate_limiter.minute_calls) if rate_limiter else 0,
                    "hour_calls": len(rate_limiter.hour_calls) if rate_limiter else 0,
                    "can_make_call": rate_limiter.can_make_call() if rate_limiter else False
                } if rate_limiter else None
            }
        return status

def create_api_config_template():
    """Create API configuration template"""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    config_path = config_dir / "api_config.json"
    if not config_path.exists():
        manager = APIManager()
        with open(config_path, 'w') as f:
            json.dump(manager._get_default_config(), f, indent=2)
        print(f"API configuration template created at {config_path}")
        print("Please add your API keys to enable external services.")

if __name__ == "__main__":
    # Test API manager
    create_api_config_template()
    
    manager = APIManager()
    print("API Status:")
    for api, status in manager.get_api_status().items():
        print(f"  {api}: {'✅' if status['enabled'] else '❌'} {status}")
