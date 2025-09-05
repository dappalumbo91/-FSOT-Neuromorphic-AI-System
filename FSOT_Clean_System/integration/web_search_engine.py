#!/usr/bin/env python3
"""
ENHANCED FSOT 2.0 WEB SEARCH ENGINE
===================================

Production-grade web search engine with Google Custom Search API integration,
intelligent caching, content parsing, and brain module coordination for the
enhanced neuromorphic architecture.

Author: GitHub Copilot
"""

import os
import sys
import time
import json
import requests
import hashlib
import pickle
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse, urljoin, quote_plus
import re
import logging
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Advanced content parsing libraries
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("Warning: BeautifulSoup not available. Install with: pip install beautifulsoup4")

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    print("Warning: Trafilatura not available. Install with: pip install trafilatura")

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    print("Warning: Newspaper3k not available. Install with: pip install newspaper3k")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchResult:
    """Enhanced search result with content analysis"""
    
    def __init__(self, title: str, url: str, snippet: str, content: str = ""):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.content = content
        self.timestamp = datetime.now()
        self.relevance_score = 0.0
        self.content_type = self._classify_content_type()
        self.keywords = self._extract_keywords()
    
    def _classify_content_type(self) -> str:
        """Classify the type of content"""
        url_lower = self.url.lower()
        title_lower = self.title.lower()
        
        if any(domain in url_lower for domain in ['github.com', 'stackoverflow.com', 'docs.python.org']):
            return "technical"
        elif any(word in title_lower for word in ['research', 'study', 'paper', 'journal']):
            return "academic"
        elif any(word in title_lower for word in ['news', 'breaking', 'report']):
            return "news"
        elif any(word in title_lower for word in ['tutorial', 'guide', 'how to']):
            return "educational"
        else:
            return "general"
    
    def _extract_keywords(self) -> List[str]:
        """Extract keywords from title and snippet"""
        text = f"{self.title} {self.snippet}".lower()
        # Simple keyword extraction - can be enhanced with NLP
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        return list(set(words))[:10]  # Top 10 unique keywords
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "content": self.content[:1000],  # Truncate for storage
            "timestamp": self.timestamp.isoformat(),
            "relevance_score": self.relevance_score,
            "content_type": self.content_type,
            "keywords": self.keywords
        }

class WebSearchCache:
    """Intelligent caching system for web search results"""

    def __init__(self, cache_dir: str = "data/web_search_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry = 3600  # 1 hour default
        self.max_cache_size = 1000  # Maximum cached queries
        
        # Memory cache for frequent queries
        self.memory_cache = {}
        self.cache_access_times = {}
    
    def _get_cache_key(self, query: str, params: Optional[Dict] = None) -> str:
        """Generate cache key from query and parameters"""
        key_data = {"query": query, "params": params or {}}
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, query: str, params: Optional[Dict] = None) -> Optional[List[SearchResult]]:
        """Get cached search results"""
        cache_key = self._get_cache_key(query, params)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            self.cache_access_times[cache_key] = time.time()
            return self.memory_cache[cache_key]
        
        # Check file cache
        cache_file = self.cache_dir / f"{cache_key}.cache"
        if cache_file.exists():
            try:
                # Check if cache is still valid
                if cache_file.stat().st_mtime + self.cache_expiry > time.time():
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                    
                    # Convert back to SearchResult objects
                    results = []
                    for item in cached_data:
                        result = SearchResult(
                            title=item["title"],
                            url=item["url"],
                            snippet=item["snippet"],
                            content=item.get("content", "")
                        )
                        result.relevance_score = item.get("relevance_score", 0.0)
                        results.append(result)
                    
                    # Store in memory cache
                    self.memory_cache[cache_key] = results
                    self.cache_access_times[cache_key] = time.time()
                    
                    return results
                else:
                    # Cache expired, remove file
                    cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def set(self, query: str, results: List[SearchResult], params: Optional[Dict] = None):
        """Cache search results"""
        cache_key = self._get_cache_key(query, params)
        
        # Store in memory cache
        self.memory_cache[cache_key] = results
        self.cache_access_times[cache_key] = time.time()
        
        # Clean memory cache if too large
        if len(self.memory_cache) > 100:
            # Remove least recently used
            oldest_key = min(self.cache_access_times.items(), key=lambda x: x[1])[0]
            del self.memory_cache[oldest_key]
            del self.cache_access_times[oldest_key]
        
        # Store in file cache
        try:
            cache_file = self.cache_dir / f"{cache_key}.cache"
            with open(cache_file, 'wb') as f:
                pickle.dump([result.to_dict() for result in results], f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def clear_expired(self):
        """Clear expired cache files"""
        current_time = time.time()
        for cache_file in self.cache_dir.glob("*.cache"):
            if cache_file.stat().st_mtime + self.cache_expiry < current_time:
                cache_file.unlink()

class EnhancedWebSearchEngine:
    """Production-grade web search engine for FSOT 2.0"""
    
    def __init__(self, brain_orchestrator=None, api_manager=None):
        self.brain_orchestrator = brain_orchestrator
        self.api_manager = api_manager
        self.cache = WebSearchCache()
        self.session = requests.Session()
        
        # Load API configuration
        self._load_api_config()
        
        # Initialize search engines
        self.search_engines = self._initialize_search_engines()
        
        # Content parsing configuration
        self.max_content_length = 10000
        self.parsing_timeout = 30
        
        # Brain module coordination
        self.search_history = []
        
        # Free search engine fallback
        self.free_search_engine = None
    
    def _load_api_config(self):
        """Load API configuration from config file"""
        try:
            config_file = Path(__file__).parent.parent / "config" / "api_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Load Google Custom Search API configuration
                google_config = config.get("google_custom_search", {})
                self.google_api_key = google_config.get("api_key", "")
                self.google_cse_id = google_config.get("search_engine_id", "")
                
                # Clean up placeholder values
                if self.google_api_key.startswith("your-"):
                    self.google_api_key = ""
                if self.google_cse_id.startswith("your-"):
                    self.google_cse_id = ""
                
                # If no Google API, try to use free search engine
                if not self.google_api_key or not self.google_cse_id:
                    self._setup_free_search_fallback()
            else:
                # Fallback to environment variables
                self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
                self.google_cse_id = os.getenv("GOOGLE_CSE_ID", "")
                
                # If no API keys, setup free search
                if not self.google_api_key or not self.google_cse_id:
                    self._setup_free_search_fallback()
        except Exception as e:
            logger.warning(f"Failed to load API config: {e}")
            self.google_api_key = ""
            self.google_cse_id = ""
            self._setup_free_search_fallback()
    
    def _setup_free_search_fallback(self):
        """Setup free search engine as fallback"""
        try:
            from .free_web_search_engine import get_free_web_search_engine
            self.free_search_engine = get_free_web_search_engine(self.brain_orchestrator)
            logger.info("Free web search engine configured as fallback")
        except ImportError as e:
            logger.warning(f"Free search engine not available: {e}")
            self.free_search_engine = None
        
    def _initialize_search_engines(self) -> Dict[str, Dict]:
        """Initialize available search engines"""
        engines = {}
        
        # Google Custom Search
        if self.google_api_key and self.google_cse_id:
            engines["google"] = {
                "name": "Google Custom Search",
                "url": "https://www.googleapis.com/customsearch/v1",
                "enabled": True,
                "rate_limit": 100  # per day
            }
        
        # DuckDuckGo (backup)
        engines["duckduckgo"] = {
            "name": "DuckDuckGo",
            "url": "https://api.duckduckgo.com/",
            "enabled": True,
            "rate_limit": 1000  # per day
        }
        
        return engines
    
    def search(self, query: str, num_results: int = 10, engine: str = "auto", 
              brain_module: str = "temporal_lobe") -> List[SearchResult]:
        """Enhanced search with brain module coordination"""
        
        # Check cache first
        cached_results = self.cache.get(query, {"num_results": num_results})
        if cached_results:
            logger.info(f"Cache hit for query: {query}")
            self._send_brain_signal("search_cache_hit", query, brain_module)
            return cached_results[:num_results]
        
        # Determine search engine
        if engine == "auto":
            if "google" in self.search_engines and self.search_engines["google"]["enabled"]:
                engine = "google"
            elif self.free_search_engine:
                engine = "free"
            else:
                engine = "duckduckgo"
        
        # Perform search
        results = []
        try:
            if engine == "google":
                results = self._google_search(query, num_results)
            elif engine == "free" and self.free_search_engine:
                # Use free search engine
                free_results = self.free_search_engine.search(query, num_results)
                # Convert to our SearchResult format if needed
                for free_result in free_results:
                    result = SearchResult(
                        title=free_result.title,
                        url=free_result.url,
                        snippet=free_result.snippet
                    )
                    result.content = free_result.content
                    result.relevance_score = free_result.relevance_score
                    results.append(result)
            elif engine == "duckduckgo":
                results = self._duckduckgo_search(query, num_results)
            
            # Enhance results with content extraction
            results = self._enhance_results_with_content(results)
            
            # Calculate relevance scores
            results = self._calculate_relevance_scores(results, query)
            
            # Sort by relevance
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Cache results
            self.cache.set(query, results, {"num_results": num_results})
            
            # Record search
            search_record = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "engine": engine,
                "results_count": len(results),
                "brain_module": brain_module
            }
            self.search_history.append(search_record)
            
            # Brain module coordination
            self._send_brain_signal("search_completed", query, brain_module, {
                "results_count": len(results),
                "engine": engine
            })
            
            return results[:num_results]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            self._send_brain_signal("search_error", query, brain_module, {"error": str(e)})
            return []
    
    def _google_search(self, query: str, num_results: int) -> List[SearchResult]:
        """Perform Google Custom Search"""
        if not self.google_api_key or not self.google_cse_id:
            raise ValueError("Google API key or CSE ID not configured")
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_api_key,
            "cx": self.google_cse_id,
            "q": query,
            "num": min(num_results, 10)  # Google allows max 10 per request
        }
        
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get("items", []):
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", "")
            )
            results.append(result)
        
        return results
    
    def _duckduckgo_search(self, query: str, num_results: int) -> List[SearchResult]:
        """Perform DuckDuckGo search (simplified)"""
        # Note: DuckDuckGo doesn't provide a direct search API
        # This is a placeholder - in production, you might use web scraping
        # or alternative search APIs
        
        results = []
        # Placeholder implementation
        for i in range(min(num_results, 5)):
            result = SearchResult(
                title=f"DuckDuckGo Result {i+1} for: {query}",
                url=f"https://example.com/result{i+1}",
                snippet=f"This is a placeholder result for query: {query}"
            )
            results.append(result)
        
        return results
    
    def _enhance_results_with_content(self, results: List[SearchResult]) -> List[SearchResult]:
        """Extract full content from search results"""
        enhanced_results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_result = {
                executor.submit(self._extract_content, result): result 
                for result in results
            }
            
            for future in as_completed(future_to_result, timeout=30):
                result = future_to_result[future]
                try:
                    content = future.result()
                    result.content = content
                    enhanced_results.append(result)
                except Exception as e:
                    logger.warning(f"Content extraction failed for {result.url}: {e}")
                    enhanced_results.append(result)  # Keep result without content
        
        return enhanced_results
    
    def _extract_content(self, result: SearchResult) -> str:
        """Extract content from a URL"""
        try:
            response = self.session.get(result.url, timeout=10)
            response.raise_for_status()
            
            content = ""
            
            # Try trafilatura first (best for article extraction)
            if TRAFILATURA_AVAILABLE:
                content = trafilatura.extract(response.text)
                if content:
                    return content[:self.max_content_length]
            
            # Try newspaper3k
            if NEWSPAPER_AVAILABLE:
                article = Article(result.url)
                article.set_html(response.text)
                article.parse()
                if article.text:
                    return article.text[:self.max_content_length]
            
            # Fallback to BeautifulSoup
            if BEAUTIFULSOUP_AVAILABLE:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                content = soup.get_text()
                # Clean up whitespace
                lines = (line.strip() for line in content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                content = ' '.join(chunk for chunk in chunks if chunk)
                return content[:self.max_content_length]
            
            # Basic text extraction as last resort
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.get_text()[:self.max_content_length]
            
        except Exception as e:
            logger.warning(f"Content extraction failed for {result.url}: {e}")
            return ""
    
    def _calculate_relevance_scores(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Calculate relevance scores for search results"""
        query_words = set(query.lower().split())
        
        for result in results:
            score = 0.0
            
            # Title relevance (highest weight)
            title_words = set(result.title.lower().split())
            title_overlap = len(query_words.intersection(title_words))
            score += title_overlap * 3.0
            
            # Snippet relevance
            snippet_words = set(result.snippet.lower().split())
            snippet_overlap = len(query_words.intersection(snippet_words))
            score += snippet_overlap * 2.0
            
            # Content relevance (if available)
            if result.content:
                content_words = set(result.content.lower().split())
                content_overlap = len(query_words.intersection(content_words))
                score += content_overlap * 1.0
            
            # Content type bonus
            if result.content_type == "technical":
                score += 0.5
            elif result.content_type == "academic":
                score += 0.3
            
            # Normalize by query length
            result.relevance_score = score / len(query_words) if query_words else 0.0
        
        return results
    
    def _send_brain_signal(self, signal_type: str, query: str, brain_module: str, metadata: Dict = None):
        """Send signal to brain orchestrator"""
        if self.brain_orchestrator:
            signal = {
                "type": f"web_search_{signal_type}",
                "query": query,
                "module": brain_module,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            self.brain_orchestrator.send_signal("thalamus", signal)
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        if not self.search_history:
            return {"total_searches": 0}
        
        total_searches = len(self.search_history)
        recent_searches = [s for s in self.search_history 
                          if datetime.fromisoformat(s["timestamp"]) > datetime.now() - timedelta(days=1)]
        
        engines_used = {}
        for search in self.search_history:
            engine = search["engine"]
            engines_used[engine] = engines_used.get(engine, 0) + 1
        
        return {
            "total_searches": total_searches,
            "recent_searches_24h": len(recent_searches),
            "engines_used": engines_used,
            "cache_hit_rate": getattr(self.cache, 'hit_rate', 0.0),
            "average_results_per_search": sum(s["results_count"] for s in self.search_history) / total_searches
        }

if __name__ == "__main__":
    # Test web search engine
    search_engine = EnhancedWebSearchEngine()
    
    print("Web Search Engine Status:")
    print(f"Available engines: {list(search_engine.search_engines.keys())}")
    
    # Test search (if API keys are configured)
    if search_engine.google_api_key:
        results = search_engine.search("artificial intelligence", num_results=3)
        print(f"\nSearch results: {len(results)}")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.title[:50]}... (Score: {result.relevance_score:.2f})")
    else:
        print("\nGoogle API not configured - add GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables")
