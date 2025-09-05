#!/usr/bin/env python3
"""
FREE WEB SEARCH ENGINE FOR ENHANCED FSOT 2.0
=============================================

A completely free web search and information retrieval system that uses:
- Web scraping of free search engines (DuckDuckGo, Bing, etc.)
- Free API endpoints (Wikipedia, ArXiv, Reddit, etc.)
- Advanced content parsing and extraction
- Intelligent caching system
- No paid API keys required

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
from urllib.parse import urlparse, urljoin, quote_plus, parse_qs
import re
import logging
from functools import lru_cache
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import urllib.request
import urllib.parse

# Content parsing libraries with proper type hints
try:
    from bs4 import BeautifulSoup, Tag, NavigableString
    from bs4.element import PageElement
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    print("Warning: BeautifulSoup not available. Install with: pip install beautifulsoup4")
    BeautifulSoup = None
    Tag = None
    NavigableString = None
    PageElement = None

# Helper functions for safe BeautifulSoup operations
def safe_find(element, *args, **kwargs):
    """Safely find an element with proper type checking"""
    if element is None or not hasattr(element, 'find'):
        return None
    try:
        return element.find(*args, **kwargs)
    except Exception:
        return None

def safe_get_text(element, default: str = '') -> str:
    """Safely get text from an element"""
    if element is None or not hasattr(element, 'get_text'):
        return default
    try:
        return element.get_text().strip()
    except Exception:
        return default

def safe_get_attr(element, attr: str, default: str = '') -> str:
    """Safely get an attribute from an element"""
    if element is None or not hasattr(element, 'get'):
        return default
    try:
        value = element.get(attr, default)
        # Handle AttributeValueList and other BeautifulSoup types
        if isinstance(value, list):
            return str(value[0]) if value else default
        return str(value) if value is not None else default
    except Exception:
        return default

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False
    print("Info: Trafilatura not available. Using basic parsing.")

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    print("Info: Newspaper3k not available. Using basic parsing.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchResult:
    """Represents a single search result"""
    
    def __init__(self, title: str = "", url: str = "", snippet: str = "", content: str = ""):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.content = content
        self.relevance_score = 0.0
        self.timestamp = datetime.now()
        self.source = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        result = cls(
            title=data.get("title", ""),
            url=data.get("url", ""),
            snippet=data.get("snippet", ""),
            content=data.get("content", "")
        )
        result.relevance_score = data.get("relevance_score", 0.0)
        result.source = data.get("source", "unknown")
        return result

class FreeWebSearchEngine:
    """Free web search engine using scraping and free APIs"""
    
    def __init__(self, brain_orchestrator=None):
        self.brain_orchestrator = brain_orchestrator
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Load free API configuration
        self.config = self._load_free_api_config()
        
        # Cache setup
        self.cache_dir = Path(__file__).parent.parent / "cache" / "free_search"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry = 3600  # 1 hour
        
        # Rate limiting
        self.last_request_time = {}
        self.request_delays = {
            'duckduckgo': 1.0,
            'bing': 2.0,
            'wikipedia': 0.5,
            'reddit': 1.0
        }
        
        logger.info("Free Web Search Engine initialized")
    
    def _load_free_api_config(self) -> Dict[str, Any]:
        """Load free API configuration"""
        try:
            config_file = Path(__file__).parent.parent / "config" / "free_api_config.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load free API config: {e}")
        
        # Default configuration
        return {
            "free_apis": {"enabled": True},
            "web_search": {"enabled": True},
            "knowledge_sources": {}
        }
    
    def _rate_limit(self, source: str):
        """Simple rate limiting"""
        delay = self.request_delays.get(source, 1.0)
        last_time = self.last_request_time.get(source, 0)
        time_since_last = time.time() - last_time
        
        if time_since_last < delay:
            time.sleep(delay - time_since_last)
        
        self.last_request_time[source] = time.time()
    
    def _get_cache_key(self, query: str, source: str) -> str:
        """Generate cache key"""
        key_data = f"{source}:{query}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_results(self, query: str, source: str) -> Optional[List[SearchResult]]:
        """Get cached search results"""
        cache_key = self._get_cache_key(query, source)
        cache_file = self.cache_dir / f"{cache_key}.cache"
        
        if cache_file.exists():
            try:
                if cache_file.stat().st_mtime + self.cache_expiry > time.time():
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                        return [SearchResult.from_dict(item) for item in data]
                else:
                    cache_file.unlink()  # Remove expired cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def _cache_results(self, query: str, source: str, results: List[SearchResult]):
        """Cache search results"""
        try:
            cache_key = self._get_cache_key(query, source)
            cache_file = self.cache_dir / f"{cache_key}.cache"
            
            with open(cache_file, 'wb') as f:
                pickle.dump([result.to_dict() for result in results], f)
        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")
    
    def search_duckduckgo(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search using DuckDuckGo (free, no API key required)"""
        self._rate_limit('duckduckgo')
        
        # Check cache first
        cached = self._get_cached_results(query, 'duckduckgo')
        if cached:
            return cached[:num_results]
        
        results = []
        try:
            # DuckDuckGo HTML search
            search_url = "https://html.duckduckgo.com/html/"
            params = {
                'q': query,
                'b': '',  # no ads
                'kl': 'us-en',
                's': '0'
            }
            
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            
            if BEAUTIFULSOUP_AVAILABLE and BeautifulSoup:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find search results
                result_divs = soup.find_all('div', class_='result')
                
                for div in result_divs[:num_results]:
                    try:
                        title_elem = safe_find(div, 'a', class_='result__a')
                        snippet_elem = safe_find(div, 'a', class_='result__snippet')
                        
                        if title_elem and snippet_elem:
                            title = safe_get_text(title_elem)
                            url = safe_get_attr(title_elem, 'href')
                            snippet = safe_get_text(snippet_elem)
                            
                            # Handle DuckDuckGo redirect URLs
                            if url and url.startswith('/l/?uddg='):
                                # Extract real URL from DuckDuckGo redirect
                                try:
                                    url = urllib.parse.unquote(url.split('uddg=')[1])
                                except (IndexError, ValueError):
                                    continue
                            
                            if title and url:
                                result = SearchResult(title=title, url=url, snippet=snippet)
                                result.source = "duckduckgo"
                                results.append(result)
                    except Exception as e:
                        logger.warning(f"Error parsing DuckDuckGo result: {e}")
                        continue
            
            # Cache results
            self._cache_results(query, 'duckduckgo', results)
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
        
        return results
    
    def search_wikipedia(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search Wikipedia using free API"""
        self._rate_limit('wikipedia')
        
        # Check cache first
        cached = self._get_cached_results(query, 'wikipedia')
        if cached:
            return cached[:num_results]
        
        results = []
        try:
            # Wikipedia search API
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': query,
                'srlimit': num_results
            }
            
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for page in data.get('query', {}).get('search', []):
                title = page.get('title', '')
                snippet = page.get('snippet', '').replace('<span class="searchmatch">', '').replace('</span>', '')
                url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                
                result = SearchResult(title=title, url=url, snippet=snippet)
                result.source = "wikipedia"
                results.append(result)
            
            # Cache results
            self._cache_results(query, 'wikipedia', results)
            
        except Exception as e:
            logger.error(f"Wikipedia search failed: {e}")
        
        return results
    
    def search_arxiv(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search ArXiv for scientific papers (free)"""
        self._rate_limit('arxiv')
        
        # Check cache first
        cached = self._get_cached_results(query, 'arxiv')
        if cached:
            return cached[:num_results]
        
        results = []
        try:
            # ArXiv API
            search_url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': num_results
            }
            
            response = self.session.get(search_url, params=params, timeout=15)
            response.raise_for_status()
            
            if BEAUTIFULSOUP_AVAILABLE and BeautifulSoup:
                soup = BeautifulSoup(response.text, 'xml')
                entries = soup.find_all('entry')
                
                for entry in entries:
                    title_elem = safe_find(entry, 'title')
                    summary_elem = safe_find(entry, 'summary')
                    id_elem = safe_find(entry, 'id')
                    
                    title = safe_get_text(title_elem) if title_elem else ''
                    summary = safe_get_text(summary_elem) if summary_elem else ''
                    link = safe_get_text(id_elem) if id_elem else ''
                    
                    result = SearchResult(title=title, url=link, snippet=summary)
                    result.source = "arxiv"
                    results.append(result)
            
            # Cache results
            self._cache_results(query, 'arxiv', results)
            
        except Exception as e:
            logger.error(f"ArXiv search failed: {e}")
        
        return results
    
    def search_reddit(self, query: str, subreddit: str = "all", num_results: int = 5) -> List[SearchResult]:
        """Search Reddit (free JSON API)"""
        self._rate_limit('reddit')
        
        cache_key = f"{subreddit}:{query}"
        cached = self._get_cached_results(cache_key, 'reddit')
        if cached:
            return cached[:num_results]
        
        results = []
        try:
            # Reddit JSON API
            search_url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                'q': query,
                'sort': 'relevance',
                'limit': num_results,
                'restrict_sr': 'on' if subreddit != 'all' else 'off'
            }
            
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for post in data.get('data', {}).get('children', []):
                post_data = post.get('data', {})
                title = post_data.get('title', '')
                url = post_data.get('url', '')
                snippet = post_data.get('selftext', '')[:200] + '...' if post_data.get('selftext') else ''
                
                result = SearchResult(title=title, url=url, snippet=snippet)
                result.source = "reddit"
                results.append(result)
            
            # Cache results
            self._cache_results(cache_key, 'reddit', results)
            
        except Exception as e:
            logger.error(f"Reddit search failed: {e}")
        
        return results
    
    def search_github(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Search GitHub repositories (free API, no auth required)"""
        self._rate_limit('github')
        
        # Check cache first
        cached = self._get_cached_results(query, 'github')
        if cached:
            return cached[:num_results]
        
        results = []
        try:
            # GitHub search API (public access)
            search_url = "https://api.github.com/search/repositories"
            params = {
                'q': query,
                'sort': 'stars',
                'order': 'desc',
                'per_page': num_results
            }
            
            response = self.session.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            for repo in data.get('items', []):
                title = repo.get('full_name', '')
                url = repo.get('html_url', '')
                snippet = repo.get('description', '') or 'No description available'
                
                result = SearchResult(title=title, url=url, snippet=snippet)
                result.source = "github"
                results.append(result)
            
            # Cache results
            self._cache_results(query, 'github', results)
            
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
        
        return results
    
    def extract_content(self, url: str) -> str:
        """Extract content from a webpage"""
        try:
            self._rate_limit('content_extraction')
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            content = ""
            
            # Try trafilatura first (best for article extraction)
            if TRAFILATURA_AVAILABLE:
                content = trafilatura.extract(response.text)
                if content:
                    return content[:5000]  # Limit content length
            
            # Try newspaper3k
            if NEWSPAPER_AVAILABLE:
                try:
                    article = Article(url)
                    article.set_html(response.text)
                    article.parse()
                    if article.text:
                        return article.text[:5000]
                except Exception:
                    pass
            
            # Fallback to BeautifulSoup
            if BEAUTIFULSOUP_AVAILABLE and BeautifulSoup:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                content = soup.get_text()
                # Clean up whitespace
                lines = (line.strip() for line in content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                content = ' '.join(chunk for chunk in chunks if chunk)
                return content[:5000]
            
        except Exception as e:
            logger.warning(f"Content extraction failed for {url}: {e}")
        
        return ""
    
    def comprehensive_search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Perform comprehensive search across all free sources"""
        all_results = []
        
        # Search sources in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.search_duckduckgo, query, num_results): "duckduckgo",
                executor.submit(self.search_wikipedia, query, max(2, num_results // 3)): "wikipedia",
                executor.submit(self.search_github, query, max(2, num_results // 4)): "github"
            }
            
            # Add ArXiv for scientific queries
            if any(word in query.lower() for word in ['research', 'science', 'study', 'paper', 'algorithm']):
                futures[executor.submit(self.search_arxiv, query, max(2, num_results // 4))] = "arxiv"
            
            # Add Reddit for general discussion
            if any(word in query.lower() for word in ['opinion', 'discussion', 'community', 'experience']):
                futures[executor.submit(self.search_reddit, query, "all", max(2, num_results // 4))] = "reddit"
            
            for future in as_completed(futures, timeout=30):
                source = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    logger.info(f"Got {len(results)} results from {source}")
                except Exception as e:
                    logger.warning(f"Search failed for {source}: {e}")
        
        # Remove duplicates and rank results
        unique_results = self._deduplicate_results(all_results)
        ranked_results = self._rank_results(unique_results, query)
        
        return ranked_results[:num_results]
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Simple ranking based on query relevance"""
        query_words = set(query.lower().split())
        
        for result in results:
            score = 0
            text = f"{result.title} {result.snippet}".lower()
            
            # Count query word matches
            for word in query_words:
                score += text.count(word)
            
            # Boost certain sources
            source_boost = {
                'wikipedia': 0.2,
                'arxiv': 0.3,
                'github': 0.1
            }
            score += source_boost.get(result.source, 0)
            
            result.relevance_score = score
        
        # Sort by relevance score
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    def search(self, query: str, num_results: int = 10, **kwargs) -> List[SearchResult]:
        """Main search interface"""
        logger.info(f"Searching for: {query}")
        
        # Send brain signal if orchestrator available
        if self.brain_orchestrator:
            try:
                self.brain_orchestrator.send_signal("temporal_lobe", "search_initiated", {
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.warning(f"Failed to send brain signal: {e}")
        
        # Perform comprehensive search
        results = self.comprehensive_search(query, num_results)
        
        # Extract content for top results
        for result in results[:3]:  # Only extract content for top 3 to save time
            if result.content == "":
                result.content = self.extract_content(result.url)
        
        logger.info(f"Search completed: {len(results)} results")
        
        # Send completion signal
        if self.brain_orchestrator:
            try:
                self.brain_orchestrator.send_signal("temporal_lobe", "search_completed", {
                    "query": query,
                    "results_count": len(results),
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.warning(f"Failed to send brain signal: {e}")
        
        return results

# Global instance
_free_search_engine = None

def get_free_web_search_engine(brain_orchestrator=None) -> FreeWebSearchEngine:
    """Get global free web search engine instance"""
    global _free_search_engine
    if _free_search_engine is None:
        _free_search_engine = FreeWebSearchEngine(brain_orchestrator)
    return _free_search_engine

if __name__ == "__main__":
    # Test the free search engine
    engine = FreeWebSearchEngine()
    
    test_queries = [
        "artificial intelligence 2025",
        "climate change solutions",
        "Python programming tutorials"
    ]
    
    for query in test_queries:
        print(f"\\nTesting query: {query}")
        results = engine.search(query, num_results=5)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Source: {result.source}")
            print(f"   Snippet: {result.snippet[:100]}...")
            print()
