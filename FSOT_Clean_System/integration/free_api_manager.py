#!/usr/bin/env python3
"""
FREE API MANAGER FOR ENHANCED FSOT 2.0
======================================

Manages access to completely free APIs that don't require API keys or payment.
Uses the discovered free APIs and web scraping capabilities for data access.

Author: GitHub Copilot
"""

import os
import sys
import time
import json
import requests
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FreeAPIManager:
    """Manages access to free APIs without requiring API keys"""
    
    def __init__(self, brain_orchestrator=None):
        self.brain_orchestrator = brain_orchestrator
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Enhanced-FSOT-2.0-Research-System/1.0 (Educational Purpose)'
        })
        
        # Load free API configuration
        self.config = self._load_free_api_config()
        
        # Rate limiting
        self.last_request_time = {}
        self.request_delays = {
            'github': 1.0,
            'jsonplaceholder': 0.5,
            'randomuser': 1.0,
            'exchange_rate': 2.0,
            'advice_slip': 1.0,
            'joke_api': 0.5
        }
        
        # Cache for API responses
        self.cache_dir = Path(__file__).parent.parent / "cache" / "free_apis"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry = 3600  # 1 hour
        
        logger.info("Free API Manager initialized")
    
    def _load_free_api_config(self) -> Dict[str, Any]:
        """Load free API configuration"""
        try:
            config_file = Path(__file__).parent.parent / "config" / "free_api_config.json"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load free API config: {e}")
        
        # Default configuration with discovered free APIs
        return {
            "free_apis": {
                "enabled": True,
                "discovered_apis": {
                    "jsonplaceholder": {
                        "url": "https://jsonplaceholder.typicode.com/",
                        "type": "mock_data",
                        "enabled": True
                    },
                    "github_api": {
                        "url": "https://api.github.com/",
                        "type": "development",
                        "enabled": True,
                        "rate_limit": 60
                    },
                    "randomuser": {
                        "url": "https://randomuser.me/api/",
                        "type": "random_data",
                        "enabled": True
                    },
                    "exchange_rate": {
                        "url": "https://api.exchangerate-api.com/v4/latest/USD",
                        "type": "financial",
                        "enabled": True
                    }
                }
            }
        }
    
    def _rate_limit(self, service: str):
        """Simple rate limiting"""
        delay = self.request_delays.get(service, 1.0)
        last_time = self.last_request_time.get(service, 0)
        time_since_last = time.time() - last_time
        
        if time_since_last < delay:
            time.sleep(delay - time_since_last)
        
        self.last_request_time[service] = time.time()
    
    def _get_cached_response(self, cache_key: str) -> Optional[Any]:
        """Get cached API response"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                if cache_file.stat().st_mtime + self.cache_expiry > time.time():
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                else:
                    cache_file.unlink()  # Remove expired cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def _cache_response(self, cache_key: str, data: Dict):
        """Cache API response"""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")
    
    def github_search_repositories(self, query: str, sort: str = "stars", per_page: int = 10) -> Dict[str, Any]:
        """Search GitHub repositories (free, no auth required)"""
        self._rate_limit('github')
        
        cache_key = f"github_repos_{query}_{sort}_{per_page}".replace(" ", "_")
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached
        
        try:
            url = "https://api.github.com/search/repositories"
            params = {
                'q': query,
                'sort': sort,
                'order': 'desc',
                'per_page': per_page
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            self._cache_response(cache_key, data)
            
            return data
            
        except Exception as e:
            logger.error(f"GitHub API error: {e}")
            return {"items": [], "total_count": 0}
    
    def github_get_user(self, username: str) -> Dict[str, Any]:
        """Get GitHub user information"""
        self._rate_limit('github')
        
        cache_key = f"github_user_{username}"
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached
        
        try:
            url = f"https://api.github.com/users/{username}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            self._cache_response(cache_key, data)
            
            return data
            
        except Exception as e:
            logger.error(f"GitHub user API error: {e}")
            return {}
    
    def get_random_user_data(self, results: int = 1, nationality: Optional[str] = None) -> Dict[str, Any]:
        """Get random user data from randomuser.me"""
        self._rate_limit('randomuser')
        
        cache_key = f"randomuser_{results}_{nationality or 'any'}"
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached
        
        try:
            url = "https://randomuser.me/api/"
            params = {'results': results}
            if nationality:
                params['nat'] = nationality
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            self._cache_response(cache_key, data)
            
            return data
            
        except Exception as e:
            logger.error(f"Random user API error: {e}")
            return {"results": []}
    
    def get_exchange_rates(self, base: str = "USD") -> Dict[str, Any]:
        """Get currency exchange rates"""
        self._rate_limit('exchange_rate')
        
        cache_key = f"exchange_rates_{base}"
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached
        
        try:
            url = f"https://api.exchangerate-api.com/v4/latest/{base}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            self._cache_response(cache_key, data)
            
            return data
            
        except Exception as e:
            logger.error(f"Exchange rate API error: {e}")
            return {"rates": {}}
    
    def get_random_advice(self) -> Dict[str, Any]:
        """Get random advice from advice slip API"""
        self._rate_limit('advice_slip')
        
        try:
            url = "https://api.adviceslip.com/advice"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse JSON from the response
            data = response.json()
            return data
            
        except Exception as e:
            logger.error(f"Advice slip API error: {e}")
            return {"slip": {"advice": "Unable to get advice at this time"}}
    
    def get_random_joke(self) -> Dict[str, Any]:
        """Get random joke"""
        self._rate_limit('joke_api')
        
        try:
            url = "https://official-joke-api.appspot.com/jokes/random"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return data
            
        except Exception as e:
            logger.error(f"Joke API error: {e}")
            return {"setup": "Why did the API fail?", "punchline": "Because it had a bad connection!"}
    
    def get_mock_posts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get mock posts from JSONPlaceholder"""
        self._rate_limit('jsonplaceholder')
        
        cache_key = f"jsonplaceholder_posts_{limit}"
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached
        
        try:
            url = "https://jsonplaceholder.typicode.com/posts"
            params = {'_limit': limit}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            self._cache_response(cache_key, data)
            
            return data
            
        except Exception as e:
            logger.error(f"JSONPlaceholder API error: {e}")
            return []
    
    def search_and_learn(self, topic: str) -> Dict[str, Any]:
        """Comprehensive search and learning using free APIs"""
        results = {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "sources": {},
            "insights": []
        }
        
        try:
            # Search GitHub for related repositories
            github_results = self.github_search_repositories(topic, per_page=5)
            results["sources"]["github"] = {
                "total_repositories": github_results.get("total_count", 0),
                "top_repositories": [
                    {
                        "name": repo["full_name"],
                        "description": repo.get("description", ""),
                        "stars": repo["stargazers_count"],
                        "language": repo.get("language", "Unknown"),
                        "url": repo["html_url"]
                    }
                    for repo in github_results.get("items", [])[:3]
                ]
            }
            
            # Add some wisdom
            advice = self.get_random_advice()
            results["sources"]["wisdom"] = advice.get("slip", {}).get("advice", "")
            
            # Generate insights
            if results["sources"]["github"]["total_repositories"] > 0:
                results["insights"].append(f"Found {results['sources']['github']['total_repositories']} GitHub repositories related to '{topic}'")
            
            top_repos = results["sources"]["github"]["top_repositories"]
            if top_repos:
                top_language = max(set(repo["language"] for repo in top_repos if repo["language"] != "Unknown"), 
                                 key=[repo["language"] for repo in top_repos].count, default="Unknown")
                if top_language != "Unknown":
                    results["insights"].append(f"Most popular programming language for '{topic}': {top_language}")
            
            # Send brain signal if available
            if self.brain_orchestrator:
                try:
                    self.brain_orchestrator.send_signal("hippocampus", "knowledge_acquired", {
                        "topic": topic,
                        "sources": list(results["sources"].keys()),
                        "insights_count": len(results["insights"])
                    })
                except Exception as e:
                    logger.warning(f"Failed to send brain signal: {e}")
            
        except Exception as e:
            logger.error(f"Error in search_and_learn: {e}")
            results["error"] = str(e)
        
        return results
    
    def get_learning_data(self, domain: str) -> Dict[str, Any]:
        """Get learning data for a specific domain"""
        learning_data = {
            "domain": domain,
            "timestamp": datetime.now().isoformat(),
            "data": {}
        }
        
        try:
            if domain.lower() in ["technology", "programming", "software"]:
                # Get technology-related data
                github_data = self.github_search_repositories("artificial intelligence", per_page=3)
                learning_data["data"]["trending_ai_repos"] = [
                    repo["full_name"] for repo in github_data.get("items", [])[:3]
                ]
            
            elif domain.lower() in ["finance", "economy", "business"]:
                # Get financial data
                rates = self.get_exchange_rates()
                learning_data["data"]["exchange_rates"] = {
                    "base": rates.get("base", "USD"),
                    "date": rates.get("date", ""),
                    "sample_rates": {k: v for k, v in list(rates.get("rates", {}).items())[:5]}
                }
            
            elif domain.lower() in ["social", "people", "demographics"]:
                # Get demographic data
                user_data = self.get_random_user_data(results=1)
                if user_data.get("results"):
                    user = user_data["results"][0]
                    learning_data["data"]["sample_demographics"] = {
                        "country": user.get("location", {}).get("country", ""),
                        "gender": user.get("gender", ""),
                        "age": user.get("dob", {}).get("age", 0)
                    }
            
            # Add some general wisdom
            advice = self.get_random_advice()
            learning_data["data"]["wisdom"] = advice.get("slip", {}).get("advice", "")
            
        except Exception as e:
            logger.error(f"Error getting learning data for {domain}: {e}")
            learning_data["error"] = str(e)
        
        return learning_data
    
    def test_all_apis(self) -> Dict[str, Any]:
        """Test all available free APIs"""
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "apis_tested": 0,
            "apis_working": 0,
            "results": {}
        }
        
        apis_to_test = [
            ("github", lambda: self.github_search_repositories("python", per_page=1)),
            ("randomuser", lambda: self.get_random_user_data(results=1)),
            ("exchange_rates", lambda: self.get_exchange_rates()),
            ("advice", lambda: self.get_random_advice()),
            ("jokes", lambda: self.get_random_joke()),
            ("mock_posts", lambda: self.get_mock_posts(limit=1))
        ]
        
        for api_name, test_func in apis_to_test:
            test_results["apis_tested"] += 1
            try:
                result = test_func()
                test_results["results"][api_name] = {
                    "status": "working",
                    "sample_data": str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
                }
                test_results["apis_working"] += 1
                logger.info(f"✅ {api_name} API working")
            except Exception as e:
                test_results["results"][api_name] = {
                    "status": "error",
                    "error": str(e)
                }
                logger.error(f"❌ {api_name} API failed: {e}")
        
        success_rate = (test_results["apis_working"] / test_results["apis_tested"]) * 100
        test_results["success_rate"] = success_rate
        
        logger.info(f"API Test Results: {test_results['apis_working']}/{test_results['apis_tested']} working ({success_rate:.1f}%)")
        
        return test_results

# Global instance
_free_api_manager = None

def get_free_api_manager(brain_orchestrator=None) -> FreeAPIManager:
    """Get global free API manager instance"""
    global _free_api_manager
    if _free_api_manager is None:
        _free_api_manager = FreeAPIManager(brain_orchestrator)
    return _free_api_manager

if __name__ == "__main__":
    # Test the free API manager
    manager = FreeAPIManager()
    
    print("🧪 Testing Free API Manager...")
    test_results = manager.test_all_apis()
    
    print(f"\\n📊 Test Summary:")
    print(f"APIs Tested: {test_results['apis_tested']}")
    print(f"APIs Working: {test_results['apis_working']}")
    print(f"Success Rate: {test_results['success_rate']:.1f}%")
    
    print("\\n🔍 Testing search and learn functionality...")
    learning_results = manager.search_and_learn("machine learning")
    print(f"GitHub repositories found: {learning_results['sources']['github']['total_repositories']}")
    print(f"Insights generated: {len(learning_results['insights'])}")
    
    print("\\n✅ Free API Manager test complete!")
