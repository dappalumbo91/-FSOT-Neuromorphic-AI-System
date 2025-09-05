#!/usr/bin/env python3
"""
Continuous API Discovery Enhancement for FSOT 2.0
================================================

Background API discovery system that continuously searches for new free APIs
to expand the knowledge access capabilities of the Enhanced FSOT 2.0 system.

Author: GitHub Copilot
"""

import json
import sqlite3
import threading
import time
import requests
import hashlib
import random
import re
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContinuousAPIDiscovery:
    """Enhanced continuous API discovery system for expanding free API access"""
    
    def __init__(self, discovery_db: str = "data/continuous_api_discovery.db"):
        self.discovery_db = Path(discovery_db)
        self.discovery_db.parent.mkdir(parents=True, exist_ok=True)
        
        self.search_active = False
        self.search_thread = None
        self.discovered_apis = {}
        self.discovery_stats = {
            "total_searches": 0,
            "apis_discovered": 0,
            "high_value_apis": 0,
            "last_discovery": None,
            "discovery_sessions": 0,
            "success_rate": 0.0
        }
        
        # Known free API categories to search for
        self.api_categories = [
            "weather", "news", "finance", "entertainment", "utilities",
            "data", "tools", "reference", "science", "government",
            "social", "productivity", "education", "health", "geography"
        ]
        
        # Quality indicators for evaluating APIs
        self.quality_indicators = [
            "free", "open", "public", "no auth", "no key", "json", "rest",
            "documentation", "examples", "rate limit", "https", "cors",
            "reliable", "stable", "active", "maintained"
        ]
        
        # Search patterns for finding APIs
        self.search_patterns = [
            "free public API {category}",
            "no auth API {category}",
            "open API {category} 2025",
            "{category} API no key required",
            "free REST API {category}",
            "public {category} data API"
        ]
        
        self._init_database()
        logger.info("Continuous API Discovery system initialized")
    
    def _init_database(self):
        """Initialize API discovery database"""
        with sqlite3.connect(self.discovery_db) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS discovered_apis (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    url TEXT,
                    description TEXT,
                    category TEXT,
                    quality_score REAL,
                    auth_required BOOLEAN,
                    rate_limit TEXT,
                    documentation_url TEXT,
                    discovered_at TEXT,
                    tested BOOLEAN DEFAULT FALSE,
                    test_results TEXT,
                    working_status TEXT,
                    last_tested TEXT,
                    usage_count INTEGER DEFAULT 0
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS discovery_sessions (
                    id TEXT PRIMARY KEY,
                    session_start TEXT,
                    session_end TEXT,
                    apis_found INTEGER,
                    search_queries INTEGER,
                    success_rate REAL,
                    session_notes TEXT
                )
            ''')
            
            # Indexes for performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_category ON discovered_apis(category)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_quality ON discovered_apis(quality_score)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_working ON discovered_apis(working_status)')
    
    def start_continuous_discovery(self, interval_minutes: int = 120):
        """Start continuous API discovery in background"""
        if self.search_active:
            logger.warning("API discovery already running")
            return
        
        self.search_active = True
        
        def discovery_loop():
            """Main discovery loop"""
            logger.info(f"Starting continuous API discovery (interval: {interval_minutes} minutes)")
            
            while self.search_active:
                try:
                    session_start = datetime.now()
                    logger.info("🔍 Starting API discovery session...")
                    
                    session_result = self._perform_discovery_session()
                    
                    session_end = datetime.now()
                    session_duration = (session_end - session_start).total_seconds()
                    
                    # Log session results
                    self._log_discovery_session(session_start, session_end, session_result)
                    
                    logger.info(f"✅ Discovery session complete: {session_result['apis_found']} APIs found in {session_duration:.1f}s")
                    
                    # Wait for next cycle
                    for _ in range(interval_minutes * 60):
                        if not self.search_active:
                            break
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Discovery cycle error: {e}")
                    time.sleep(300)  # 5 minute pause before retry
            
            logger.info("Continuous API discovery stopped")
        
        self.search_thread = threading.Thread(target=discovery_loop, daemon=True)
        self.search_thread.start()
    
    def stop_continuous_discovery(self):
        """Stop continuous API discovery"""
        self.search_active = False
        if self.search_thread:
            self.search_thread.join(timeout=10)
        logger.info("API discovery stopped")
    
    def _perform_discovery_session(self) -> Dict[str, Any]:
        """Perform one discovery session"""
        
        session_result = {
            "apis_found": 0,
            "search_queries": 0,
            "high_quality_apis": 0,
            "categories_searched": [],
            "success_rate": 0.0
        }
        
        # Select random categories to search
        search_categories = random.sample(self.api_categories, min(3, len(self.api_categories)))
        session_result["categories_searched"] = search_categories
        
        total_searches = 0
        successful_searches = 0
        
        for category in search_categories:
            for pattern in self.search_patterns:
                try:
                    query = pattern.format(category=category)
                    total_searches += 1
                    session_result["search_queries"] += 1
                    
                    # Simulate API search (would be replaced with real web search)
                    found_apis = self._simulate_api_search(query, category)
                    
                    if found_apis:
                        successful_searches += 1
                        
                        for api_info in found_apis:
                            if self._evaluate_and_store_api(api_info):
                                session_result["apis_found"] += 1
                                
                                if api_info.get("quality_score", 0) > 0.7:
                                    session_result["high_quality_apis"] += 1
                    
                    # Rate limiting
                    time.sleep(random.uniform(1, 3))
                    
                except Exception as e:
                    logger.error(f"Search error for '{query}': {e}")
        
        # Calculate success rate
        if total_searches > 0:
            session_result["success_rate"] = successful_searches / total_searches
        
        # Update global stats
        self.discovery_stats["total_searches"] += total_searches
        self.discovery_stats["apis_discovered"] += session_result["apis_found"]
        self.discovery_stats["high_value_apis"] += session_result["high_quality_apis"]
        self.discovery_stats["last_discovery"] = datetime.now().isoformat()
        self.discovery_stats["discovery_sessions"] += 1
        
        if self.discovery_stats["total_searches"] > 0:
            self.discovery_stats["success_rate"] = (
                self.discovery_stats["apis_discovered"] / self.discovery_stats["total_searches"]
            )
        
        return session_result
    
    def _simulate_api_search(self, query: str, category: str) -> List[Dict[str, Any]]:
        """Simulate API search results (replace with real web search integration)"""
        
        # Mock API results - in real implementation, this would:
        # 1. Use web search to find API directories
        # 2. Parse results for API information
        # 3. Extract API details from documentation pages
        
        mock_apis = []
        
        # Generate 1-3 mock APIs per search
        num_apis = random.randint(0, 3)
        
        for i in range(num_apis):
            api_id = random.randint(1000, 9999)
            
            # Some APIs are higher quality than others
            is_high_quality = random.random() > 0.6
            
            mock_api = {
                "name": f"{category.title()} API {api_id}",
                "url": f"https://api.{category}{api_id}.com/v1",
                "description": f"Free {category} data API with JSON responses",
                "category": category,
                "auth_required": not is_high_quality,  # High quality APIs often don't need auth
                "rate_limit": "100 requests/hour" if is_high_quality else "50 requests/hour",
                "documentation_url": f"https://docs.{category}{api_id}.com",
                "https_support": is_high_quality,
                "cors_support": is_high_quality,
                "json_response": True,
                "examples_available": is_high_quality
            }
            
            mock_apis.append(mock_api)
        
        return mock_apis
    
    def _evaluate_and_store_api(self, api_info: Dict[str, Any]) -> bool:
        """Evaluate API quality and store if valuable"""
        
        # Calculate quality score
        quality_score = self._calculate_api_quality(api_info)
        api_info["quality_score"] = quality_score
        
        # Only store APIs with reasonable quality
        if quality_score < 0.4:
            return False
        
        # Generate unique ID
        api_id = hashlib.md5(
            f"{api_info['name']}_{api_info['url']}".encode()
        ).hexdigest()
        
        # Check if already exists
        if self._api_exists(api_id):
            return False
        
        # Store in database
        try:
            with sqlite3.connect(self.discovery_db) as conn:
                conn.execute('''
                    INSERT INTO discovered_apis
                    (id, name, url, description, category, quality_score,
                     auth_required, rate_limit, documentation_url, discovered_at,
                     working_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    api_id,
                    api_info.get('name', ''),
                    api_info.get('url', ''),
                    api_info.get('description', ''),
                    api_info.get('category', 'unknown'),
                    quality_score,
                    api_info.get('auth_required', True),
                    api_info.get('rate_limit', 'unknown'),
                    api_info.get('documentation_url', ''),
                    datetime.now().isoformat(),
                    'untested'
                ))
                
                logger.info(f"📚 Stored API: {api_info['name']} (quality: {quality_score:.2f})")
                return True
                
        except Exception as e:
            logger.error(f"Error storing API: {e}")
            return False
    
    def _calculate_api_quality(self, api_info: Dict[str, Any]) -> float:
        """Calculate comprehensive API quality score"""
        score = 0.2  # Base score
        
        description = api_info.get('description', '').lower()
        name = api_info.get('name', '').lower()
        
        # Check for quality indicators in description/name
        for indicator in self.quality_indicators:
            if indicator in description or indicator in name:
                score += 0.05
        
        # Boost for no authentication required
        if not api_info.get('auth_required', True):
            score += 0.3
        
        # Boost for HTTPS support
        if api_info.get('url', '').startswith('https://'):
            score += 0.15
        
        # Boost for CORS support
        if api_info.get('cors_support', False):
            score += 0.1
        
        # Boost for JSON responses
        if api_info.get('json_response', False):
            score += 0.1
        
        # Boost for documentation
        if api_info.get('documentation_url'):
            score += 0.2
        
        # Boost for examples
        if api_info.get('examples_available', False):
            score += 0.1
        
        # Rate limiting considerations
        rate_limit = api_info.get('rate_limit', '').lower()
        if 'unlimited' in rate_limit or '1000' in rate_limit:
            score += 0.15
        elif '100' in rate_limit:
            score += 0.1
        
        return min(1.0, score)
    
    def _api_exists(self, api_id: str) -> bool:
        """Check if API already exists in database"""
        try:
            with sqlite3.connect(self.discovery_db) as conn:
                cursor = conn.execute('SELECT 1 FROM discovered_apis WHERE id = ?', (api_id,))
                return cursor.fetchone() is not None
        except Exception:
            return False
    
    def _log_discovery_session(self, start_time: datetime, end_time: datetime, result: Dict[str, Any]):
        """Log discovery session results"""
        
        session_id = hashlib.md5(start_time.isoformat().encode()).hexdigest()
        
        try:
            with sqlite3.connect(self.discovery_db) as conn:
                conn.execute('''
                    INSERT INTO discovery_sessions
                    (id, session_start, session_end, apis_found, search_queries,
                     success_rate, session_notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    start_time.isoformat(),
                    end_time.isoformat(),
                    result["apis_found"],
                    result["search_queries"],
                    result["success_rate"],
                    json.dumps(result)
                ))
        except Exception as e:
            logger.error(f"Error logging session: {e}")
    
    def get_discovered_apis(self, 
                           category: Optional[str] = None,
                           min_quality: float = 0.5,
                           working_only: bool = False,
                           limit: int = 20) -> List[Dict[str, Any]]:
        """Get discovered APIs by criteria"""
        
        try:
            with sqlite3.connect(self.discovery_db) as conn:
                query = "SELECT * FROM discovered_apis WHERE quality_score >= ?"
                params = [min_quality]
                
                if category:
                    query += " AND category = ?"
                    params.append(category)
                
                if working_only:
                    query += " AND working_status = 'working'"
                
                query += " ORDER BY quality_score DESC, discovered_at DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                
                apis = []
                for row in cursor.fetchall():
                    apis.append({
                        "id": row[0],
                        "name": row[1],
                        "url": row[2],
                        "description": row[3],
                        "category": row[4],
                        "quality_score": row[5],
                        "auth_required": bool(row[6]),
                        "rate_limit": row[7],
                        "documentation_url": row[8],
                        "discovered_at": row[9],
                        "tested": bool(row[10]),
                        "working_status": row[12]
                    })
                
                return apis
                
        except Exception as e:
            logger.error(f"Error retrieving APIs: {e}")
            return []
    
    def test_discovered_apis(self, max_tests: int = 5) -> Dict[str, Any]:
        """Test discovered APIs to verify they're working"""
        
        # Get untested APIs
        untested_apis = self.get_discovered_apis(working_only=False, limit=max_tests)
        untested_apis = [api for api in untested_apis if not api["tested"]]
        
        test_results = {
            "tested_count": 0,
            "working_count": 0,
            "failed_count": 0,
            "results": []
        }
        
        for api in untested_apis[:max_tests]:
            try:
                logger.info(f"🧪 Testing API: {api['name']}")
                
                # Simple HTTP test
                response = requests.get(api['url'], timeout=10)
                
                if response.status_code == 200:
                    working_status = 'working'
                    test_results["working_count"] += 1
                    logger.info(f"✅ API working: {api['name']}")
                else:
                    working_status = 'failed'
                    test_results["failed_count"] += 1
                    logger.warning(f"❌ API failed: {api['name']} (status: {response.status_code})")
                
                # Update database
                with sqlite3.connect(self.discovery_db) as conn:
                    conn.execute('''
                        UPDATE discovered_apis 
                        SET tested = TRUE, 
                            working_status = ?,
                            last_tested = ?,
                            test_results = ?
                        WHERE id = ?
                    ''', (
                        working_status,
                        datetime.now().isoformat(),
                        json.dumps({"status_code": response.status_code}),
                        api['id']
                    ))
                
                test_results["results"].append({
                    "api_name": api['name'],
                    "status": working_status,
                    "status_code": response.status_code
                })
                
                test_results["tested_count"] += 1
                
                # Rate limiting between tests
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error testing API {api['name']}: {e}")
                
                # Mark as failed
                try:
                    with sqlite3.connect(self.discovery_db) as conn:
                        conn.execute('''
                            UPDATE discovered_apis 
                            SET tested = TRUE, 
                                working_status = 'failed',
                                last_tested = ?,
                                test_results = ?
                            WHERE id = ?
                        ''', (
                            datetime.now().isoformat(),
                            json.dumps({"error": str(e)}),
                            api['id']
                        ))
                    
                    test_results["failed_count"] += 1
                    test_results["tested_count"] += 1
                    
                except Exception:
                    pass
        
        return test_results
    
    def get_discovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive discovery statistics"""
        
        try:
            with sqlite3.connect(self.discovery_db) as conn:
                # Total APIs by status
                cursor = conn.execute('''
                    SELECT working_status, COUNT(*) 
                    FROM discovered_apis 
                    GROUP BY working_status
                ''')
                status_counts = dict(cursor.fetchall())
                
                # APIs by category
                cursor = conn.execute('''
                    SELECT category, COUNT(*), AVG(quality_score)
                    FROM discovered_apis 
                    GROUP BY category
                    ORDER BY COUNT(*) DESC
                ''')
                category_stats = {
                    row[0]: {"count": row[1], "avg_quality": row[2]} 
                    for row in cursor.fetchall()
                }
                
                # Recent discovery trend
                cursor = conn.execute('''
                    SELECT DATE(discovered_at) as date, COUNT(*)
                    FROM discovered_apis
                    WHERE discovered_at >= datetime('now', '-7 days')
                    GROUP BY DATE(discovered_at)
                    ORDER BY date
                ''')
                recent_trend = dict(cursor.fetchall())
                
                return {
                    "global_stats": self.discovery_stats,
                    "api_status_counts": status_counts,
                    "category_statistics": category_stats,
                    "recent_discovery_trend": recent_trend,
                    "total_apis": sum(status_counts.values()),
                    "working_apis": status_counts.get('working', 0),
                    "discovery_active": self.search_active
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

# Global instance for easy access
_api_discovery = None

def get_api_discovery_system() -> ContinuousAPIDiscovery:
    """Get or create global API discovery system instance"""
    global _api_discovery
    if _api_discovery is None:
        _api_discovery = ContinuousAPIDiscovery()
    return _api_discovery

if __name__ == "__main__":
    # Test the continuous API discovery system
    print("🔍 Testing Continuous API Discovery System")
    print("=" * 50)
    
    discovery = ContinuousAPIDiscovery()
    
    # Perform a test discovery session
    print("\n🧪 Running test discovery session...")
    session_result = discovery._perform_discovery_session()
    print(f"   APIs found: {session_result['apis_found']}")
    print(f"   Search queries: {session_result['search_queries']}")
    print(f"   Success rate: {session_result['success_rate']:.2f}")
    
    # Get discovered APIs
    print("\n📚 Getting discovered APIs...")
    apis = discovery.get_discovered_apis(min_quality=0.5, limit=5)
    print(f"   Found {len(apis)} APIs with quality >= 0.5")
    
    for api in apis:
        print(f"   • {api['name']} - Quality: {api['quality_score']:.2f}")
    
    # Test APIs
    if apis:
        print("\n🧪 Testing discovered APIs...")
        test_results = discovery.test_discovered_apis(max_tests=2)
        print(f"   Tested: {test_results['tested_count']}")
        print(f"   Working: {test_results['working_count']}")
        print(f"   Failed: {test_results['failed_count']}")
    
    # Get statistics
    print("\n📊 Discovery Statistics:")
    stats = discovery.get_discovery_statistics()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for subkey, subvalue in value.items():
                print(f"     {subkey}: {subvalue}")
        else:
            print(f"   {key}: {value}")
    
    print("\n✅ Continuous API Discovery test complete!")
