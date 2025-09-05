#!/usr/bin/env python3
"""
🌌 FSOT MAST API Integration System
=================================

Advanced astronomical data query system integrating with the Space Telescope Science Institute's
Mikulski Archive for Space Telescopes (MAST) API. This system demonstrates FSOT's capability
to interact with scientific APIs for real-world astronomical research.

Based on official MAST API Documentation: https://mast.stsci.edu/api/v0/

Author: FSOT Neuromorphic AI System
Date: September 5, 2025
Institution: Space Telescope Science Institute Integration
"""

import sys
import os
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import urllib.parse

@dataclass
class MastQuery:
    """
    🔭 MAST Query Configuration Class
    
    Represents a structured query to the MAST API following the official
    MashupRequest format documented by STScI.
    """
    service: str
    params: Dict[str, Any]
    format: str = 'json'
    pagesize: int = 2000
    removenullcolumns: bool = True
    timeout: int = 30
    cachebreaker: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary format for API request."""
        query_dict = {
            'service': self.service,
            'params': self.params,
            'format': self.format,
            'pagesize': self.pagesize,
            'removenullcolumns': self.removenullcolumns,
            'timeout': self.timeout
        }
        
        if self.cachebreaker:
            query_dict['cachebreaker'] = self.cachebreaker
            
        return query_dict

class FSotMastApiClient:
    """
    🌌 FSOT MAST API Client
    
    Advanced client for programmatic interaction with the Space Telescope Science Institute's
    MAST archive system. Provides high-level methods for astronomical data queries.
    """
    
    def __init__(self):
        """Initialize the FSOT MAST API client."""
        self.base_url = "https://mast.stsci.edu/api/v0/invoke"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FSOT-Neuromorphic-AI-System/1.0 (Scientific Research)',
            'Content-Type': 'application/json'
        })
        
        # Query history and results cache
        self.query_history: List[Dict[str, Any]] = []
        self.results_cache: Dict[str, Any] = {}
        
        # Scientific query templates
        self.query_templates = {
            'cone_search': {
                'service': 'Mast.Caom.Cone',
                'description': 'Cone search around RA/Dec coordinates'
            },
            'object_search': {
                'service': 'Mast.Name.Lookup',
                'description': 'Search for astronomical objects by name'
            },
            'mission_search': {
                'service': 'Mast.Caom.Filtered',
                'description': 'Search observations from specific missions'
            }
        }
        
        print("🌌 FSOT MAST API Client Initialized!")
        print(f"🔗 Base URL: {self.base_url}")
        print(f"🛠️ Available Templates: {', '.join(self.query_templates.keys())}")
    
    def create_cone_search(self, ra: float, dec: float, radius: float, 
                          pagesize: int = 2000) -> MastQuery:
        """
        🔭 Create a cone search query for astronomical coordinates.
        
        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees  
            radius: Search radius in degrees
            pagesize: Maximum number of results to return
            
        Returns:
            MastQuery: Configured query object
        """
        return MastQuery(
            service='Mast.Caom.Cone',
            params={
                'ra': ra,
                'dec': dec,
                'radius': radius
            },
            pagesize=pagesize,
            cachebreaker=datetime.now().strftime("%Y-%m-%dT%H%M")
        )
    
    def create_object_search(self, object_name: str) -> MastQuery:
        """
        🌟 Create a search query for astronomical objects by name.
        
        Args:
            object_name: Name of astronomical object (e.g., 'M31', 'NGC1234')
            
        Returns:
            MastQuery: Configured query object
        """
        return MastQuery(
            service='Mast.Name.Lookup',
            params={
                'input': object_name,
                'format': 'json'
            }
        )
    
    def create_mission_search(self, mission: str, instrument: Optional[str] = None,
                            filters: Optional[Dict[str, Any]] = None) -> MastQuery:
        """
        🚀 Create a search query for specific space missions.
        
        Args:
            mission: Mission name (e.g., 'HST', 'JWST', 'Kepler')
            instrument: Specific instrument (optional)
            filters: Additional filtering parameters
            
        Returns:
            MastQuery: Configured query object
        """
        params = {'mission': mission}
        
        if instrument:
            params['instrument'] = instrument
            
        if filters:
            params.update(filters)
            
        return MastQuery(
            service='Mast.Caom.Filtered',
            params=params,
            cachebreaker=datetime.now().strftime("%Y-%m-%dT%H%M")
        )
    
    def execute_query(self, query: MastQuery, method: str = 'POST') -> Dict[str, Any]:
        """
        🌐 Execute a MAST API query and return results.
        
        Args:
            query: MastQuery object with query parameters
            method: HTTP method ('GET' or 'POST')
            
        Returns:
            Dict containing query results and metadata
        """
        
        print(f"\n🚀 Executing MAST API Query...")
        print(f"🔬 Service: {query.service}")
        print(f"📊 Parameters: {query.params}")
        print(f"🌐 Method: {method}")
        
        query_dict = query.to_dict()
        
        try:
            if method.upper() == 'GET':
                # For GET requests, encode the query as URL parameter
                query_json = json.dumps(query_dict)
                url_with_params = f"{self.base_url}?request={urllib.parse.quote(query_json)}"
                response = self.session.get(url_with_params, timeout=query.timeout)
            else:
                # For POST requests, send as form data with proper encoding
                query_json = json.dumps(query_dict)
                response = self.session.post(
                    self.base_url,
                    data={'request': query_json},
                    headers={'Content-Type': 'application/x-www-form-urlencoded'},
                    timeout=query.timeout
                )
            
            print(f"📡 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    results = response.json()
                    
                    # Process and analyze results
                    analysis = self._analyze_results(results, query)
                    
                    # Store in history and cache
                    query_record = {
                        'timestamp': datetime.now().isoformat(),
                        'query': query_dict,
                        'results_count': analysis.get('record_count', 0),
                        'success': True
                    }
                    self.query_history.append(query_record)
                    
                    print(f"✅ Query successful!")
                    print(f"📊 Records returned: {analysis.get('record_count', 0)}")
                    print(f"🔬 Data columns: {analysis.get('column_count', 0)}")
                    
                    return {
                        'success': True,
                        'data': results,
                        'analysis': analysis,
                        'query': query_dict,
                        'metadata': {
                            'timestamp': datetime.now().isoformat(),
                            'response_time': response.elapsed.total_seconds(),
                            'status_code': response.status_code
                        }
                    }
                    
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse JSON response: {e}")
                    return {
                        'success': False,
                        'error': f"JSON parsing error: {e}",
                        'raw_response': response.text
                    }
            else:
                print(f"❌ API request failed with status {response.status_code}")
                print(f"🔍 Response content: {response.text[:500]}...")
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'status_code': response.status_code
                }
                
        except requests.exceptions.Timeout:
            print(f"⏰ Query timed out after {query.timeout} seconds")
            return {
                'success': False,
                'error': 'Request timeout',
                'timeout': query.timeout
            }
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return {
                'success': False,
                'error': f"Network error: {e}"
            }
    
    def _analyze_results(self, results: Dict[str, Any], query: MastQuery) -> Dict[str, Any]:
        """
        📊 Analyze MAST API results and extract metadata.
        
        Args:
            results: Raw API response data
            query: Original query object
            
        Returns:
            Dict containing analysis metadata
        """
        analysis = {
            'record_count': 0,
            'column_count': 0,
            'columns': [],
            'data_types': {},
            'summary': {}
        }
        
        try:
            # Check for data in results
            if 'data' in results and isinstance(results['data'], list):
                data = results['data']
                analysis['record_count'] = len(data)
                
                if data:
                    # Analyze first record to get column information
                    first_record = data[0]
                    if isinstance(first_record, dict):
                        analysis['columns'] = list(first_record.keys())
                        analysis['column_count'] = len(analysis['columns'])
                        
                        # Analyze data types
                        for column in analysis['columns']:
                            value = first_record[column]
                            analysis['data_types'][column] = type(value).__name__
                        
                        # Create summary statistics
                        analysis['summary'] = {
                            'first_record_preview': {k: str(v)[:100] for k, v in first_record.items()},
                            'has_coordinates': any(col in analysis['columns'] for col in ['ra', 'dec', 's_ra', 's_dec']),
                            'has_mission_info': any(col in analysis['columns'] for col in ['mission', 'instrument', 'obs_collection']),
                            'has_temporal_info': any(col in analysis['columns'] for col in ['t_min', 't_max', 'obsdate'])
                        }
            
            # Check for error messages
            if 'msg' in results:
                analysis['api_message'] = results['msg']
            
            if 'status' in results:
                analysis['api_status'] = results['status']
                
        except Exception as e:
            analysis['analysis_error'] = str(e)
        
        return analysis
    
    def search_around_coordinates(self, ra: float, dec: float, radius: float = 0.1) -> Dict[str, Any]:
        """
        🎯 Convenient method to search for observations around specific coordinates.
        
        Args:
            ra: Right Ascension in degrees
            dec: Declination in degrees
            radius: Search radius in degrees (default 0.1)
            
        Returns:
            Dict containing search results
        """
        print(f"\n🔭 Searching for observations around RA={ra}, Dec={dec} (radius={radius}°)")
        
        query = self.create_cone_search(ra, dec, radius)
        return self.execute_query(query)
    
    def search_astronomical_object(self, object_name: str) -> Dict[str, Any]:
        """
        🌟 Search for observations of a named astronomical object.
        
        Args:
            object_name: Name of the astronomical object
            
        Returns:
            Dict containing search results
        """
        print(f"\n🌟 Searching for observations of '{object_name}'")
        
        query = self.create_object_search(object_name)
        return self.execute_query(query)
    
    def get_mission_observations(self, mission: str, limit: int = 100) -> Dict[str, Any]:
        """
        🚀 Get observations from a specific space mission.
        
        Args:
            mission: Mission name (e.g., 'HST', 'JWST', 'Kepler')
            limit: Maximum number of observations to return
            
        Returns:
            Dict containing mission observations
        """
        print(f"\n🚀 Getting observations from mission: {mission}")
        
        query = self.create_mission_search(mission)
        query.pagesize = limit
        return self.execute_query(query)
    
    def get_query_history(self) -> List[Dict[str, Any]]:
        """
        📚 Get history of all queries executed in this session.
        
        Returns:
            List of query records with metadata
        """
        return self.query_history
    
    def save_results(self, results: Dict[str, Any], filename: str) -> bool:
        """
        💾 Save query results to a JSON file.
        
        Args:
            results: Query results to save
            filename: Output filename
            
        Returns:
            bool: Success status
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 Results saved to: {filename}")
            return True
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
            return False

def demonstrate_mast_api():
    """
    🌌 Demonstration of FSOT MAST API integration capabilities.
    
    This function showcases various astronomical queries using the MAST API
    to demonstrate real-world scientific data access.
    """
    
    print("\n" + "="*80)
    print("🌌 FSOT MAST API INTEGRATION DEMONSTRATION")
    print("="*80)
    print("🎯 Mission: Astronomical Data Query Capabilities")
    print("🔭 Target: Space Telescope Science Institute MAST Archive")
    print("🧠 AI System: FSOT Neuromorphic Intelligence")
    print("="*80 + "\n")
    
    # Initialize MAST API client
    client = FSotMastApiClient()
    
    demonstration_results = {
        'session_timestamp': datetime.now().isoformat(),
        'demonstrations': [],
        'summary': {}
    }
    
    # Demonstration 1: Cone search around a famous astronomical object
    print("\n🔭 DEMONSTRATION 1: Cone Search Around Andromeda Galaxy (M31)")
    print("-" * 60)
    
    # M31 (Andromeda Galaxy) coordinates: RA = 10.684708°, Dec = 41.268750°
    andromeda_results = client.search_around_coordinates(10.684708, 41.268750, 0.5)
    
    if andromeda_results['success']:
        demo1_summary = {
            'demo_name': 'Andromeda Galaxy Cone Search',
            'coordinates': {'ra': 10.684708, 'dec': 41.268750, 'radius': 0.5},
            'results_count': andromeda_results['analysis']['record_count'],
            'columns_found': andromeda_results['analysis']['column_count'],
            'success': True
        }
        print(f"✅ Found {demo1_summary['results_count']} observations around Andromeda!")
        
        # Save detailed results
        client.save_results(andromeda_results, f"MAST_Andromeda_Search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    else:
        demo1_summary = {
            'demo_name': 'Andromeda Galaxy Cone Search',
            'success': False,
            'error': andromeda_results.get('error', 'Unknown error')
        }
        print(f"❌ Andromeda search failed: {demo1_summary['error']}")
    
    demonstration_results['demonstrations'].append(demo1_summary)
    
    # Demonstration 2: Search for Hubble observations
    print("\n🚀 DEMONSTRATION 2: Hubble Space Telescope Observations")
    print("-" * 60)
    
    hubble_results = client.get_mission_observations('HST', limit=50)
    
    if hubble_results['success']:
        demo2_summary = {
            'demo_name': 'Hubble Space Telescope Observations',
            'mission': 'HST',
            'results_count': hubble_results['analysis']['record_count'],
            'columns_found': hubble_results['analysis']['column_count'],
            'success': True
        }
        print(f"✅ Found {demo2_summary['results_count']} Hubble observations!")
        
        # Save detailed results
        client.save_results(hubble_results, f"MAST_Hubble_Observations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    else:
        demo2_summary = {
            'demo_name': 'Hubble Space Telescope Observations',
            'success': False,
            'error': hubble_results.get('error', 'Unknown error')
        }
        print(f"❌ Hubble search failed: {demo2_summary['error']}")
    
    demonstration_results['demonstrations'].append(demo2_summary)
    
    # Demonstration 3: Famous object search
    print("\n🌟 DEMONSTRATION 3: Famous Astronomical Object Lookup")
    print("-" * 60)
    
    object_results = client.search_astronomical_object('Orion Nebula')
    
    if object_results['success']:
        demo3_summary = {
            'demo_name': 'Orion Nebula Object Search',
            'object_name': 'Orion Nebula',
            'results_count': object_results['analysis']['record_count'],
            'success': True
        }
        print(f"✅ Found information for Orion Nebula!")
        
        # Save detailed results
        client.save_results(object_results, f"MAST_Orion_Search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    else:
        demo3_summary = {
            'demo_name': 'Orion Nebula Object Search',
            'success': False,
            'error': object_results.get('error', 'Unknown error')
        }
        print(f"❌ Orion Nebula search failed: {demo3_summary['error']}")
    
    demonstration_results['demonstrations'].append(demo3_summary)
    
    # Session summary
    successful_demos = sum(1 for demo in demonstration_results['demonstrations'] if demo['success'])
    total_demos = len(demonstration_results['demonstrations'])
    
    demonstration_results['summary'] = {
        'total_demonstrations': total_demos,
        'successful_demonstrations': successful_demos,
        'success_rate': f"{(successful_demos/total_demos)*100:.1f}%",
        'query_history_count': len(client.get_query_history())
    }
    
    print("\n" + "="*80)
    print("🏆 FSOT MAST API INTEGRATION SUMMARY")
    print("="*80)
    print(f"📊 Total Demonstrations: {total_demos}")
    print(f"✅ Successful Queries: {successful_demos}")
    print(f"📈 Success Rate: {demonstration_results['summary']['success_rate']}")
    print(f"📚 Query History: {demonstration_results['summary']['query_history_count']} queries")
    
    if successful_demos > 0:
        print("\n🌟 FSOT MAST API Integration: SUCCESSFUL!")
        print("🔭 The FSOT system can now access astronomical data from space telescopes!")
        print("🧠 Autonomous scientific research capabilities unlocked!")
    else:
        print("\n⚠️ Some queries may have failed due to network or API limitations")
        print("🔧 MAST API client is ready for future astronomical research")
    
    print("="*80)
    
    # Save comprehensive session results
    session_filename = f"FSOT_MAST_API_Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(session_filename, 'w', encoding='utf-8') as f:
            json.dump(demonstration_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"📄 Session results saved: {session_filename}")
    except Exception as e:
        print(f"⚠️ Failed to save session results: {e}")
    
    return demonstration_results

def main():
    """
    🌌 Main execution function for FSOT MAST API integration.
    """
    print("🚀 FSOT MAST API Integration System")
    print("🎯 Demonstrating autonomous astronomical data access capabilities")
    
    # Execute demonstrations
    results = demonstrate_mast_api()
    
    print("\n🌌 FSOT MAST API Integration completed! 🔭✨")
    print("🧠 The FSOT system now has direct access to space telescope archives!")

if __name__ == "__main__":
    main()
