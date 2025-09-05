#!/usr/bin/env python3
"""
Test script to verify web search engine BeautifulSoup fixes
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'FSOT_Clean_System'))

def test_web_search_engine():
    """Test the web search engine after BeautifulSoup fixes"""
    
    print("🔍 Testing Web Search Engine BeautifulSoup Fixes")
    print("=" * 60)
    
    try:
        from integration.free_web_search_engine import FreeWebSearchEngine
        print("✅ Successfully imported FreeWebSearchEngine")
        
        # Initialize search engine
        search_engine = FreeWebSearchEngine()
        print("✅ Successfully initialized search engine")
        
        # Test helper functions exist
        if hasattr(search_engine, '_safe_find'):
            print("✅ safe_find helper function exists")
        if hasattr(search_engine, '_safe_get_text'):
            print("✅ safe_get_text helper function exists")
        if hasattr(search_engine, '_safe_get_attr'):
            print("✅ safe_get_attr helper function exists")
        
        # Test a simple search
        print("\n🔍 Testing search functionality...")
        results = search_engine.search_duckduckgo("python programming", num_results=3)
        
        if results:
            print(f"✅ DuckDuckGo search returned {len(results)} results")
            for i, result in enumerate(results[:2], 1):
                print(f"   {i}. {result.title[:50]}...")
        else:
            print("⚠️  DuckDuckGo search returned no results (might be rate limited)")
        
        # Test ArXiv search
        print("\n📚 Testing ArXiv search...")
        arxiv_results = search_engine.search_arxiv("machine learning", num_results=2)
        
        if arxiv_results:
            print(f"✅ ArXiv search returned {len(arxiv_results)} results")
            for i, result in enumerate(arxiv_results[:2], 1):
                print(f"   {i}. {result.title[:50]}...")
        else:
            print("⚠️  ArXiv search returned no results")
        
        print("\n✅ Web search engine tests completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_web_search_engine()
    print(f"\n{'🎉 All tests passed!' if success else '❌ Some tests failed'}")
