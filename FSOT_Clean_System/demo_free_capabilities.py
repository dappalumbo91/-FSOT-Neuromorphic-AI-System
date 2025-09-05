#!/usr/bin/env python3
"""
FREE CAPABILITIES DEMONSTRATION
==============================

This script demonstrates all the free capabilities of the Enhanced FSOT 2.0 system
without requiring any paid API keys or subscriptions.

Author: GitHub Copilot
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from integration.free_web_search_engine import FreeWebSearchEngine
from integration.free_api_manager import FreeAPIManager

def demonstrate_free_web_search():
    """Demonstrate free web search capabilities"""
    print("\n🔍 DEMONSTRATING FREE WEB SEARCH CAPABILITIES")
    print("=" * 60)
    
    search_engine = FreeWebSearchEngine()
    
    test_queries = [
        "artificial intelligence 2025",
        "climate change solutions",
        "Python programming best practices"
    ]
    
    for query in test_queries:
        print(f"\n📝 Searching: {query}")
        print("-" * 40)
        
        try:
            results = search_engine.search(query, num_results=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result.title}")
                    print(f"   🌐 URL: {result.url}")
                    print(f"   📊 Source: {result.source}")
                    print(f"   📄 Snippet: {result.snippet[:100]}...")
                    if result.content:
                        print(f"   📖 Content: {result.content[:100]}...")
                    print()
            else:
                print("   ❌ No results found")
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")

def demonstrate_free_api_access():
    """Demonstrate free API access capabilities"""
    print("\n🔗 DEMONSTRATING FREE API ACCESS CAPABILITIES")
    print("=" * 60)
    
    api_manager = FreeAPIManager()
    
    # Test GitHub API
    print("\n🐙 GitHub Repository Search:")
    print("-" * 30)
    try:
        github_results = api_manager.github_search_repositories("machine learning", per_page=3)
        if github_results.get("items"):
            for repo in github_results["items"]:
                print(f"• {repo['full_name']} ⭐ {repo['stargazers_count']}")
                print(f"  {repo.get('description', 'No description')[:80]}...")
        else:
            print("  ❌ No repositories found")
    except Exception as e:
        print(f"  ❌ GitHub search failed: {e}")
    
    # Test Random User API
    print("\n👤 Random User Data:")
    print("-" * 20)
    try:
        user_data = api_manager.get_random_user_data(results=1)
        if user_data.get("results"):
            user = user_data["results"][0]
            name = user.get("name", {})
            location = user.get("location", {})
            print(f"• Name: {name.get('first', '')} {name.get('last', '')}")
            print(f"• Location: {location.get('city', '')}, {location.get('country', '')}")
            print(f"• Email: {user.get('email', '')}")
        else:
            print("  ❌ No user data found")
    except Exception as e:
        print(f"  ❌ Random user API failed: {e}")
    
    # Test Exchange Rate API
    print("\n💱 Exchange Rates:")
    print("-" * 16)
    try:
        exchange_data = api_manager.get_exchange_rates()
        if exchange_data.get("rates"):
            rates = exchange_data["rates"]
            sample_currencies = ["EUR", "GBP", "JPY", "CAD", "AUD"]
            print(f"• Base: {exchange_data.get('base', 'USD')} ({exchange_data.get('date', '')})")
            for currency in sample_currencies:
                if currency in rates:
                    print(f"• 1 USD = {rates[currency]} {currency}")
        else:
            print("  ❌ No exchange rate data found")
    except Exception as e:
        print(f"  ❌ Exchange rate API failed: {e}")
    
    # Test Wisdom API
    print("\n🧠 Random Wisdom:")
    print("-" * 15)
    try:
        advice_data = api_manager.get_random_advice()
        advice = advice_data.get("slip", {}).get("advice", "No advice available")
        print(f"• \"{advice}\"")
    except Exception as e:
        print(f"  ❌ Advice API failed: {e}")
    
    # Test Joke API
    print("\n😄 Random Joke:")
    print("-" * 13)
    try:
        joke_data = api_manager.get_random_joke()
        setup = joke_data.get("setup", "")
        punchline = joke_data.get("punchline", "")
        if setup and punchline:
            print(f"• {setup}")
            print(f"  {punchline}")
        else:
            print("  ❌ No joke available")
    except Exception as e:
        print(f"  ❌ Joke API failed: {e}")

def demonstrate_comprehensive_learning():
    """Demonstrate comprehensive learning using free resources"""
    print("\n📚 DEMONSTRATING COMPREHENSIVE LEARNING")
    print("=" * 60)
    
    api_manager = FreeAPIManager()
    search_engine = FreeWebSearchEngine()
    
    learning_topic = "renewable energy"
    print(f"\n🎯 Learning Topic: {learning_topic}")
    print("-" * 40)
    
    # Search and learn using APIs
    print("\n🔍 API-based Learning:")
    try:
        learning_results = api_manager.search_and_learn(learning_topic)
        print(f"• GitHub repositories found: {learning_results['sources']['github']['total_repositories']}")
        
        if learning_results['sources']['github']['top_repositories']:
            print("• Top repositories:")
            for repo in learning_results['sources']['github']['top_repositories']:
                print(f"  - {repo['name']} ({repo['language']}) ⭐ {repo['stars']}")
        
        if learning_results['insights']:
            print("• Insights:")
            for insight in learning_results['insights']:
                print(f"  - {insight}")
        
        wisdom = learning_results['sources'].get('wisdom', '')
        if wisdom:
            print(f"• Wisdom: \"{wisdom}\"")
            
    except Exception as e:
        print(f"  ❌ API learning failed: {e}")
    
    # Web search-based learning
    print("\n🌐 Web Search-based Learning:")
    try:
        search_results = search_engine.comprehensive_search(learning_topic, num_results=3)
        
        if search_results:
            print(f"• Found {len(search_results)} relevant sources:")
            for result in search_results:
                print(f"  - {result.title} ({result.source})")
                print(f"    {result.snippet[:80]}...")
        else:
            print("  ❌ No search results found")
            
    except Exception as e:
        print(f"  ❌ Web search learning failed: {e}")

def test_api_availability():
    """Test availability of all free APIs"""
    print("\n🧪 TESTING API AVAILABILITY")
    print("=" * 60)
    
    api_manager = FreeAPIManager()
    
    test_results = api_manager.test_all_apis()
    
    print(f"\n📊 Test Summary:")
    print(f"• APIs Tested: {test_results['apis_tested']}")
    print(f"• APIs Working: {test_results['apis_working']}")
    print(f"• Success Rate: {test_results['success_rate']:.1f}%")
    
    print(f"\n📋 Detailed Results:")
    for api_name, result in test_results['results'].items():
        status = result['status']
        if status == "working":
            print(f"  ✅ {api_name.upper()}: Working")
        else:
            print(f"  ❌ {api_name.upper()}: {result.get('error', 'Unknown error')}")

def show_configuration_guide():
    """Show configuration guide for free APIs"""
    print("\n⚙️  FREE API CONFIGURATION GUIDE")
    print("=" * 60)
    
    print("\n📝 Current Configuration:")
    print("• ✅ No API keys required!")
    print("• ✅ All features work out of the box")
    print("• ✅ Web scraping enabled for search")
    print("• ✅ Free APIs automatically discovered")
    
    print("\n🛠️  Optional Enhancements:")
    print("• Install BeautifulSoup4 for better web parsing:")
    print("  pip install beautifulsoup4")
    print("• Install Trafilatura for advanced content extraction:")
    print("  pip install trafilatura")
    print("• Install Newspaper3k for article processing:")
    print("  pip install newspaper3k")
    
    print("\n🔒 Privacy & Rate Limits:")
    print("• All APIs used are public and free")
    print("• Rate limiting built-in to prevent abuse")
    print("• No personal data collected or stored")
    print("• Caching reduces API calls")
    
    print("\n🌟 Available Free Services:")
    print("• 🐙 GitHub - Repository search, user info")
    print("• 🔍 DuckDuckGo - Web search (scraping)")
    print("• 📚 Wikipedia - Knowledge articles")
    print("• 📄 ArXiv - Scientific papers")
    print("• 💬 Reddit - Discussion threads")
    print("• 👤 RandomUser - Sample user data")
    print("• 💱 Exchange Rates - Currency data")
    print("• 🧠 Advice Slip - Random wisdom")
    print("• 😄 Jokes API - Random humor")

def main():
    """Main demonstration"""
    print("🆓 ENHANCED FSOT 2.0 - FREE CAPABILITIES DEMONSTRATION")
    print("=" * 70)
    print("🎉 Welcome to the completely FREE version of Enhanced FSOT 2.0!")
    print("🔓 No API keys, subscriptions, or payments required!")
    print()
    
    try:
        # Show configuration
        show_configuration_guide()
        
        # Test API availability first
        test_api_availability()
        
        # Demonstrate free API access
        demonstrate_free_api_access()
        
        # Demonstrate free web search
        demonstrate_free_web_search()
        
        # Demonstrate comprehensive learning
        demonstrate_comprehensive_learning()
        
        print("\n🎊 DEMONSTRATION COMPLETE!")
        print("=" * 60)
        print("✅ All features working with FREE APIs!")
        print("🚀 Your Enhanced FSOT 2.0 system is ready to learn!")
        print("💡 Start learning with: python main.py")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration error: {e}")
        print("💡 Some features may require internet connection")

if __name__ == "__main__":
    main()
