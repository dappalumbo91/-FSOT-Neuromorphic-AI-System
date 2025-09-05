#!/usr/bin/env python3
"""
FSOT Web Exploration - Ohara.ai Profile Analysis
================================================

Using the FSOT Autonomous Web Crawler to explore and analyze the Ohara.ai profile.
"""

from fsot_autonomous_web_crawler import FSotAutonomousWebCrawler
import time
import json
from datetime import datetime

def explore_ohara_ai_profile():
    """
    Explore the Ohara.ai profile using FSOT web automation capabilities.
    """
    print("🌐 FSOT Autonomous Web Crawler - Ohara.ai Exploration")
    print("=" * 60)
    
    crawler = FSotAutonomousWebCrawler()
    
    try:
        # Initialize the browser
        print("🚀 Initializing browser...")
        crawler.initialize_browser()
        print("✅ Browser initialized successfully")
        
        # Navigate to the Ohara.ai profile
        url = "https://ohara.ai/users/Dappalumbo91"
        print(f"🔍 Navigating to: {url}")
        
        if crawler.navigate_to_url(url):
            print("✅ Successfully navigated to Ohara.ai profile")
            
            # Give the page time to load
            print("⏳ Waiting for page to fully load...")
            time.sleep(5)
            
            # Analyze the page content
            print("📊 Analyzing page content...")
            page_data = crawler.explore_page_intelligently()
            
            # Display comprehensive analysis
            print("\n🎯 OHARA.AI PROFILE ANALYSIS RESULTS:")
            print("=" * 50)
            
            # Basic page information
            title = page_data.get("title", "Unknown")
            print(f"📄 Page Title: {title}")
            
            # Content analysis
            links = page_data.get("links_found", [])
            forms = page_data.get("forms_found", [])
            images = page_data.get("images_found", [])
            interactive = page_data.get("interactive_elements", [])
            
            print(f"🔗 Links Found: {len(links)}")
            print(f"📝 Forms Found: {len(forms)}")
            print(f"🖼️  Images Found: {len(images)}")
            print(f"⚡ Interactive Elements: {len(interactive)}")
            
            # Show sample links
            if links:
                print(f"\n🔗 Sample Navigation Links:")
                for i, link in enumerate(links[:7]):
                    link_text = link.get("text", "").strip()
                    link_url = link.get("url", "")
                    if link_text and link_url:
                        print(f"   {i+1}. {link_text} -> {link_url[:50]}...")
            
            # Show interactive elements
            if interactive:
                print(f"\n⚡ Interactive Elements Detected:")
                for i, element in enumerate(interactive[:5]):
                    elem_type = element.get("type", "")
                    elem_text = element.get("text", element.get("placeholder", "N/A"))
                    print(f"   {i+1}. {elem_type} - {elem_text}")
            
            # Show forms if any
            if forms:
                print(f"\n📝 Forms Detected:")
                for i, form in enumerate(forms[:3]):
                    action = form.get("action", "")
                    method = form.get("method", "GET")
                    inputs = form.get("inputs", [])
                    print(f"   {i+1}. Method: {method}, Action: {action}, Inputs: {len(inputs)}")
            
            # Extract and show page content preview
            text_content = page_data.get("text_content", "")
            if text_content:
                print(f"\n📝 Page Content Preview:")
                preview = text_content[:300] + "..." if len(text_content) > 300 else text_content
                print(f"   {preview}")
            
            # Perform some intelligent exploration
            print(f"\n🧠 INTELLIGENT EXPLORATION:")
            print("🔍 Looking for profile-related elements...")
            
            # Look for profile-specific content
            profile_indicators = ["profile", "user", "dashboard", "settings", "account"]
            found_profile_elements = []
            
            for link in links:
                link_text = link.get("text", "").lower()
                link_url = link.get("url", "").lower()
                for indicator in profile_indicators:
                    if indicator in link_text or indicator in link_url:
                        found_profile_elements.append(link)
                        break
            
            if found_profile_elements:
                print(f"✅ Found {len(found_profile_elements)} profile-related elements:")
                for elem in found_profile_elements[:3]:
                    print(f"   • {elem.get('text', '')} -> {elem.get('url', '')[:50]}...")
            
            # Check for AI/ML related content
            ai_keywords = ["ai", "ml", "machine learning", "neural", "model", "algorithm"]
            ai_content_found = []
            
            for keyword in ai_keywords:
                if keyword.lower() in text_content.lower():
                    ai_content_found.append(keyword)
            
            if ai_content_found:
                print(f"🤖 AI/ML Related Content Detected: {', '.join(ai_content_found)}")
            
            # Save detailed analysis
            analysis_report = {
                "ohara_ai_exploration": {
                    "timestamp": datetime.now().isoformat(),
                    "url_explored": url,
                    "page_title": title,
                    "content_analysis": {
                        "total_links": len(links),
                        "total_forms": len(forms),
                        "total_images": len(images),
                        "total_interactive_elements": len(interactive),
                        "text_content_length": len(text_content)
                    },
                    "profile_elements_found": len(found_profile_elements),
                    "ai_content_keywords": ai_content_found,
                    "exploration_summary": "Successfully analyzed Ohara.ai profile using FSOT autonomous web crawler"
                },
                "detailed_page_data": page_data
            }
            
            # Save report
            report_filename = f"Ohara_AI_Profile_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_filename, 'w') as f:
                json.dump(analysis_report, f, indent=2)
            
            print(f"\n📊 Detailed analysis saved to: {report_filename}")
            
            print(f"\n🎉 OHARA.AI EXPLORATION COMPLETE!")
            print(f"🧠 FSOT AI successfully analyzed the profile using autonomous web crawling capabilities!")
            
        else:
            print("❌ Failed to navigate to the Ohara.ai profile page")
            print("🔍 This might be due to network issues or site accessibility")
            
    except Exception as e:
        print(f"❌ Error during web exploration: {str(e)}")
        print("🛠️  The FSOT web crawler encountered an issue during exploration")
        
    finally:
        # Clean up browser resources
        try:
            crawler.cleanup()
            print("🧹 Browser cleanup completed successfully")
        except Exception as cleanup_error:
            print(f"⚠️  Cleanup warning: {str(cleanup_error)}")

if __name__ == "__main__":
    explore_ohara_ai_profile()
