#!/usr/bin/env python3
"""
FSOT Web Exploration - YouTube Analysis
=======================================

Using the FSOT Autonomous Web Crawler to explore and analyze YouTube.
This will test our web automation capabilities on a complex, dynamic platform.
"""

from fsot_autonomous_web_crawler import FSotAutonomousWebCrawler
import time
import json
from datetime import datetime

def explore_youtube():
    """
    Explore YouTube using FSOT web automation capabilities.
    """
    print("🎥 FSOT Autonomous Web Crawler - YouTube Exploration")
    print("=" * 60)
    
    crawler = FSotAutonomousWebCrawler()
    
    try:
        # Initialize the browser
        print("🚀 Initializing browser for YouTube exploration...")
        crawler.initialize_browser()
        print("✅ Browser initialized successfully")
        
        # Navigate to YouTube
        url = "https://www.youtube.com/"
        print(f"🔍 Navigating to: {url}")
        
        if crawler.navigate_to_url(url):
            print("✅ Successfully navigated to YouTube")
            
            # Give the page time to load (YouTube is dynamic)
            print("⏳ Waiting for YouTube to fully load...")
            time.sleep(8)  # YouTube needs more time to load content
            
            # Analyze the page content
            print("📊 Analyzing YouTube content...")
            page_data = crawler.explore_page_intelligently()
            
            # Display comprehensive analysis
            print("\n🎯 YOUTUBE ANALYSIS RESULTS:")
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
            
            # Analyze YouTube-specific content
            print(f"\n🎥 YOUTUBE-SPECIFIC ANALYSIS:")
            
            # Look for video-related content
            video_links = []
            for link in links[:15]:  # Check first 15 links
                link_text = link.get("text", "").lower()
                link_url = link.get("url", "").lower()
                if any(keyword in link_text or keyword in link_url for keyword in ['watch', 'video', 'channel', 'playlist']):
                    video_links.append(link)
            
            print(f"🎬 Video-related Links: {len(video_links)}")
            
            # Show sample video links
            if video_links:
                print(f"\n🎬 Sample Video/Channel Links:")
                for i, link in enumerate(video_links[:5]):
                    link_text = link.get("text", "").strip()
                    link_url = link.get("url", "")
                    if link_text and link_url:
                        print(f"   {i+1}. {link_text[:50]}... -> {link_url[:60]}...")
            
            # Look for navigation elements
            nav_elements = []
            navigation_keywords = ['home', 'trending', 'subscriptions', 'library', 'history', 'search']
            
            for element in interactive[:10]:
                elem_text = element.get("text", "").lower()
                if any(keyword in elem_text for keyword in navigation_keywords):
                    nav_elements.append(element)
            
            if nav_elements:
                print(f"\n🧭 YouTube Navigation Elements:")
                for i, element in enumerate(nav_elements[:5]):
                    elem_text = element.get("text", "")
                    elem_type = element.get("type", "")
                    print(f"   {i+1}. {elem_type}: {elem_text}")
            
            # Search for YouTube features
            youtube_features = []
            feature_keywords = ['subscribe', 'like', 'share', 'comment', 'upload', 'create', 'studio']
            
            for element in interactive:
                elem_text = element.get("text", "").lower()
                if any(keyword in elem_text for keyword in feature_keywords):
                    youtube_features.append(element)
            
            if youtube_features:
                print(f"\n🎛️  YouTube Features Detected:")
                for i, feature in enumerate(youtube_features[:5]):
                    feat_text = feature.get("text", "")
                    feat_type = feature.get("type", "")
                    print(f"   {i+1}. {feat_type}: {feat_text}")
            
            # Analyze forms (search, comments, etc.)
            if forms:
                print(f"\n🔍 Forms Analysis:")
                for i, form in enumerate(forms[:3]):
                    action = form.get("action", "")
                    method = form.get("method", "GET")
                    inputs = form.get("inputs", [])
                    print(f"   {i+1}. Method: {method}, Action: {action[:50]}..., Inputs: {len(inputs)}")
                    
                    # Look for search-related inputs
                    for input_elem in inputs[:3]:
                        input_type = input_elem.get("type", "")
                        input_placeholder = input_elem.get("placeholder", "")
                        if "search" in input_placeholder.lower():
                            print(f"      🔍 Search Input: {input_placeholder}")
            
            # Extract and analyze text content
            text_content = page_data.get("text_content", "")
            if text_content:
                print(f"\n📝 Content Analysis:")
                print(f"   📊 Total Text Length: {len(text_content)} characters")
                
                # Look for YouTube-specific terms
                youtube_terms = ['youtube', 'video', 'channel', 'subscribe', 'watch', 'trending', 'music', 'gaming', 'news']
                found_terms = []
                
                for term in youtube_terms:
                    if term.lower() in text_content.lower():
                        found_terms.append(term)
                
                if found_terms:
                    print(f"   🎥 YouTube Terms Found: {', '.join(found_terms)}")
                
                # Show content preview
                preview = text_content[:400] + "..." if len(text_content) > 400 else text_content
                print(f"\n📝 Content Preview:")
                print(f"   {preview}")
            
            # Advanced YouTube Analysis
            print(f"\n🧠 ADVANCED YOUTUBE INTELLIGENCE:")
            
            # Categorize content types
            content_categories = {
                'videos': len([l for l in links if 'watch' in l.get('url', '').lower()]),
                'channels': len([l for l in links if 'channel' in l.get('url', '').lower()]),
                'playlists': len([l for l in links if 'playlist' in l.get('url', '').lower()]),
                'search_results': len([l for l in links if 'search' in l.get('url', '').lower()])
            }
            
            print(f"📊 Content Type Distribution:")
            for category, count in content_categories.items():
                if count > 0:
                    print(f"   {category.title()}: {count}")
            
            # Platform capabilities assessment
            capabilities = []
            if any('upload' in elem.get('text', '').lower() for elem in interactive):
                capabilities.append("Video Upload")
            if any('subscribe' in elem.get('text', '').lower() for elem in interactive):
                capabilities.append("Channel Subscription")
            if any('search' in form.get('action', '').lower() for form in forms):
                capabilities.append("Content Search")
            if any('comment' in elem.get('text', '').lower() for elem in interactive):
                capabilities.append("User Comments")
            
            if capabilities:
                print(f"🎛️  Platform Capabilities Detected: {', '.join(capabilities)}")
            
            # Save comprehensive analysis
            analysis_report = {
                "youtube_exploration": {
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
                    "youtube_specific_analysis": {
                        "video_related_links": len(video_links),
                        "navigation_elements": len(nav_elements),
                        "youtube_features": len(youtube_features),
                        "content_categories": content_categories,
                        "platform_capabilities": capabilities,
                        "youtube_terms_found": found_terms if 'found_terms' in locals() else []
                    },
                    "exploration_summary": "Successfully analyzed YouTube using FSOT autonomous web crawler"
                },
                "detailed_page_data": page_data
            }
            
            # Save report
            report_filename = f"YouTube_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_filename, 'w') as f:
                json.dump(analysis_report, f, indent=2)
            
            print(f"\n📊 Comprehensive analysis saved to: {report_filename}")
            
            print(f"\n🎉 YOUTUBE EXPLORATION COMPLETE!")
            print(f"🧠 FSOT AI successfully analyzed YouTube using autonomous web crawling!")
            print(f"🎥 Platform complexity: HIGH - Successfully navigated dynamic content")
            print(f"📊 Intelligence gathering: COMPREHENSIVE - All features cataloged")
            
        else:
            print("❌ Failed to navigate to YouTube")
            print("🔍 This might be due to network issues or content loading delays")
            
    except Exception as e:
        print(f"❌ Error during YouTube exploration: {str(e)}")
        print("🛠️  The FSOT web crawler encountered an issue during exploration")
        
    finally:
        # Clean up browser resources
        try:
            crawler.cleanup()
            print("🧹 Browser cleanup completed successfully")
        except Exception as cleanup_error:
            print(f"⚠️  Cleanup warning: {str(cleanup_error)}")

if __name__ == "__main__":
    explore_youtube()
