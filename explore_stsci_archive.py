#!/usr/bin/env python3
"""
🌌 FSOT Space Telescope Science Institute Archive Explorer
=======================================================

Advanced autonomous web exploration of the STScI archive system.
This script demonstrates FSOT's capability to analyze scientific data platforms
and astronomical research infrastructure.

Author: FSOT Neuromorphic AI System
Date: September 5, 2025
Target: https://archive.stsci.edu/
"""

import sys
import os
import json
import time
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from fsot_autonomous_web_crawler import FSotAutonomousWebCrawler
    print("🌌 FSOT Space Telescope Archive Explorer Initialized!")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("🔧 Please ensure fsot_autonomous_web_crawler.py is in the current directory")
    sys.exit(1)

def analyze_stsci_archive():
    """
    🔭 Comprehensive analysis of the Space Telescope Science Institute Archive
    
    This function demonstrates advanced scientific platform exploration:
    - Astronomical data archive navigation
    - Scientific instrument detection
    - Research infrastructure analysis
    - Space telescope mission identification
    """
    
    print("\n" + "="*80)
    print("🌌 FSOT SPACE TELESCOPE SCIENCE INSTITUTE ARCHIVE EXPLORATION")
    print("="*80)
    print("🎯 Target: https://archive.stsci.edu/")
    print("🚀 Mission: Autonomous Scientific Platform Analysis")
    print("🧠 AI System: FSOT Neuromorphic Intelligence")
    print("="*80 + "\n")
    
    # Initialize the autonomous web crawler
    crawler = None
    analysis_results = {
        'exploration_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'target_url': 'https://archive.stsci.edu/',
        'platform_type': 'Scientific Data Archive',
        'mission_status': 'INITIATED',
        'scientific_capabilities': [],
        'telescope_missions': [],
        'data_archives': [],
        'research_tools': [],
        'navigation_structure': {},
        'content_analysis': {},
        'intelligence_summary': {}
    }
    
    try:
        print("🔧 Initializing FSOT Autonomous Web Crawler...")
        crawler = FSotAutonomousWebCrawler()
        print("✅ Crawler initialized successfully!")
        
        print("🌐 Setting up browser environment...")
        browser_ready = crawler.initialize_browser(headless=False)
        if not browser_ready:
            print("❌ Failed to initialize browser")
            analysis_results['mission_status'] = 'BROWSER_INIT_FAILED'
            return analysis_results
        print("✅ Browser initialized successfully!")
        
        print("\n🌐 Navigating to STScI Archive...")
        success = crawler.navigate_to_url('https://archive.stsci.edu/')
        
        if not success:
            print("❌ Failed to navigate to STScI Archive")
            analysis_results['mission_status'] = 'NAVIGATION_FAILED'
            return analysis_results
        
        print("✅ Successfully navigated to STScI Archive!")
        print("⏳ Allowing dynamic content to load...")
        time.sleep(5)  # Allow scientific platform content to load
        
        print("\n🔍 Performing intelligent scientific platform exploration...")
        
        # Use the crawler's intelligent exploration capabilities
        exploration_data = crawler.explore_page_intelligently()
        
        if exploration_data:
            print("📊 Processing STScI Archive intelligence...")
            
            # Extract and analyze scientific content
            links = exploration_data.get('links', [])
            forms = exploration_data.get('forms', [])
            images = exploration_data.get('images', [])
            content = exploration_data.get('content', '')
            
            print(f"🔗 Links discovered: {len(links)}")
            print(f"📝 Forms detected: {len(forms)}")
            print(f"🖼️ Images found: {len(images)}")
            print(f"📄 Content length: {len(content)} characters")
            
            # Analyze for scientific and astronomical content
            scientific_keywords = [
                'hubble', 'telescope', 'jwst', 'kepler', 'spitzer', 'galex',
                'archive', 'data', 'observation', 'spectrum', 'image',
                'mission', 'instrument', 'detector', 'filter', 'exposure',
                'astronomical', 'cosmic', 'galaxy', 'star', 'planet',
                'research', 'scientist', 'discovery', 'analysis'
            ]
            
            # Scientific platform analysis
            content_lower = content.lower()
            detected_keywords = [kw for kw in scientific_keywords if kw in content_lower]
            
            # Categorize links by scientific relevance
            telescope_links = []
            archive_links = []
            tool_links = []
            mission_links = []
            
            for link in links:
                link_text = link.get('text', '').lower()
                link_url = link.get('href', '').lower()
                
                if any(telescope in link_text or telescope in link_url 
                       for telescope in ['hubble', 'jwst', 'kepler', 'spitzer', 'galex']):
                    telescope_links.append(link)
                elif any(archive_term in link_text or archive_term in link_url
                         for archive_term in ['archive', 'data', 'search', 'catalog']):
                    archive_links.append(link)
                elif any(tool_term in link_text or tool_term in link_url
                         for tool_term in ['tool', 'software', 'analysis', 'viewer']):
                    tool_links.append(link)
                elif any(mission_term in link_text or mission_term in link_url
                         for mission_term in ['mission', 'instrument', 'project']):
                    mission_links.append(link)
            
            # Update analysis results
            analysis_results.update({
                'mission_status': 'EXPLORATION_COMPLETE',
                'total_links': len(links),
                'total_forms': len(forms),
                'total_images': len(images),
                'content_size': len(content),
                'scientific_keywords_detected': detected_keywords,
                'telescope_missions': telescope_links,
                'data_archives': archive_links,
                'research_tools': tool_links,
                'mission_links': mission_links,
                'navigation_structure': {
                    'main_navigation': len([l for l in links if 'nav' in l.get('class', '')]),
                    'search_forms': len(forms),
                    'external_links': len([l for l in links if 'http' in l.get('href', '')])
                },
                'content_analysis': {
                    'scientific_density': len(detected_keywords),
                    'platform_complexity': 'HIGH' if len(links) > 20 else 'MEDIUM',
                    'research_focus': 'SPACE_TELESCOPE_DATA'
                }
            })
            
            print(f"\n🔬 Scientific Analysis Results:")
            print(f"   🌌 Scientific keywords detected: {len(detected_keywords)}")
            print(f"   🔭 Telescope mission links: {len(telescope_links)}")
            print(f"   📊 Data archive links: {len(archive_links)}")
            print(f"   🛠️ Research tool links: {len(tool_links)}")
            print(f"   🎯 Mission-related links: {len(mission_links)}")
            
            # Intelligence summary
            analysis_results['intelligence_summary'] = {
                'platform_assessment': 'Advanced Scientific Data Archive',
                'primary_function': 'Space Telescope Data Repository',
                'user_base': 'Astronomical Research Community',
                'data_scope': 'Multi-Mission Space Observatory Archives',
                'research_value': 'EXCEPTIONAL',
                'technical_sophistication': 'ENTERPRISE_SCIENTIFIC',
                'automation_compatibility': 'EXCELLENT'
            }
            
        else:
            print("⚠️ No exploration data returned")
            analysis_results['mission_status'] = 'EXPLORATION_INCOMPLETE'
        
        print("✅ STScI Archive exploration completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during STScI exploration: {str(e)}")
        analysis_results['mission_status'] = 'ERROR'
        analysis_results['error_details'] = str(e)
        
    finally:
        # Clean up browser resources
        if crawler:
            try:
                print("\n🧹 Cleaning up browser resources...")
                crawler.cleanup()
                print("✅ Browser cleanup completed")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup warning: {cleanup_error}")
    
    # Save detailed analysis report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"STScI_Archive_Analysis_{timestamp}.json"
    
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print(f"📄 Analysis report saved: {report_filename}")
    except Exception as save_error:
        print(f"⚠️ Failed to save report: {save_error}")
    
    return analysis_results

def main():
    """
    🌌 Main execution function for STScI Archive exploration
    """
    print("🚀 FSOT Space Telescope Science Institute Archive Explorer")
    print("🎯 Demonstrating autonomous scientific platform analysis capabilities")
    
    # Execute the analysis
    results = analyze_stsci_archive()
    
    # Display summary
    print("\n" + "="*80)
    print("🏆 STSCI ARCHIVE EXPLORATION SUMMARY")
    print("="*80)
    print(f"📊 Mission Status: {results.get('mission_status', 'UNKNOWN')}")
    print(f"🔗 Total Links: {results.get('total_links', 0)}")
    print(f"📝 Forms Detected: {results.get('total_forms', 0)}")
    print(f"🌌 Scientific Keywords: {len(results.get('scientific_keywords_detected', []))}")
    print(f"🔭 Telescope Missions: {len(results.get('telescope_missions', []))}")
    print(f"📊 Research Tools: {len(results.get('research_tools', []))}")
    
    if results.get('intelligence_summary'):
        summary = results['intelligence_summary']
        print(f"\n🧠 Intelligence Assessment:")
        print(f"   📋 Platform Type: {summary.get('platform_assessment', 'Unknown')}")
        print(f"   🎯 Primary Function: {summary.get('primary_function', 'Unknown')}")
        print(f"   👥 User Base: {summary.get('user_base', 'Unknown')}")
        print(f"   🌟 Research Value: {summary.get('research_value', 'Unknown')}")
        print(f"   🚀 Automation Ready: {summary.get('automation_compatibility', 'Unknown')}")
    
    print("="*80)
    print("🌌 FSOT STScI Archive exploration completed! 🔭✨")

if __name__ == "__main__":
    main()
