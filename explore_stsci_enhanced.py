#!/usr/bin/env python3
"""
🌌 Enhanced FSOT Space Telescope Science Institute Archive Explorer
================================================================

Advanced autonomous web exploration of the STScI archive system with enhanced
debugging and dynamic content handling for complex scientific platforms.

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
    print("🌌 Enhanced FSOT Space Telescope Archive Explorer Initialized!")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("🔧 Please ensure fsot_autonomous_web_crawler.py is in the current directory")
    sys.exit(1)

def analyze_stsci_archive_enhanced():
    """
    🔭 Enhanced comprehensive analysis of the Space Telescope Science Institute Archive
    
    This enhanced version includes:
    - Extended loading times for complex scientific platforms
    - Enhanced debugging and content detection
    - Alternative content extraction methods
    - Robust error handling for dynamic sites
    """
    
    print("\n" + "="*80)
    print("🌌 ENHANCED FSOT SPACE TELESCOPE ARCHIVE EXPLORATION")
    print("="*80)
    print("🎯 Target: https://archive.stsci.edu/")
    print("🚀 Mission: Enhanced Scientific Platform Analysis")
    print("🧠 AI System: FSOT Neuromorphic Intelligence")
    print("🔬 Enhancement: Dynamic Content & Extended Analysis")
    print("="*80 + "\n")
    
    # Initialize the autonomous web crawler
    crawler = None
    analysis_results = {
        'exploration_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'target_url': 'https://archive.stsci.edu/',
        'platform_type': 'Scientific Data Archive',
        'mission_status': 'INITIATED',
        'browser_diagnostics': {},
        'content_extraction_methods': [],
        'page_source_analysis': {},
        'navigation_attempts': [],
        'scientific_intelligence': {},
        'debug_information': []
    }
    
    try:
        print("🔧 Initializing Enhanced FSOT Autonomous Web Crawler...")
        crawler = FSotAutonomousWebCrawler()
        print("✅ Crawler initialized successfully!")
        
        print("🌐 Setting up enhanced browser environment...")
        browser_ready = crawler.initialize_browser(headless=False)
        if not browser_ready:
            print("❌ Failed to initialize browser")
            analysis_results['mission_status'] = 'BROWSER_INIT_FAILED'
            return analysis_results
        print("✅ Enhanced browser initialized successfully!")
        
        print("\n🌐 Navigating to STScI Archive...")
        success = crawler.navigate_to_url('https://archive.stsci.edu/')
        
        if not success:
            print("❌ Failed to navigate to STScI Archive")
            analysis_results['mission_status'] = 'NAVIGATION_FAILED'
            return analysis_results
        
        print("✅ Successfully navigated to STScI Archive!")
        
        # Enhanced loading sequence for scientific platforms
        print("⏳ Enhanced loading sequence for dynamic scientific content...")
        time.sleep(8)  # Extended wait for complex scientific platforms
        
        # Check page readiness
        try:
            if crawler.driver:
                page_ready = crawler.driver.execute_script("return document.readyState")
                print(f"📊 Page readiness state: {page_ready}")
                analysis_results['browser_diagnostics']['page_ready_state'] = page_ready
            else:
                print("⚠️ Browser driver not available for readiness check")
        except Exception as e:
            print(f"⚠️ Could not check page readiness: {e}")
            analysis_results['debug_information'].append(f"Page readiness check failed: {e}")
        
        # Get basic page information
        try:
            if crawler.driver:
                current_url = crawler.driver.current_url
                page_title = crawler.driver.title
                print(f"🌐 Current URL: {current_url}")
                print(f"📄 Page Title: {page_title}")
                analysis_results['browser_diagnostics'].update({
                    'final_url': current_url,
                    'page_title': page_title
                })
            else:
                print("⚠️ Browser driver not available for page info")
        except Exception as e:
            print(f"⚠️ Could not get basic page info: {e}")
            analysis_results['debug_information'].append(f"Basic page info failed: {e}")
        
        # Method 1: Use crawler's intelligent exploration
        print("\n🔍 Method 1: Intelligent Platform Exploration...")
        try:
            exploration_data = crawler.explore_page_intelligently()
            if exploration_data:
                print("✅ Intelligent exploration successful!")
                analysis_results['content_extraction_methods'].append('intelligent_exploration')
                analysis_results['intelligent_exploration_data'] = exploration_data
            else:
                print("⚠️ Intelligent exploration returned no data")
                analysis_results['debug_information'].append("Intelligent exploration returned None")
        except Exception as e:
            print(f"❌ Intelligent exploration error: {e}")
            analysis_results['debug_information'].append(f"Intelligent exploration error: {e}")
        
        # Method 2: Direct page source analysis
        print("\n🔍 Method 2: Direct Page Source Analysis...")
        try:
            if crawler.driver:
                page_source = crawler.driver.page_source
                source_length = len(page_source)
                print(f"📄 Page source length: {source_length} characters")
                
                if source_length > 0:
                    analysis_results['content_extraction_methods'].append('page_source')
                    analysis_results['page_source_analysis'] = {
                        'source_length': source_length,
                        'contains_content': source_length > 1000
                    }
                    
                    # Check for scientific keywords in page source
                    scientific_terms = [
                        'hubble', 'telescope', 'jwst', 'kepler', 'spitzer', 'galex',
                        'archive', 'data', 'observation', 'spectrum', 'image',
                        'mission', 'instrument', 'detector', 'stsci', 'nasa',
                        'astronomical', 'cosmic', 'galaxy', 'star', 'research'
                    ]
                    
                    source_lower = page_source.lower()
                    found_terms = [term for term in scientific_terms if term in source_lower]
                    
                    print(f"🔬 Scientific terms found: {len(found_terms)}")
                    if found_terms:
                        print(f"   Keywords: {', '.join(found_terms[:10])}")  # Show first 10
                    
                    analysis_results['scientific_intelligence'] = {
                        'scientific_terms_found': found_terms,
                        'scientific_density': len(found_terms),
                        'platform_detected': len(found_terms) > 0
                    }
                    
                    # Check for specific STScI content
                    stsci_indicators = ['stsci', 'space telescope', 'hubble', 'jwst', 'archive']
                    stsci_found = [term for term in stsci_indicators if term in source_lower]
                    
                    if stsci_found:
                        print(f"🎯 STScI platform confirmed! Found: {', '.join(stsci_found)}")
                        analysis_results['scientific_intelligence']['stsci_confirmed'] = True
                        analysis_results['scientific_intelligence']['stsci_indicators'] = stsci_found
                    else:
                        print("⚠️ STScI indicators not detected in page source")
                    
                else:
                    print("❌ Page source is empty")
                    analysis_results['debug_information'].append("Page source is empty")
            else:
                print("⚠️ Browser driver not available for page source analysis")
                
        except Exception as e:
            print(f"❌ Page source analysis error: {e}")
            analysis_results['debug_information'].append(f"Page source analysis error: {e}")
        
        # Method 3: Element-by-element detection
        print("\n🔍 Method 3: Element Detection...")
        try:
            if crawler.driver:
                from selenium.webdriver.common.by import By
                
                # Check for links
                links = crawler.driver.find_elements(By.TAG_NAME, "a")
                print(f"🔗 Links found: {len(links)}")
                
                # Check for forms
                forms = crawler.driver.find_elements(By.TAG_NAME, "form")
                print(f"📝 Forms found: {len(forms)}")
                
                # Check for images
                images = crawler.driver.find_elements(By.TAG_NAME, "img")
                print(f"🖼️ Images found: {len(images)}")
                
                # Check for headings
                headings = crawler.driver.find_elements(By.CSS_SELECTOR, "h1, h2, h3, h4, h5, h6")
                print(f"📋 Headings found: {len(headings)}")
                
                analysis_results['content_extraction_methods'].append('element_detection')
                analysis_results['element_detection'] = {
                    'links_count': len(links),
                    'forms_count': len(forms),
                    'images_count': len(images),
                    'headings_count': len(headings),
                    'sample_link_texts': []  # Initialize as empty list
                }
                
                # Extract some link text for analysis
                if links:
                    link_texts = []
                    for link in links[:10]:  # First 10 links
                        try:
                            text = link.text.strip()
                            if text:
                                link_texts.append(text)
                        except:
                            pass
                    
                    if link_texts:
                        print(f"🔗 Sample link texts: {', '.join(link_texts[:5])}")
                        analysis_results['element_detection']['sample_link_texts'] = link_texts
            else:
                print("⚠️ Browser driver not available for element detection")
            
        except Exception as e:
            print(f"❌ Element detection error: {e}")
            analysis_results['debug_information'].append(f"Element detection error: {e}")
        
        # Final assessment
        methods_used = len(analysis_results['content_extraction_methods'])
        print(f"\n📊 Content extraction methods used: {methods_used}")
        
        if methods_used > 0:
            analysis_results['mission_status'] = 'ENHANCED_EXPLORATION_COMPLETE'
            print("✅ Enhanced STScI Archive exploration completed successfully!")
        else:
            analysis_results['mission_status'] = 'CONTENT_EXTRACTION_FAILED'
            print("⚠️ All content extraction methods failed")
        
    except Exception as e:
        print(f"❌ Critical error during enhanced STScI exploration: {str(e)}")
        analysis_results['mission_status'] = 'CRITICAL_ERROR'
        analysis_results['critical_error'] = str(e)
        
    finally:
        # Clean up browser resources
        if crawler:
            try:
                print("\n🧹 Cleaning up browser resources...")
                crawler.cleanup()
                print("✅ Browser cleanup completed")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup warning: {cleanup_error}")
    
    # Save enhanced analysis report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"STScI_Enhanced_Analysis_{timestamp}.json"
    
    try:
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        print(f"📄 Enhanced analysis report saved: {report_filename}")
    except Exception as save_error:
        print(f"⚠️ Failed to save report: {save_error}")
    
    return analysis_results

def main():
    """
    🌌 Main execution function for enhanced STScI Archive exploration
    """
    print("🚀 Enhanced FSOT Space Telescope Science Institute Archive Explorer")
    print("🎯 Demonstrating advanced scientific platform analysis with enhanced debugging")
    
    # Execute the enhanced analysis
    results = analyze_stsci_archive_enhanced()
    
    # Display enhanced summary
    print("\n" + "="*80)
    print("🏆 ENHANCED STSCI ARCHIVE EXPLORATION SUMMARY")
    print("="*80)
    print(f"📊 Mission Status: {results.get('mission_status', 'UNKNOWN')}")
    print(f"🔬 Content Extraction Methods: {len(results.get('content_extraction_methods', []))}")
    print(f"🧠 Debug Information Entries: {len(results.get('debug_information', []))}")
    
    # Show browser diagnostics
    if results.get('browser_diagnostics'):
        diag = results['browser_diagnostics']
        print(f"\n🌐 Browser Diagnostics:")
        print(f"   📄 Page Title: {diag.get('page_title', 'Unknown')}")
        print(f"   🌐 Final URL: {diag.get('final_url', 'Unknown')}")
        print(f"   ✅ Page Ready: {diag.get('page_ready_state', 'Unknown')}")
    
    # Show scientific intelligence
    if results.get('scientific_intelligence'):
        sci_intel = results['scientific_intelligence']
        print(f"\n🔬 Scientific Intelligence:")
        print(f"   🌌 Scientific Terms: {sci_intel.get('scientific_density', 0)}")
        print(f"   🎯 STScI Confirmed: {sci_intel.get('stsci_confirmed', False)}")
        if sci_intel.get('stsci_indicators'):
            print(f"   🔍 STScI Indicators: {', '.join(sci_intel['stsci_indicators'][:5])}")
    
    # Show element detection results
    if results.get('element_detection'):
        elem = results['element_detection']
        print(f"\n🔍 Element Detection:")
        print(f"   🔗 Links: {elem.get('links_count', 0)}")
        print(f"   📝 Forms: {elem.get('forms_count', 0)}")
        print(f"   🖼️ Images: {elem.get('images_count', 0)}")
        print(f"   📋 Headings: {elem.get('headings_count', 0)}")
    
    print("="*80)
    print("🌌 Enhanced FSOT STScI Archive exploration completed! 🔭✨")

if __name__ == "__main__":
    main()
