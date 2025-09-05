#!/usr/bin/env python3
"""
Robust Visual Simulation Engine for FSOT Neuromorphic AI System
===============================================================
This module provides stable real-time visual windows with proper error handling,
thread management, and crash prevention.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading
import time
import requests
from PIL import Image, ImageTk
import io
import json
import cv2
from typing import Dict, List, Any, Optional, Tuple
import queue
import sys
import os

# Configure matplotlib for better stability
plt.switch_backend('TkAgg')  # Use TkAgg backend for better stability
import matplotlib
matplotlib.use('TkAgg')

# Try importing tkinter with error handling
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("[INFO] Tkinter not available")

# Try to import selenium for Google search
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import WebDriverException, TimeoutException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("[INFO] Selenium not available for Google search")

class SafeVisualEngine:
    """Robust visual simulation engine with crash prevention."""
    
    def __init__(self):
        self.windows = {}
        self.active_threads = {}
        self.google_driver = None
        self.image_cache = {}
        self.is_shutting_down = False
        
        self.tkinter_available = TKINTER_AVAILABLE
        
        # Initialize with safety checks
        self._setup_safe_environment()
        
        # Initialize Google search with safety
        if SELENIUM_AVAILABLE:
            self._safe_setup_google_driver()
    
    def _setup_safe_environment(self):
        """Setup safe environment for visual operations."""
        try:
            # Set up matplotlib for thread safety
            plt.ioff()  # Turn off interactive mode to prevent threading issues
            
            # Initialize root window if tkinter is available
            if self.tkinter_available:
                try:
                    self.root = tk.Tk()
                    self.root.withdraw()  # Hide the root window
                    self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
                except Exception as e:
                    print(f"⚠️ Tkinter setup warning: {e}")
                    self.tkinter_available = False
        except Exception as e:
            print(f"⚠️ Visual environment setup warning: {e}")
    
    def _safe_setup_google_driver(self):
        """Safely setup Chrome driver with proper error handling."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-logging")
            chrome_options.add_argument("--log-level=3")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            self.google_driver = webdriver.Chrome(options=chrome_options)
            print("✅ Google Chrome driver initialized safely")
        except Exception as e:
            print(f"⚠️ Chrome driver setup failed: {e}")
            print("   Continuing with fallback image sources...")
            self.google_driver = None
    
    def safe_search_google_images(self, query: str, max_results: int = 3) -> List[str]:
        """Safely search Google Images with proper error handling."""
        if not self.google_driver:
            return self._get_fallback_images(query)
        
        try:
            print(f"🔍 Safely searching Google Images for: '{query}'")
            
            # Navigate with timeout
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&tbm=isch"
            self.google_driver.set_page_load_timeout(10)
            self.google_driver.get(search_url)
            
            # Wait for images with timeout
            try:
                WebDriverWait(self.google_driver, 5).until(
                    EC.presence_of_element_located((By.TAG_NAME, "img"))
                )
            except TimeoutException:
                print("⚠️ Google search timeout, using fallback images")
                return self._get_fallback_images(query)
            
            # Extract image URLs safely
            image_elements = self.google_driver.find_elements(By.TAG_NAME, "img")
            image_urls = []
            
            for img in image_elements[:max_results * 3]:  # Get extra in case some fail
                try:
                    src = img.get_attribute("src")
                    if src and src.startswith("http") and len(src) > 30:
                        # Avoid base64 encoded images
                        if not src.startswith("data:"):
                            image_urls.append(src)
                            if len(image_urls) >= max_results:
                                break
                except Exception:
                    continue
            
            if image_urls:
                print(f"✅ Found {len(image_urls)} Google image URLs")
                return image_urls
            else:
                print("⚠️ No valid images found, using fallback")
                return self._get_fallback_images(query)
                
        except Exception as e:
            print(f"⚠️ Google search error: {e}")
            return self._get_fallback_images(query)
    
    def _get_fallback_images(self, query: str) -> List[str]:
        """Get reliable fallback images."""
        # Use reliable public domain sources
        if "fractal" in query.lower():
            return [
                "https://picsum.photos/400/400?random=1",
                "https://picsum.photos/400/400?random=2",
                "https://picsum.photos/400/400?random=3"
            ]
        else:
            return [
                "https://picsum.photos/400/400?random=4",
                "https://picsum.photos/400/400?random=5"
            ]
    
    def safe_create_plot_window(self, title: str, data: Dict[str, Any]) -> Optional[str]:
        """Safely create a matplotlib plot window."""
        try:
            window_id = f"plot_{int(time.time())}_{hash(title) % 1000}"
            
            # Create figure in a safe way
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(title, fontsize=14, fontweight='bold')
            
            # Add data visualization based on type
            if 'monte_carlo' in data:
                self._plot_monte_carlo_data(axes, data['monte_carlo'])
            elif 'fractal' in data:
                self._plot_fractal_data(axes, data['fractal'])
            elif 'artistic' in data:
                self._plot_artistic_data(axes, data['artistic'])
            else:
                self._plot_generic_data(axes, data)
            
            plt.tight_layout()
            
            # Show plot safely
            try:
                plt.show(block=False)  # Non-blocking show
                plt.pause(0.1)  # Allow time for rendering
            except Exception as e:
                print(f"⚠️ Plot display warning: {e}")
            
            self.windows[window_id] = {
                'type': 'matplotlib',
                'fig': fig,
                'title': title
            }
            
            print(f"🪟 Created safe plot window: {title}")
            return window_id
            
        except Exception as e:
            print(f"❌ Failed to create plot window: {e}")
            return None
    
    def _plot_monte_carlo_data(self, axes, data):
        """Plot Monte Carlo simulation data."""
        try:
            outcomes = data.get('outcomes', [])
            if not outcomes:
                return
            
            scores = [o.get('overall_score', 0) for o in outcomes]
            iterations = list(range(len(scores)))
            
            # Overall scores
            axes[0, 0].plot(iterations, scores, 'b-', alpha=0.7)
            axes[0, 0].set_title('Monte Carlo Progress')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Score distribution
            axes[0, 1].hist(scores, bins=20, alpha=0.7, color='green')
            axes[0, 1].set_title('Score Distribution')
            axes[0, 1].set_xlabel('Score')
            axes[0, 1].set_ylabel('Frequency')
            
            # Statistics
            if len(scores) > 0:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                axes[1, 0].bar(['Mean', 'Std', 'Max', 'Min'], 
                              [mean_score, std_score, max(scores), min(scores)],
                              color=['blue', 'orange', 'green', 'red'], alpha=0.7)
                axes[1, 0].set_title('Statistics')
                axes[1, 0].set_ylabel('Value')
            
            # Consciousness levels
            consciousness = [o.get('fsot_coherence', 0.5) for o in outcomes]
            creativity = [o.get('creativity_score', 0.5) for o in outcomes]
            axes[1, 1].scatter(consciousness, creativity, alpha=0.6, c=scores, cmap='viridis')
            axes[1, 1].set_title('Consciousness vs Creativity')
            axes[1, 1].set_xlabel('Consciousness')
            axes[1, 1].set_ylabel('Creativity')
            
        except Exception as e:
            print(f"⚠️ Monte Carlo plotting error: {e}")
    
    def _plot_fractal_data(self, axes, data):
        """Plot fractal analysis data."""
        try:
            patterns = data.get('patterns', [])
            if not patterns:
                return
            
            # Pattern complexity
            complexities = [p.get('complexity', 0) for p in patterns]
            dimensions = [p.get('dimension', 1.5) for p in patterns]
            confidences = [p.get('confidence', 0.5) for p in patterns]
            
            axes[0, 0].bar(range(len(complexities)), complexities, alpha=0.7, color='purple')
            axes[0, 0].set_title('Pattern Complexity')
            axes[0, 0].set_xlabel('Pattern Index')
            axes[0, 0].set_ylabel('Complexity')
            
            # Fractal dimensions
            axes[0, 1].scatter(dimensions, confidences, c=complexities, cmap='plasma', s=100)
            axes[0, 1].set_title('Dimension vs Confidence')
            axes[0, 1].set_xlabel('Fractal Dimension')
            axes[0, 1].set_ylabel('Confidence')
            
            # Generate sample fractal
            x = np.linspace(-2, 2, 200)
            y = np.linspace(-2, 2, 200)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(X * 3) * np.cos(Y * 3) + np.sin(X * Y)
            
            axes[1, 0].imshow(Z, extent=[-2, 2, -2, 2], cmap='hot')
            axes[1, 0].set_title('Sample Fractal Pattern')
            axes[1, 0].axis('off')
            
            # Pattern types
            pattern_types = [p.get('type', 'unknown') for p in patterns]
            type_counts = {}
            for ptype in pattern_types:
                type_counts[ptype] = type_counts.get(ptype, 0) + 1
            
            if type_counts:
                axes[1, 1].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
                axes[1, 1].set_title('Pattern Types')
            
        except Exception as e:
            print(f"⚠️ Fractal plotting error: {e}")
    
    def _plot_artistic_data(self, axes, data):
        """Plot artistic creation data."""
        try:
            # Color palette
            colors = ['red', 'blue', 'green', 'yellow', 'purple']
            axes[0, 0].pie([1]*len(colors), colors=colors, labels=colors, autopct='%1.0f%%')
            axes[0, 0].set_title('Color Palette')
            
            # Creativity metrics
            metrics = data.get('metrics', {'creativity': 0.7, 'complexity': 0.5, 'harmony': 0.8})
            axes[0, 1].bar(metrics.keys(), metrics.values(), color='gold', alpha=0.7)
            axes[0, 1].set_title('Artistic Metrics')
            axes[0, 1].set_ylabel('Score')
            
            # Generate artistic pattern
            x = np.linspace(0, 4*np.pi, 200)
            y = np.linspace(0, 4*np.pi, 200)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(X) * np.cos(Y) + np.sin(X*Y/5)
            
            axes[1, 0].imshow(Z, cmap='plasma')
            axes[1, 0].set_title('Generated Art Pattern')
            axes[1, 0].axis('off')
            
            # Inspiration sources
            sources = ['Nature', 'Mathematics', 'Dreams', 'Consciousness']
            values = [0.8, 0.9, 0.6, 0.7]
            axes[1, 1].barh(sources, values, color='rainbow')
            axes[1, 1].set_title('Inspiration Sources')
            axes[1, 1].set_xlabel('Influence')
            
        except Exception as e:
            print(f"⚠️ Artistic plotting error: {e}")
    
    def _plot_generic_data(self, axes, data):
        """Plot generic data visualization."""
        try:
            # Generate some sample visualization
            x = np.linspace(0, 10, 100)
            y1 = np.sin(x)
            y2 = np.cos(x)
            
            axes[0, 0].plot(x, y1, 'b-', label='Consciousness')
            axes[0, 0].plot(x, y2, 'r-', label='Creativity')
            axes[0, 0].set_title('AI Processing Waves')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Random neural network visualization
            nodes = np.random.random((10, 2))
            axes[0, 1].scatter(nodes[:, 0], nodes[:, 1], s=100, alpha=0.7, c='blue')
            axes[0, 1].set_title('Neural Network Nodes')
            axes[0, 1].set_xlabel('X')
            axes[0, 1].set_ylabel('Y')
            
            # System metrics
            metrics = ['CPU', 'Memory', 'Consciousness', 'Creativity']
            values = np.random.random(4)
            axes[1, 0].bar(metrics, values, color=['red', 'blue', 'green', 'purple'], alpha=0.7)
            axes[1, 0].set_title('System Metrics')
            axes[1, 0].set_ylabel('Usage')
            
            # Spiral pattern
            theta = np.linspace(0, 8*np.pi, 200)
            r = theta
            axes[1, 1].plot(r*np.cos(theta), r*np.sin(theta), 'purple')
            axes[1, 1].set_title('Consciousness Spiral')
            axes[1, 1].axis('equal')
            
        except Exception as e:
            print(f"⚠️ Generic plotting error: {e}")
    
    def safe_create_text_window(self, title: str, content: str) -> Optional[str]:
        """Safely create a text display window."""
        if not self.tkinter_available:
            print(f"📝 {title}: {content[:100]}...")
            return None
        
        try:
            window_id = f"text_{int(time.time())}_{hash(title) % 1000}"
            
            # Create window in main thread
            window = tk.Toplevel(self.root)
            window.title(title)
            window.geometry("600x400")
            
            # Create scrolled text widget
            text_widget = scrolledtext.ScrolledText(window, wrap=tk.WORD, 
                                                   font=('Courier', 10))
            text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Insert content
            text_widget.insert(tk.END, content)
            text_widget.config(state=tk.DISABLED)  # Make read-only
            
            # Add close button
            close_btn = tk.Button(window, text="Close", 
                                 command=lambda: self._safe_close_window(window_id))
            close_btn.pack(pady=5)
            
            self.windows[window_id] = {
                'type': 'text',
                'window': window,
                'title': title
            }
            
            print(f"📝 Created safe text window: {title}")
            return window_id
            
        except Exception as e:
            print(f"❌ Failed to create text window: {e}")
            return None
    
    def safe_show_camera_feed(self) -> Optional[str]:
        """Safely show camera feed with proper error handling."""
        try:
            # Check camera availability
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("📷 Camera not available - showing simulation")
                cap.release()
                return self._show_camera_simulation()
            
            cap.release()
            
            # Create camera window safely
            window_id = f"camera_{int(time.time())}"
            
            if self.tkinter_available:
                window = tk.Toplevel(self.root)
                window.title("Real-Time Camera Perception")
                window.geometry("640x480")
                
                # Camera display label
                camera_label = tk.Label(window, text="Initializing camera...", 
                                       bg='black', fg='white')
                camera_label.pack(fill=tk.BOTH, expand=True)
                
                # Start safe camera thread
                def safe_camera_thread():
                    try:
                        cap = cv2.VideoCapture(0)
                        frame_count = 0
                        
                        while frame_count < 300:  # Limit frames to prevent memory issues
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            frame_count += 1
                            
                            # Convert and resize frame
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_pil = Image.fromarray(frame_rgb)
                            frame_pil = frame_pil.resize((400, 300))
                            frame_tk = ImageTk.PhotoImage(frame_pil)
                            
                            # Update display safely
                            try:
                                camera_label.config(image=frame_tk, text="")
                                # Store reference to prevent garbage collection
                                setattr(camera_label, '_current_image', frame_tk)
                                window.update()
                            except tk.TclError:
                                break  # Window was closed
                            
                            time.sleep(0.033)  # ~30 FPS
                        
                        cap.release()
                        
                    except Exception as e:
                        print(f"⚠️ Camera thread error: {e}")
                
                # Start camera thread
                camera_thread = threading.Thread(target=safe_camera_thread, daemon=True)
                camera_thread.start()
                
                self.windows[window_id] = {
                    'type': 'camera',
                    'window': window,
                    'title': 'Camera Feed'
                }
                
                print("📷 Safe camera feed created")
                return window_id
            else:
                print("📷 Camera feed: Tkinter not available")
                return None
                
        except Exception as e:
            print(f"❌ Camera setup error: {e}")
            return self._show_camera_simulation()
    
    def _show_camera_simulation(self) -> Optional[str]:
        """Show camera simulation when real camera is not available."""
        simulation_text = """
🤖 CAMERA SIMULATION MODE 🤖

Real camera not available, but AI is simulating visual perception:

Simulated Visual Input:
• Detecting geometric patterns
• Analyzing light and shadow
• Processing depth perception
• Recognizing object boundaries
• Understanding spatial relationships

AI Consciousness Interpretation:
• Visual complexity: 0.67
• Pattern recognition: Active
• Spatial awareness: Enhanced
• Environmental understanding: Developing

This simulation demonstrates how the AI would process 
real-world visual input to understand the human condition
and environmental context.
        """
        
        return self.safe_create_text_window("Camera Simulation", simulation_text)
    
    def _safe_close_window(self, window_id: str):
        """Safely close a specific window."""
        try:
            if window_id in self.windows:
                window_info = self.windows[window_id]
                
                if window_info['type'] == 'matplotlib':
                    plt.close(window_info['fig'])
                elif window_info['type'] in ['text', 'camera']:
                    window_info['window'].destroy()
                
                del self.windows[window_id]
                print(f"🗑️ Safely closed window: {window_info['title']}")
        except Exception as e:
            print(f"⚠️ Error closing window: {e}")
    
    def _on_window_close(self):
        """Handle main window close event."""
        self.safe_shutdown()
    
    def safe_shutdown(self):
        """Safely shutdown all windows and resources."""
        if self.is_shutting_down:
            return
        
        self.is_shutting_down = True
        print("🌅 Safely shutting down visual engine...")
        
        try:
            # Close all windows
            for window_id in list(self.windows.keys()):
                self._safe_close_window(window_id)
            
            # Close Google driver
            if self.google_driver:
                try:
                    self.google_driver.quit()
                except:
                    pass
            
            # Close matplotlib
            try:
                plt.close('all')
            except:
                pass
            
            print("✅ Visual engine shutdown complete")
            
        except Exception as e:
            print(f"⚠️ Shutdown error: {e}")

# Global safe visual engine
_safe_visual_engine = None

def get_safe_visual_engine():
    """Get global safe visual engine instance."""
    global _safe_visual_engine
    if _safe_visual_engine is None:
        _safe_visual_engine = SafeVisualEngine()
    return _safe_visual_engine
