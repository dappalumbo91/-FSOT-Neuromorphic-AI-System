#!/usr/bin/env python3
"""
Visual Simulation Engine for FSOT Neuromorphic AI System
========================================================
This module provides real-time visual windows for all simulations and processes,
including Google image search integration and live visualization displays.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import requests
from PIL import Image, ImageTk
import io
import json
import webbrowser
from urllib.parse import quote_plus
import cv2
from typing import Dict, List, Any, Optional, Tuple
import queue

# Try to import selenium for Google search
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("[INFO] Selenium not available for Google search. Install with: pip install selenium")

class VisualSimulationEngine:
    """Real-time visual simulation engine with Google search integration."""
    
    def __init__(self):
        self.windows = {}
        self.active_simulations = {}
        self.animation_threads = {}
        self.google_driver = None
        self.image_cache = {}
        
        # Initialize Google search if available
        if SELENIUM_AVAILABLE:
            self._setup_google_driver()
    
    def _setup_google_driver(self):
        """Setup Chrome driver for Google image search."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run in background
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            self.google_driver = webdriver.Chrome(options=chrome_options)
            print("✅ Google Chrome driver initialized for image search")
        except Exception as e:
            print(f"⚠️ Failed to initialize Chrome driver: {e}")
            print("   Download ChromeDriver from: https://chromedriver.chromium.org/")
            SELENIUM_AVAILABLE = False
    
    def search_google_images(self, query: str, max_results: int = 5) -> List[str]:
        """Search Google Images and return image URLs."""
        if not self.google_driver:
            print("🔍 Google search not available, using fallback URLs")
            return self._fallback_image_urls(query)
        
        try:
            print(f"🔍 Searching Google Images for: '{query}'")
            
            # Navigate to Google Images
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&tbm=isch"
            self.google_driver.get(search_url)
            
            # Wait for images to load
            WebDriverWait(self.google_driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "img[data-src]"))
            )
            
            # Extract image URLs
            image_elements = self.google_driver.find_elements(By.CSS_SELECTOR, "img[data-src]")
            image_urls = []
            
            for img in image_elements[:max_results * 2]:  # Get extra in case some fail
                try:
                    src = img.get_attribute("data-src") or img.get_attribute("src")
                    if src and src.startswith("http") and len(src) > 20:
                        image_urls.append(src)
                        if len(image_urls) >= max_results:
                            break
                except:
                    continue
            
            print(f"✅ Found {len(image_urls)} Google image URLs")
            return image_urls[:max_results]
            
        except Exception as e:
            print(f"❌ Google search failed: {e}")
            return self._fallback_image_urls(query)
    
    def _fallback_image_urls(self, query: str) -> List[str]:
        """Fallback image URLs when Google search fails."""
        # Wikimedia Commons has reliable URLs for mathematical/scientific content
        fallback_urls = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Mandel_zoom_00_mandelbrot_set.jpg/800px-Mandel_zoom_00_mandelbrot_set.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Von_Koch_curve.gif/400px-Von_Koch_curve.gif",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Julia_set_%28highres_01%29.jpg/400px-Julia_set_%28highres_01%29.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/Sierpinski_triangle.svg/400px-Sierpinski_triangle.svg.png",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Lorenz_attractor_yb.svg/400px-Lorenz_attractor_yb.svg.png"
        ]
        
        if "fractal" in query.lower():
            return fallback_urls
        elif "nature" in query.lower():
            return [
                "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.svg/400px-Vd-Orig.svg.png",
                "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f4/Honeycomb.jpg/400px-Honeycomb.jpg"
            ]
        else:
            return fallback_urls[:3]
    
    def create_simulation_window(self, title: str, window_type: str = "plot") -> str:
        """Create a new simulation window."""
        window_id = f"{window_type}_{int(time.time())}_{hash(title) % 1000}"
        
        if window_type == "plot":
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.suptitle(title, fontsize=14, fontweight='bold')
            
            # Enable interactive mode
            plt.ion()
            
            self.windows[window_id] = {
                'type': 'matplotlib',
                'fig': fig,
                'ax': ax,
                'title': title
            }
            
            # Show the window
            plt.show()
            
        elif window_type == "tkinter":
            root = tk.Toplevel() if hasattr(tk, '_default_root') and tk._default_root else tk.Tk()
            root.title(title)
            root.geometry("800x600")
            
            self.windows[window_id] = {
                'type': 'tkinter',
                'root': root,
                'title': title
            }
        
        print(f"🪟 Created {window_type} window: {title}")
        return window_id
    
    def show_monte_carlo_simulation(self, dream_state, outcomes: List[Dict]) -> str:
        """Show real-time Monte Carlo simulation visualization."""
        window_id = self.create_simulation_window(
            f"Monte Carlo Dream Simulation - {dream_state.dream_id}", 
            "plot"
        )
        
        window = self.windows[window_id]
        fig, ax = window['fig'], window['ax']
        
        # Prepare data for visualization
        iterations = list(range(len(outcomes)))
        scores = [outcome.get('overall_score', 0) for outcome in outcomes]
        success_probs = [outcome.get('success_probability', 0) for outcome in outcomes]
        creativity_scores = [outcome.get('creativity_score', 0) for outcome in outcomes]
        
        # Create subplots
        fig.clear()
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Overall scores over time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(iterations, scores, 'b-', alpha=0.7, linewidth=2, label='Overall Score')
        ax1.fill_between(iterations, scores, alpha=0.3)
        ax1.set_title('Monte Carlo Simulation Progress', fontweight='bold')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Score')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Success probability distribution
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.hist(success_probs, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_title('Success Probability Distribution')
        ax2.set_xlabel('Success Probability')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Creativity vs Success scatter
        ax3 = fig.add_subplot(gs[1, 1])
        scatter = ax3.scatter(creativity_scores, success_probs, 
                            c=scores, cmap='viridis', alpha=0.6, s=50)
        ax3.set_title('Creativity vs Success')
        ax3.set_xlabel('Creativity Score')
        ax3.set_ylabel('Success Probability')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Overall Score')
        
        # Add statistics text
        mean_score = np.mean(scores)
        best_score = np.max(scores)
        std_score = np.std(scores)
        
        fig.suptitle(f'Monte Carlo Results: Mean={mean_score:.3f}, Best={best_score:.3f}, Std={std_score:.3f}', 
                     fontsize=12, fontweight='bold')
        
        plt.draw()
        plt.pause(0.1)  # Allow time for rendering
        
        return window_id
    
    def show_fractal_analysis(self, image_url: str, patterns: List[Dict]) -> str:
        """Show fractal pattern analysis visualization."""
        window_id = self.create_simulation_window(
            "Fractal Pattern Analysis", "plot"
        )
        
        window = self.windows[window_id]
        fig, ax = window['fig'], window['ax']
        
        try:
            # Download and display the image
            print(f"📥 Downloading image for analysis: {image_url}")
            response = requests.get(image_url, timeout=10)
            
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
                img_array = np.array(image.convert('RGB'))
                
                # Create visualization
                fig.clear()
                gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
                
                # Original image
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(img_array)
                ax1.set_title('Original Image')
                ax1.axis('off')
                
                # Grayscale version
                ax2 = fig.add_subplot(gs[0, 1])
                gray_img = np.array(image.convert('L'))
                ax2.imshow(gray_img, cmap='gray')
                ax2.set_title('Grayscale Analysis')
                ax2.axis('off')
                
                # Pattern complexity visualization
                ax3 = fig.add_subplot(gs[1, 0])
                if patterns:
                    pattern_types = [p.get('type', 'unknown') for p in patterns]
                    complexities = [p.get('complexity', 0) for p in patterns]
                    
                    bars = ax3.bar(range(len(pattern_types)), complexities, 
                                  color='purple', alpha=0.7)
                    ax3.set_title('Pattern Complexity')
                    ax3.set_xlabel('Pattern Index')
                    ax3.set_ylabel('Complexity Score')
                    ax3.set_xticks(range(len(pattern_types)))
                    ax3.set_xticklabels([f"P{i+1}" for i in range(len(pattern_types))])
                    ax3.grid(True, alpha=0.3)
                
                # Fractal dimensions
                ax4 = fig.add_subplot(gs[1, 1])
                if patterns:
                    dimensions = [p.get('dimension', 1.5) for p in patterns]
                    confidences = [p.get('confidence', 0.5) for p in patterns]
                    
                    scatter = ax4.scatter(dimensions, confidences, 
                                        c=complexities if patterns else [0], 
                                        cmap='plasma', s=100, alpha=0.7)
                    ax4.set_title('Fractal Dimensions vs Confidence')
                    ax4.set_xlabel('Fractal Dimension')
                    ax4.set_ylabel('Confidence')
                    ax4.grid(True, alpha=0.3)
                    
                    if len(dimensions) > 0:
                        plt.colorbar(scatter, ax=ax4, label='Complexity')
                
                # Add pattern information as text
                if patterns:
                    pattern_info = "\n".join([
                        f"Pattern {i+1}: {p.get('type', 'unknown')} "
                        f"(dim={p.get('dimension', 0):.3f}, "
                        f"conf={p.get('confidence', 0):.3f})"
                        for i, p in enumerate(patterns[:5])
                    ])
                    fig.text(0.02, 0.02, pattern_info, fontsize=8, 
                            verticalalignment='bottom', fontfamily='monospace')
                
                plt.draw()
                plt.pause(0.1)
                
                print(f"✅ Fractal analysis visualization complete")
                
            else:
                print(f"❌ Failed to download image: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Fractal visualization error: {e}")
        
        return window_id
    
    def show_artistic_creation(self, artistic_concept: Dict) -> str:
        """Show artistic creation visualization."""
        window_id = self.create_simulation_window(
            "AI Artistic Creation Process", "plot"
        )
        
        window = self.windows[window_id]
        fig, ax = window['fig'], window['ax']
        
        # Create artistic visualization
        fig.clear()
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Color palette visualization
        ax1 = fig.add_subplot(gs[0, 0])
        color_palette = artistic_concept.get('color_palette', {})
        primary_colors = color_palette.get('primary_colors', ['blue', 'red', 'green'])
        
        # Create color swatches
        color_map = {
            'deep_blue': '#003366', 'vibrant_purple': '#8B00FF', 'electric_green': '#00FF00',
            'warm_orange': '#FF8C00', 'soft_blue': '#87CEEB', 'natural_green': '#228B22',
            'pure_white': '#FFFFFF', 'deep_black': '#000000', 'subtle_gray': '#808080',
            'blue': '#0000FF', 'red': '#FF0000', 'green': '#008000'
        }
        
        colors = [color_map.get(color, '#888888') for color in primary_colors]
        ax1.pie([1] * len(colors), colors=colors, labels=primary_colors, autopct='%1.0f%%')
        ax1.set_title('AI-Generated Color Palette')
        
        # Fractal inspiration visualization
        ax2 = fig.add_subplot(gs[0, 1])
        fractal_inspiration = artistic_concept.get('fractal_inspiration', {})
        complexity = fractal_inspiration.get('complexity_level', 0.5)
        dimension = fractal_inspiration.get('dimensional_depth', 1.5)
        
        # Generate artistic pattern based on parameters
        x = np.linspace(-2, 2, 400)
        y = np.linspace(-2, 2, 400)
        X, Y = np.meshgrid(x, y)
        
        # Create fractal-inspired pattern
        Z = (np.sin(X * complexity * 5) * np.cos(Y * dimension * 3) +
             np.sin(X * Y * complexity) * dimension)
        
        im = ax2.imshow(Z, extent=[-2, 2, -2, 2], cmap='plasma', alpha=0.8)
        ax2.set_title(f'Fractal Inspiration\n(Complexity: {complexity:.2f})')
        ax2.axis('off')
        
        # Dream influence metrics
        ax3 = fig.add_subplot(gs[1, 0])
        dream_influence = artistic_concept.get('dream_influence', {})
        metrics = {
            'Creativity': dream_influence.get('creativity_factor', 0.5),
            'Consciousness': dream_influence.get('consciousness_integration', 0.5),
            'Success Prob': dream_influence.get('success_probability', 0.5)
        }
        
        bars = ax3.bar(metrics.keys(), metrics.values(), 
                      color=['gold', 'purple', 'green'], alpha=0.7)
        ax3.set_title('Dream State Influence')
        ax3.set_ylabel('Score')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # Composition ideas
        ax4 = fig.add_subplot(gs[1, 1])
        composition_ideas = artistic_concept.get('composition_ideas', [])
        suggestions = artistic_concept.get('artistic_suggestions', [])
        
        # Create text display of ideas
        all_ideas = composition_ideas + suggestions
        if all_ideas:
            idea_text = "\n".join([f"• {idea[:40]}..." if len(idea) > 40 else f"• {idea}" 
                                 for idea in all_ideas[:6]])
            ax4.text(0.05, 0.95, "Artistic Concepts:", fontweight='bold', 
                    transform=ax4.transAxes, verticalalignment='top')
            ax4.text(0.05, 0.85, idea_text, transform=ax4.transAxes, 
                    verticalalignment='top', fontsize=9, fontfamily='monospace')
        
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.draw()
        plt.pause(0.1)
        
        print(f"🎨 Artistic creation visualization complete")
        return window_id
    
    def show_real_time_camera(self) -> str:
        """Show real-time camera feed window."""
        window_id = self.create_simulation_window(
            "Real-Time World Perception", "tkinter"
        )
        
        window = self.windows[window_id]
        root = window['root']
        
        # Create camera display
        camera_frame = ttk.Frame(root)
        camera_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Camera display label
        camera_label = tk.Label(camera_frame, text="Initializing camera...", 
                               bg='black', fg='white', font=('Arial', 12))
        camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Analysis text area
        analysis_frame = ttk.Frame(root)
        analysis_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        analysis_text = scrolledtext.ScrolledText(analysis_frame, height=8, 
                                                 font=('Courier', 10))
        analysis_text.pack(fill=tk.X)
        
        # Start camera thread
        def camera_thread():
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    camera_label.config(text="Camera not available\nUsing simulated perception")
                    analysis_text.insert(tk.END, "📷 Camera not detected\n")
                    analysis_text.insert(tk.END, "🤖 Switching to simulated perception mode\n")
                    analysis_text.insert(tk.END, "💭 AI is imagining visual patterns...\n")
                    return
                
                analysis_text.insert(tk.END, "📷 Camera initialized successfully\n")
                analysis_text.insert(tk.END, "👁️ AI perception system active\n")
                
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Convert frame for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_pil = frame_pil.resize((400, 300))
                    frame_tk = ImageTk.PhotoImage(frame_pil)
                    
                    # Update display
                    camera_label.config(image=frame_tk, text="")
                    camera_label.image = frame_tk
                    
                    # Analyze frame periodically
                    if frame_count % 30 == 0:  # Every 30 frames (~1 second)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        complexity = np.std(gray) / 255.0
                        edges = cv2.Canny(gray, 50, 150)
                        edge_density = np.sum(edges > 0) / edges.size
                        
                        analysis_text.insert(tk.END, 
                            f"🔍 Frame {frame_count}: complexity={complexity:.3f}, "
                            f"edges={edge_density:.3f}\n")
                        analysis_text.see(tk.END)
                    
                    time.sleep(0.033)  # ~30 FPS
                    
            except Exception as e:
                camera_label.config(text=f"Camera error: {e}")
                analysis_text.insert(tk.END, f"❌ Camera error: {e}\n")
        
        # Start camera in separate thread
        camera_thread_obj = threading.Thread(target=camera_thread, daemon=True)
        camera_thread_obj.start()
        
        print(f"📷 Real-time camera window created")
        return window_id
    
    def show_image_gallery(self, images: List[str], title: str = "Image Gallery") -> str:
        """Show a gallery of downloaded images."""
        window_id = self.create_simulation_window(title, "tkinter")
        
        window = self.windows[window_id]
        root = window['root']
        root.geometry("900x700")
        
        # Create scrollable frame
        main_frame = ttk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Canvas for scrolling
        canvas = tk.Canvas(main_frame, bg='white')
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Load and display images
        for i, image_url in enumerate(images):
            try:
                print(f"📥 Loading image {i+1}/{len(images)}: {image_url[:50]}...")
                
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))
                    image.thumbnail((250, 250))  # Resize for display
                    photo = ImageTk.PhotoImage(image)
                    
                    # Create frame for each image
                    img_frame = ttk.Frame(scrollable_frame)
                    img_frame.grid(row=i//3, column=i%3, padx=10, pady=10, sticky="nsew")
                    
                    # Image label
                    img_label = tk.Label(img_frame, image=photo)
                    img_label.image = photo  # Keep a reference
                    img_label.pack()
                    
                    # URL label
                    url_label = tk.Label(img_frame, text=f"Image {i+1}", 
                                       font=('Arial', 8), fg='blue')
                    url_label.pack()
                    
                else:
                    print(f"❌ Failed to load image {i+1}: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error loading image {i+1}: {e}")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        print(f"🖼️ Image gallery window created with {len(images)} images")
        return window_id
    
    def close_window(self, window_id: str):
        """Close a simulation window."""
        if window_id in self.windows:
            window = self.windows[window_id]
            
            if window['type'] == 'matplotlib':
                plt.close(window['fig'])
            elif window['type'] == 'tkinter':
                window['root'].destroy()
            
            del self.windows[window_id]
            print(f"🗑️ Closed window: {window['title']}")
    
    def close_all_windows(self):
        """Close all simulation windows."""
        for window_id in list(self.windows.keys()):
            self.close_window(window_id)
    
    def __del__(self):
        """Cleanup when engine is destroyed."""
        if self.google_driver:
            try:
                self.google_driver.quit()
            except:
                pass
        self.close_all_windows()

# Global visual engine instance
_visual_engine = None

def get_visual_engine():
    """Get global visual engine instance."""
    global _visual_engine
    if _visual_engine is None:
        _visual_engine = VisualSimulationEngine()
    return _visual_engine
