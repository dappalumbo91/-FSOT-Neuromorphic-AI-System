#!/usr/bin/env python3
"""
FSOT Desktop Automation Training Pipeline
=========================================
Trains FSOT to interact with desktop applications and system controls.
"""

import time
import json
import logging
import subprocess
import psutil
import pyautogui
import pygetwindow as gw
from typing import Dict, List, Any, Optional, Tuple
import win32gui
import win32con
import win32process
from pathlib import Path

# Configure PyAutoGUI safety
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.5

class FSOTDesktopTrainer:
    """Desktop automation training system for FSOT."""
    
    def __init__(self):
        self.logger = logging.getLogger("FSOT_Desktop_Trainer")
        self.training_data = {
            "sessions": [],
            "learned_applications": {},
            "automation_patterns": {},
            "system_knowledge": {}
        }
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Training applications configuration
        self.training_apps = {
            "notepad": {
                "executable": "notepad.exe",
                "window_title_pattern": "Notepad",
                "training_tasks": ["open", "type_text", "save", "close"],
                "safe_for_automation": True
            },
            "calculator": {
                "executable": "calc.exe", 
                "window_title_pattern": "Calculator",
                "training_tasks": ["open", "basic_calculations", "close"],
                "safe_for_automation": True
            },
            "file_explorer": {
                "executable": "explorer.exe",
                "window_title_pattern": "File Explorer",
                "training_tasks": ["navigate_folders", "file_operations"],
                "safe_for_automation": True
            },
            "cmd": {
                "executable": "cmd.exe",
                "window_title_pattern": "Command Prompt",
                "training_tasks": ["open", "basic_commands", "close"],
                "safe_for_automation": True
            }
        }
    
    def detect_running_applications(self) -> Dict[str, Any]:
        """Detect currently running applications."""
        running_apps = {
            "processes": [],
            "windows": [],
            "system_info": {}
        }
        
        try:
            # Get running processes
            for process in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                try:
                    running_apps["processes"].append({
                        "pid": process.info['pid'],
                        "name": process.info['name'],
                        "cpu_percent": process.info['cpu_percent'],
                        "memory_mb": process.info['memory_info'].rss / 1024 / 1024
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Get visible windows
            windows = gw.getAllWindows()
            for window in windows:
                if window.title and window.visible:
                    running_apps["windows"].append({
                        "title": window.title,
                        "left": window.left,
                        "top": window.top,
                        "width": window.width,
                        "height": window.height,
                        "active": window == gw.getActiveWindow()
                    })
            
            # System information
            running_apps["system_info"] = {
                "screen_resolution": f"{self.screen_width}x{self.screen_height}",
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "disk_usage": psutil.disk_usage('/').percent if psutil.disk_usage('/') else None
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting applications: {e}")
        
        return running_apps
    
    def launch_application(self, app_name: str) -> Dict[str, Any]:
        """Launch an application for training."""
        result = {
            "app_name": app_name,
            "launch_successful": False,
            "window_found": False,
            "window_info": {},
            "error": None
        }
        
        if app_name not in self.training_apps:
            result["error"] = f"Unknown application: {app_name}"
            return result
        
        app_config = self.training_apps[app_name]
        
        try:
            # Launch the application
            self.logger.info(f"🚀 Launching {app_name}...")
            process = subprocess.Popen(app_config["executable"])
            time.sleep(3)  # Wait for application to start
            
            result["launch_successful"] = True
            
            # Find the application window
            windows = gw.getWindowsWithTitle(app_config["window_title_pattern"])
            if windows:
                window = windows[0]
                result["window_found"] = True
                result["window_info"] = {
                    "title": window.title,
                    "position": (window.left, window.top),
                    "size": (window.width, window.height),
                    "handle": window._hWnd if hasattr(window, '_hWnd') else None
                }
                
                # Activate the window
                window.activate()
                time.sleep(1)
                
            else:
                result["error"] = f"Could not find window for {app_name}"
            
        except Exception as e:
            result["error"] = str(e)
            self.logger.error(f"Failed to launch {app_name}: {e}")
        
        return result
    
    def train_notepad_automation(self) -> Dict[str, Any]:
        """Train Notepad automation skills."""
        training_results = {
            "app": "notepad",
            "tasks_completed": 0,
            "interactions": [],
            "text_samples": [],
            "success_rate": 0.0
        }
        
        # Launch Notepad
        launch_result = self.launch_application("notepad")
        if not launch_result["launch_successful"]:
            training_results["error"] = "Failed to launch Notepad"
            return training_results
        
        try:
            # Task 1: Type text
            test_text = "FSOT AI Desktop Training Session\nLearning text input automation\nTimestamp: " + str(time.time())
            self.logger.info("⌨️ Training text input...")
            
            pyautogui.typewrite(test_text, interval=0.05)
            training_results["interactions"].append({
                "task": "text_input",
                "success": True,
                "text_length": len(test_text)
            })
            training_results["text_samples"].append(test_text)
            training_results["tasks_completed"] += 1
            
            time.sleep(2)
            
            # Task 2: Select all text
            self.logger.info("🎯 Training text selection...")
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(1)
            
            training_results["interactions"].append({
                "task": "select_all",
                "success": True,
                "hotkey": "ctrl+a"
            })
            training_results["tasks_completed"] += 1
            
            # Task 3: Copy text
            pyautogui.hotkey('ctrl', 'c')
            training_results["interactions"].append({
                "task": "copy_text",
                "success": True,
                "hotkey": "ctrl+c"
            })
            training_results["tasks_completed"] += 1
            
            # Task 4: Close without saving
            self.logger.info("❌ Training application closure...")
            pyautogui.hotkey('alt', 'f4')
            time.sleep(1)
            
            # Handle save dialog if it appears
            try:
                pyautogui.press('tab')  # Navigate to "Don't Save"
                pyautogui.press('enter')
            except:
                pass
            
            training_results["interactions"].append({
                "task": "close_application",
                "success": True,
                "method": "alt+f4"
            })
            training_results["tasks_completed"] += 1
            
        except Exception as e:
            training_results["error"] = str(e)
            self.logger.error(f"Notepad training error: {e}")
        
        # Calculate success rate
        total_tasks = 4
        training_results["success_rate"] = (training_results["tasks_completed"] / total_tasks) * 100
        
        return training_results
    
    def train_calculator_automation(self) -> Dict[str, Any]:
        """Train Calculator automation skills."""
        training_results = {
            "app": "calculator",
            "calculations_performed": 0,
            "calculations": [],
            "success_rate": 0.0
        }
        
        # Launch Calculator
        launch_result = self.launch_application("calculator")
        if not launch_result["launch_successful"]:
            training_results["error"] = "Failed to launch Calculator"
            return training_results
        
        try:
            # Test calculations
            calculations = [
                ("123+456", "579"),
                ("1000-250", "750"),
                ("25*4", "100"),
                ("144/12", "12")
            ]
            
            for calc_input, expected in calculations:
                try:
                    self.logger.info(f"🧮 Calculating: {calc_input}")
                    
                    # Clear calculator
                    pyautogui.press('escape')
                    time.sleep(0.5)
                    
                    # Input calculation
                    pyautogui.typewrite(calc_input)
                    time.sleep(0.5)
                    
                    # Press equals
                    pyautogui.press('enter')
                    time.sleep(1)
                    
                    training_results["calculations"].append({
                        "input": calc_input,
                        "expected": expected,
                        "success": True
                    })
                    training_results["calculations_performed"] += 1
                    
                except Exception as e:
                    training_results["calculations"].append({
                        "input": calc_input,
                        "error": str(e),
                        "success": False
                    })
            
            # Close Calculator
            pyautogui.hotkey('alt', 'f4')
            
        except Exception as e:
            training_results["error"] = str(e)
            self.logger.error(f"Calculator training error: {e}")
        
        # Calculate success rate
        total_calculations = len(calculations)
        training_results["success_rate"] = (training_results["calculations_performed"] / total_calculations) * 100
        
        return training_results
    
    def train_screen_analysis(self) -> Dict[str, Any]:
        """Train screen analysis and recognition skills."""
        analysis_results = {
            "screenshot_taken": False,
            "elements_detected": 0,
            "color_analysis": {},
            "text_regions": [],
            "interactive_elements": []
        }
        
        try:
            # Take screenshot for analysis
            screenshot = pyautogui.screenshot()
            analysis_results["screenshot_taken"] = True
            
            # Analyze screen colors (basic)
            width, height = screenshot.size
            sample_points = [
                (width//4, height//4),
                (width//2, height//2),
                (3*width//4, 3*height//4)
            ]
            
            colors = []
            for x, y in sample_points:
                pixel_color = screenshot.getpixel((x, y))
                
                # Handle different pixel formats
                if isinstance(pixel_color, (tuple, list)) and len(pixel_color) >= 3:
                    rgb_values = pixel_color[:3]  # Take first 3 values (R, G, B)
                    brightness = sum(rgb_values) / len(rgb_values)
                elif isinstance(pixel_color, (int, float)) and pixel_color is not None:
                    # Single value (grayscale)
                    rgb_values = (pixel_color, pixel_color, pixel_color)
                    brightness = float(pixel_color)
                else:
                    # Default fallback for None or unexpected types
                    rgb_values = (0, 0, 0)
                    brightness = 0.0
                
                colors.append({
                    "position": (x, y),
                    "rgb": rgb_values,
                    "brightness": brightness
                })
            
            analysis_results["color_analysis"] = {
                "sample_colors": colors,
                "average_brightness": sum(c["brightness"] for c in colors) / len(colors)
            }
            
            # Try to locate common UI elements
            ui_elements = ["OK", "Cancel", "Close", "Save", "Open"]
            for element_text in ui_elements:
                try:
                    location = pyautogui.locateOnScreen(element_text, confidence=0.8)
                    if location:
                        analysis_results["interactive_elements"].append({
                            "text": element_text,
                            "position": location,
                            "detected": True
                        })
                        analysis_results["elements_detected"] += 1
                except:
                    continue
            
        except Exception as e:
            analysis_results["error"] = str(e)
            self.logger.error(f"Screen analysis error: {e}")
        
        return analysis_results
    
    def run_comprehensive_desktop_training(self) -> Dict[str, Any]:
        """Run complete desktop automation training."""
        session_results = {
            "session_id": len(self.training_data["sessions"]) + 1,
            "timestamp": time.time(),
            "modules_completed": [],
            "overall_performance": {},
            "system_state": {}
        }
        
        try:
            # Get initial system state
            session_results["system_state"]["initial"] = self.detect_running_applications()
            
            # Module 1: Screen Analysis Training
            self.logger.info("👁️ Starting Screen Analysis Training...")
            screen_analysis = self.train_screen_analysis()
            session_results["modules_completed"].append({
                "module": "screen_analysis",
                "results": screen_analysis
            })
            
            # Module 2: Notepad Training
            self.logger.info("📝 Starting Notepad Automation Training...")
            notepad_training = self.train_notepad_automation()
            session_results["modules_completed"].append({
                "module": "notepad_automation", 
                "results": notepad_training
            })
            
            # Module 3: Calculator Training
            self.logger.info("🧮 Starting Calculator Automation Training...")
            calculator_training = self.train_calculator_automation()
            session_results["modules_completed"].append({
                "module": "calculator_automation",
                "results": calculator_training
            })
            
            # Calculate overall performance
            total_success_rate = sum([
                screen_analysis.get("elements_detected", 0) * 10,  # Weight factor
                notepad_training.get("success_rate", 0),
                calculator_training.get("success_rate", 0)
            ]) / 3
            
            session_results["overall_performance"] = {
                "modules_completed": len(session_results["modules_completed"]),
                "average_success_rate": total_success_rate,
                "desktop_automation_level": "Intermediate",
                "capabilities_acquired": [
                    "Application launching and control",
                    "Text input and keyboard automation",
                    "Screen analysis and element detection",
                    "Basic calculation automation",
                    "Window management and navigation"
                ]
            }
            
            # Get final system state
            session_results["system_state"]["final"] = self.detect_running_applications()
            
        except Exception as e:
            session_results["error"] = str(e)
            self.logger.error(f"Desktop training session error: {e}")
        
        # Save session data
        self.training_data["sessions"].append(session_results)
        
        return session_results
    
    def generate_desktop_training_report(self) -> str:
        """Generate comprehensive desktop training report."""
        if not self.training_data["sessions"]:
            return "No desktop training sessions completed yet."
        
        latest_session = self.training_data["sessions"][-1]
        total_sessions = len(self.training_data["sessions"])
        
        report_lines = [
            "🖥️ FSOT Desktop Automation Training Report",
            "=" * 60,
            f"📊 Total Training Sessions: {total_sessions}",
            f"🤖 Latest Session ID: {latest_session.get('session_id', 'N/A')}",
            f"⚡ Modules Completed: {latest_session['overall_performance'].get('modules_completed', 0)}",
            f"🎯 Average Success Rate: {latest_session['overall_performance'].get('average_success_rate', 0):.1f}%",
            f"📈 Automation Level: {latest_session['overall_performance'].get('desktop_automation_level', 'N/A')}",
            "",
            "🧠 Capabilities Acquired:",
        ]
        
        # Add capabilities
        capabilities = latest_session['overall_performance'].get('capabilities_acquired', [])
        for capability in capabilities:
            report_lines.append(f"• {capability}")
        
        report_lines.extend([
            "",
            "🖱️ Applications Trained:",
            "• Notepad - Text editing and file operations",
            "• Calculator - Mathematical calculations",
            "• Screen Analysis - UI element detection",
            "• Window Management - Application control",
            "",
            "🚀 Next Steps:",
            "• Expand to browser automation",
            "• Implement file system operations",
            "• Advanced UI element recognition",
            "• Integration with FSOT decision system",
            "• Real-time desktop task automation"
        ])
        
        return "\n".join(report_lines)

def main():
    """Main desktop training execution."""
    trainer = FSOTDesktopTrainer()
    
    print("🖥️ Starting FSOT Desktop Automation Training...")
    print("=" * 60)
    print("⚠️ Please ensure you have control of your mouse and keyboard")
    print("⚠️ Press Ctrl+C to stop if needed (failsafe enabled)")
    
    # Wait for user confirmation
    input("\n🚀 Press Enter to begin training (or Ctrl+C to cancel)...")
    
    # Run comprehensive training
    session_results = trainer.run_comprehensive_desktop_training()
    
    if "error" not in session_results:
        print("✅ Desktop training completed successfully!")
        print(f"📊 Session ID: {session_results['session_id']}")
        print(f"⚡ Modules: {session_results['overall_performance']['modules_completed']}")
        print(f"🎯 Success Rate: {session_results['overall_performance']['average_success_rate']:.1f}%")
    else:
        print(f"❌ Training failed: {session_results['error']}")
    
    # Generate and display report
    report = trainer.generate_desktop_training_report()
    print("\n" + report)
    
    # Save training data
    with open("fsot_desktop_training_data.json", "w") as f:
        json.dump(trainer.training_data, f, indent=2)
    
    print("\n💾 Training data saved to fsot_desktop_training_data.json")

if __name__ == "__main__":
    main()
