#!/usr/bin/env python3
"""
Free Desktop Control System for Enhanced FSOT 2.0
=================================================

Desktop automation and control using free alternatives to paid services.
Uses pyautogui, win32 APIs, and other free libraries for comprehensive desktop control.

Author: GitHub Copilot
"""

import os
import sys
import time
import subprocess
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import json
from pathlib import Path

# Desktop automation imports (all free)
try:
    import pyautogui
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

try:
    import pygetwindow as gw
    WINDOW_CONTROL_AVAILABLE = True
except ImportError:
    WINDOW_CONTROL_AVAILABLE = False

# Windows-specific imports
if sys.platform == "win32":
    try:
        import win32gui
        import win32process
        import win32api
        import win32con
        WIN32_AVAILABLE = True
    except ImportError:
        WIN32_AVAILABLE = False
else:
    WIN32_AVAILABLE = False

# Screen capture
try:
    from PIL import Image, ImageGrab
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FreeDesktopController:
    """Free desktop automation and control system"""
    
    def __init__(self):
        self.is_active = False
        self.safety_enabled = True
        self.automation_log = []
        self.max_log_entries = 1000
        
        # Available capabilities
        self.capabilities = {
            "mouse_control": PYAUTOGUI_AVAILABLE,
            "keyboard_control": PYAUTOGUI_AVAILABLE,
            "window_management": WINDOW_CONTROL_AVAILABLE or WIN32_AVAILABLE,
            "screen_capture": PIL_AVAILABLE or PYAUTOGUI_AVAILABLE,
            "application_launch": True,  # Always available via subprocess
            "file_operations": True,  # Always available via pathlib/os
            "system_monitoring": WIN32_AVAILABLE if sys.platform == "win32" else True
        }
        
        # Screen information
        self.screen_info = self._get_screen_info()
        
        # Application database (common applications)
        self.app_database = self._build_app_database()
        
        logger.info(f"Desktop Controller initialized with capabilities: {list(self.capabilities.keys())}")
    
    def _get_screen_info(self) -> Dict[str, Any]:
        """Get screen information"""
        try:
            if PYAUTOGUI_AVAILABLE:
                size = pyautogui.size()
                return {
                    "width": size.width,
                    "height": size.height,
                    "center": (size.width // 2, size.height // 2)
                }
            else:
                # Fallback
                return {
                    "width": 1920,
                    "height": 1080,
                    "center": (960, 540)
                }
        except Exception as e:
            logger.error(f"Error getting screen info: {e}")
            return {"width": 1920, "height": 1080, "center": (960, 540)}
    
    def _build_app_database(self) -> Dict[str, Dict[str, Any]]:
        """Build database of common applications"""
        apps = {
            # Web browsers
            "chrome": {
                "paths": [
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
                ],
                "keywords": ["browser", "web", "internet", "chrome", "google"],
                "category": "browser"
            },
            "firefox": {
                "paths": [
                    r"C:\Program Files\Mozilla Firefox\firefox.exe",
                    r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe"
                ],
                "keywords": ["browser", "web", "internet", "firefox", "mozilla"],
                "category": "browser"
            },
            "edge": {
                "paths": [
                    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
                ],
                "keywords": ["browser", "web", "internet", "edge", "microsoft"],
                "category": "browser"
            },
            
            # Text editors
            "notepad": {
                "paths": [r"C:\Windows\System32\notepad.exe"],
                "keywords": ["text", "editor", "notepad", "write"],
                "category": "editor"
            },
            "vscode": {
                "paths": [
                    r"C:\Users\{user}\AppData\Local\Programs\Microsoft VS Code\Code.exe",
                    r"C:\Program Files\Microsoft VS Code\Code.exe"
                ],
                "keywords": ["code", "editor", "vscode", "visual studio", "programming"],
                "category": "editor"
            },
            
            # File managers
            "explorer": {
                "paths": [r"C:\Windows\explorer.exe"],
                "keywords": ["file", "folder", "explorer", "browse"],
                "category": "file_manager"
            },
            
            # System utilities
            "calculator": {
                "paths": [r"C:\Windows\System32\calc.exe"],
                "keywords": ["calculator", "calc", "math"],
                "category": "utility"
            },
            "cmd": {
                "paths": [r"C:\Windows\System32\cmd.exe"],
                "keywords": ["command", "cmd", "terminal", "console"],
                "category": "utility"
            },
            "powershell": {
                "paths": [
                    r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
                ],
                "keywords": ["powershell", "terminal", "console", "shell"],
                "category": "utility"
            }
        }
        
        # Expand user paths
        username = os.getenv("USERNAME", "User")
        for app_name, app_info in apps.items():
            expanded_paths = []
            for path in app_info["paths"]:
                expanded_path = path.replace("{user}", username)
                expanded_paths.append(expanded_path)
            apps[app_name]["paths"] = expanded_paths
        
        return apps
    
    def log_action(self, action_type: str, details: str, success: bool = True):
        """Log desktop automation action"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action_type": action_type,
            "details": details,
            "success": success
        }
        
        self.automation_log.append(entry)
        
        # Keep log size manageable
        if len(self.automation_log) > self.max_log_entries:
            self.automation_log = self.automation_log[-self.max_log_entries:]
        
        logger.info(f"Desktop action: {action_type} - {details} - {'Success' if success else 'Failed'}")
    
    def enable_safety_mode(self, enabled: bool = True):
        """Enable/disable safety features"""
        self.safety_enabled = enabled
        if enabled:
            logger.info("Desktop control safety mode ENABLED")
        else:
            logger.warning("Desktop control safety mode DISABLED")
    
    def activate(self):
        """Activate desktop control system"""
        self.is_active = True
        self.log_action("system", "Desktop control system activated")
    
    def deactivate(self):
        """Deactivate desktop control system"""
        self.is_active = False
        self.log_action("system", "Desktop control system deactivated")
    
    # Mouse Control
    def move_mouse(self, x: int, y: int, duration: float = 0.5) -> bool:
        """Move mouse to coordinates"""
        if not self.is_active or not PYAUTOGUI_AVAILABLE:
            return False
        
        try:
            # Safety check
            if self.safety_enabled:
                if x < 0 or y < 0 or x > self.screen_info["width"] or y > self.screen_info["height"]:
                    self.log_action("mouse_move", f"Blocked unsafe coordinates: ({x}, {y})", False)
                    return False
            
            pyautogui.moveTo(x, y, duration=duration)
            self.log_action("mouse_move", f"Moved to ({x}, {y})")
            return True
            
        except Exception as e:
            self.log_action("mouse_move", f"Error: {e}", False)
            return False
    
    def click_mouse(self, x: Optional[int] = None, y: Optional[int] = None, 
                   button: str = "left", clicks: int = 1) -> bool:
        """Click mouse at coordinates or current position"""
        if not self.is_active or not PYAUTOGUI_AVAILABLE:
            return False
        
        try:
            if x is not None and y is not None:
                pyautogui.click(x, y, clicks=clicks, button=button)
                self.log_action("mouse_click", f"Clicked {button} at ({x}, {y}) x{clicks}")
            else:
                pyautogui.click(clicks=clicks, button=button)
                self.log_action("mouse_click", f"Clicked {button} x{clicks}")
            return True
            
        except Exception as e:
            self.log_action("mouse_click", f"Error: {e}", False)
            return False
    
    def drag_mouse(self, x1: int, y1: int, x2: int, y2: int, duration: float = 1.0) -> bool:
        """Drag mouse from one point to another"""
        if not self.is_active or not PYAUTOGUI_AVAILABLE:
            return False
        
        try:
            pyautogui.drag(x2 - x1, y2 - y1, duration=duration, button='left')
            self.log_action("mouse_drag", f"Dragged from ({x1}, {y1}) to ({x2}, {y2})")
            return True
            
        except Exception as e:
            self.log_action("mouse_drag", f"Error: {e}", False)
            return False
    
    # Keyboard Control
    def type_text(self, text: str, interval: float = 0.1) -> bool:
        """Type text"""
        if not self.is_active or not PYAUTOGUI_AVAILABLE:
            return False
        
        try:
            # Safety check for sensitive content
            if self.safety_enabled:
                sensitive_keywords = ["password", "credit card", "ssn", "social security"]
                if any(keyword in text.lower() for keyword in sensitive_keywords):
                    self.log_action("type_text", "Blocked potentially sensitive text", False)
                    return False
            
            pyautogui.write(text, interval=interval)
            self.log_action("type_text", f"Typed text (length: {len(text)})")
            return True
            
        except Exception as e:
            self.log_action("type_text", f"Error: {e}", False)
            return False
    
    def press_key(self, key: str, presses: int = 1) -> bool:
        """Press keyboard key(s)"""
        if not self.is_active or not PYAUTOGUI_AVAILABLE:
            return False
        
        try:
            pyautogui.press(key, presses=presses)
            self.log_action("key_press", f"Pressed {key} x{presses}")
            return True
            
        except Exception as e:
            self.log_action("key_press", f"Error: {e}", False)
            return False
    
    def key_combination(self, *keys) -> bool:
        """Press key combination (e.g., ctrl+c)"""
        if not self.is_active or not PYAUTOGUI_AVAILABLE:
            return False
        
        try:
            pyautogui.hotkey(*keys)
            self.log_action("key_combo", f"Pressed {'+'.join(keys)}")
            return True
            
        except Exception as e:
            self.log_action("key_combo", f"Error: {e}", False)
            return False
    
    # Screen Capture
    def take_screenshot(self, save_path: Optional[str] = None) -> Optional[str]:
        """Take a screenshot"""
        if not self.is_active:
            return None
        
        try:
            if PIL_AVAILABLE:
                screenshot = ImageGrab.grab()
            elif PYAUTOGUI_AVAILABLE:
                screenshot = pyautogui.screenshot()
            else:
                self.log_action("screenshot", "No screenshot capability available", False)
                return None
            
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"screenshot_{timestamp}.png"
            
            screenshot.save(save_path)
            self.log_action("screenshot", f"Screenshot saved: {save_path}")
            return save_path
            
        except Exception as e:
            self.log_action("screenshot", f"Error: {e}", False)
            return None
    
    def find_image_on_screen(self, image_path: str, confidence: float = 0.8) -> Optional[Tuple[int, int]]:
        """Find image on screen and return coordinates"""
        if not self.is_active or not PYAUTOGUI_AVAILABLE:
            return None
        
        try:
            location = pyautogui.locateOnScreen(image_path, confidence=confidence)
            if location:
                center = pyautogui.center(location)
                self.log_action("image_find", f"Found image at ({center.x}, {center.y})")
                return (center.x, center.y)
            else:
                self.log_action("image_find", f"Image not found: {image_path}", False)
                return None
                
        except Exception as e:
            self.log_action("image_find", f"Error: {e}", False)
            return None
    
    # Window Management
    def get_active_window(self) -> Optional[Dict[str, Any]]:
        """Get information about the active window"""
        try:
            if WIN32_AVAILABLE:
                hwnd = win32gui.GetForegroundWindow()
                window_text = win32gui.GetWindowText(hwnd)
                rect = win32gui.GetWindowRect(hwnd)
                
                return {
                    "title": window_text,
                    "handle": hwnd,
                    "x": rect[0],
                    "y": rect[1],
                    "width": rect[2] - rect[0],
                    "height": rect[3] - rect[1]
                }
            elif WINDOW_CONTROL_AVAILABLE:
                active_window = gw.getActiveWindow()
                if active_window:
                    return {
                        "title": active_window.title,
                        "x": active_window.left,
                        "y": active_window.top,
                        "width": active_window.width,
                        "height": active_window.height
                    }
            
            return None
            
        except Exception as e:
            self.log_action("get_window", f"Error: {e}", False)
            return None
    
    def list_windows(self) -> List[Dict[str, Any]]:
        """List all visible windows"""
        windows = []
        
        try:
            if WINDOW_CONTROL_AVAILABLE:
                for window in gw.getAllWindows():
                    if window.title and window.visible:
                        windows.append({
                            "title": window.title,
                            "x": window.left,
                            "y": window.top,
                            "width": window.width,
                            "height": window.height
                        })
            elif WIN32_AVAILABLE:
                def enum_windows_proc(hwnd, windows_list):
                    if win32gui.IsWindowVisible(hwnd):
                        window_text = win32gui.GetWindowText(hwnd)
                        if window_text:
                            rect = win32gui.GetWindowRect(hwnd)
                            windows_list.append({
                                "title": window_text,
                                "handle": hwnd,
                                "x": rect[0],
                                "y": rect[1],
                                "width": rect[2] - rect[0],
                                "height": rect[3] - rect[1]
                            })
                    return True
                
                win32gui.EnumWindows(enum_windows_proc, windows)
            
            self.log_action("list_windows", f"Found {len(windows)} windows")
            return windows
            
        except Exception as e:
            self.log_action("list_windows", f"Error: {e}", False)
            return []
    
    def focus_window(self, title_pattern: str) -> bool:
        """Focus window by title pattern"""
        try:
            if WINDOW_CONTROL_AVAILABLE:
                windows = gw.getWindowsWithTitle(title_pattern)
                if windows:
                    windows[0].activate()
                    self.log_action("focus_window", f"Focused window: {title_pattern}")
                    return True
            elif WIN32_AVAILABLE:
                def enum_windows_proc(hwnd, lParam):
                    window_text = win32gui.GetWindowText(hwnd)
                    if title_pattern.lower() in window_text.lower():
                        win32gui.SetForegroundWindow(hwnd)
                        return False  # Stop enumeration
                    return True
                
                win32gui.EnumWindows(enum_windows_proc, None)
                self.log_action("focus_window", f"Attempted to focus: {title_pattern}")
                return True
            
            self.log_action("focus_window", f"Window not found: {title_pattern}", False)
            return False
            
        except Exception as e:
            self.log_action("focus_window", f"Error: {e}", False)
            return False
    
    # Application Launch
    def launch_application(self, app_name: str, args: List[str] = None) -> bool:
        """Launch application by name or path"""
        try:
            app_path = None
            
            # Check if it's a known application
            if app_name.lower() in self.app_database:
                app_info = self.app_database[app_name.lower()]
                for path in app_info["paths"]:
                    if os.path.exists(path):
                        app_path = path
                        break
            
            # If not found, try as direct path
            if app_path is None:
                if os.path.exists(app_name):
                    app_path = app_name
                else:
                    # Try to find in PATH
                    import shutil
                    app_path = shutil.which(app_name)
            
            if app_path is None:
                self.log_action("launch_app", f"Application not found: {app_name}", False)
                return False
            
            # Launch the application
            cmd = [app_path]
            if args:
                cmd.extend(args)
            
            subprocess.Popen(cmd, shell=False)
            self.log_action("launch_app", f"Launched: {app_name}")
            return True
            
        except Exception as e:
            self.log_action("launch_app", f"Error launching {app_name}: {e}", False)
            return False
    
    def find_application(self, query: str) -> List[Dict[str, Any]]:
        """Find applications matching query"""
        results = []
        query_lower = query.lower()
        
        for app_name, app_info in self.app_database.items():
            # Check if query matches app name or keywords
            if (query_lower in app_name or 
                any(query_lower in keyword for keyword in app_info["keywords"])):
                
                # Check if application exists
                app_exists = any(os.path.exists(path) for path in app_info["paths"])
                
                if app_exists:
                    results.append({
                        "name": app_name,
                        "category": app_info["category"],
                        "keywords": app_info["keywords"],
                        "paths": app_info["paths"]
                    })
        
        self.log_action("find_app", f"Found {len(results)} apps for query: {query}")
        return results
    
    # File Operations
    def open_file(self, file_path: str) -> bool:
        """Open file with default application"""
        try:
            if not os.path.exists(file_path):
                self.log_action("open_file", f"File not found: {file_path}", False)
                return False
            
            if sys.platform == "win32":
                os.startfile(file_path)
            else:
                subprocess.call(["xdg-open", file_path])
            
            self.log_action("open_file", f"Opened: {file_path}")
            return True
            
        except Exception as e:
            self.log_action("open_file", f"Error opening {file_path}: {e}", False)
            return False
    
    def open_folder(self, folder_path: str) -> bool:
        """Open folder in file explorer"""
        try:
            if not os.path.exists(folder_path):
                self.log_action("open_folder", f"Folder not found: {folder_path}", False)
                return False
            
            if sys.platform == "win32":
                subprocess.Popen(f'explorer "{folder_path}"')
            else:
                subprocess.Popen(["xdg-open", folder_path])
            
            self.log_action("open_folder", f"Opened folder: {folder_path}")
            return True
            
        except Exception as e:
            self.log_action("open_folder", f"Error opening folder {folder_path}: {e}", False)
            return False
    
    # System Information
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        try:
            import platform
            import psutil
            
            info = {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "architecture": platform.architecture()[0],
                "processor": platform.processor(),
                "hostname": platform.node(),
                "screen_resolution": f"{self.screen_info['width']}x{self.screen_info['height']}",
                "memory_total": f"{psutil.virtual_memory().total // (1024**3)} GB",
                "memory_available": f"{psutil.virtual_memory().available // (1024**3)} GB",
                "disk_usage": {},
                "capabilities": self.capabilities
            }
            
            # Get disk usage for main drives
            for disk in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(disk.mountpoint)
                    info["disk_usage"][disk.device] = {
                        "total": f"{usage.total // (1024**3)} GB",
                        "used": f"{usage.used // (1024**3)} GB",
                        "free": f"{usage.free // (1024**3)} GB"
                    }
                except:
                    pass
            
            self.log_action("system_info", "Retrieved system information")
            return info
            
        except Exception as e:
            self.log_action("system_info", f"Error: {e}", False)
            return {"error": str(e)}
    
    # High-level Actions
    def perform_text_editing_action(self, action: str, text: str = "") -> bool:
        """Perform common text editing actions"""
        actions = {
            "select_all": lambda: self.key_combination("ctrl", "a"),
            "copy": lambda: self.key_combination("ctrl", "c"),
            "paste": lambda: self.key_combination("ctrl", "v"),
            "cut": lambda: self.key_combination("ctrl", "x"),
            "undo": lambda: self.key_combination("ctrl", "z"),
            "redo": lambda: self.key_combination("ctrl", "y"),
            "save": lambda: self.key_combination("ctrl", "s"),
            "find": lambda: self.key_combination("ctrl", "f"),
            "replace": lambda: self.key_combination("ctrl", "h"),
            "new": lambda: self.key_combination("ctrl", "n"),
            "type": lambda: self.type_text(text)
        }
        
        if action in actions:
            return actions[action]()
        else:
            self.log_action("text_edit", f"Unknown action: {action}", False)
            return False
    
    def perform_browser_action(self, action: str, url: str = "") -> bool:
        """Perform common browser actions"""
        actions = {
            "new_tab": lambda: self.key_combination("ctrl", "t"),
            "close_tab": lambda: self.key_combination("ctrl", "w"),
            "refresh": lambda: self.press_key("f5"),
            "back": lambda: self.key_combination("alt", "left"),
            "forward": lambda: self.key_combination("alt", "right"),
            "address_bar": lambda: self.key_combination("ctrl", "l"),
            "navigate": lambda: self.key_combination("ctrl", "l") and self.type_text(url) and self.press_key("enter"),
            "search": lambda: self.key_combination("ctrl", "k")
        }
        
        if action in actions:
            return actions[action]()
        else:
            self.log_action("browser_action", f"Unknown action: {action}", False)
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get desktop controller status"""
        return {
            "is_active": self.is_active,
            "safety_enabled": self.safety_enabled,
            "capabilities": self.capabilities,
            "screen_info": self.screen_info,
            "recent_actions": self.automation_log[-10:],
            "total_actions": len(self.automation_log)
        }

# Global instance for easy access
_desktop_controller = None

def get_desktop_controller() -> FreeDesktopController:
    """Get or create global desktop controller instance"""
    global _desktop_controller
    if _desktop_controller is None:
        _desktop_controller = FreeDesktopController()
    return _desktop_controller

if __name__ == "__main__":
    # Test the desktop controller
    print("🖥️ Testing Free Desktop Controller")
    print("=" * 40)
    
    controller = FreeDesktopController()
    
    # Test system info
    print("\n📊 System Information:")
    system_info = controller.get_system_info()
    for key, value in system_info.items():
        if key != "capabilities":
            print(f"  {key}: {value}")
    
    # Test capabilities
    print(f"\n🔧 Available Capabilities:")
    for capability, available in controller.capabilities.items():
        status = "✅" if available else "❌"
        print(f"  {status} {capability}")
    
    # Test application finding
    print(f"\n🔍 Finding browsers:")
    browsers = controller.find_application("browser")
    for browser in browsers:
        print(f"  📱 {browser['name']}: {browser['category']}")
    
    # Test window listing
    print(f"\n🪟 Current Windows:")
    windows = controller.list_windows()
    for window in windows[:5]:  # Show first 5
        print(f"  🔸 {window['title'][:50]}...")
    
    print(f"\n📝 Action Log:")
    for action in controller.automation_log:
        timestamp = action["timestamp"][:19].replace("T", " ")
        status = "✅" if action["success"] else "❌"
        print(f"  [{timestamp}] {status} {action['action_type']}: {action['details']}")
    
    print(f"\n✅ Desktop Controller test completed!")
    print(f"Note: Activate controller with controller.activate() to enable automation")
