#!/usr/bin/env python3
"""
FSOT 2.0 Quick Start Script
Simple way to get the system running
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8, 0):
        print("❌ Python 3.8+ required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("💡 Try running: pip install -r requirements.txt")
        return False

def run_system(mode="cli"):
    """Run the FSOT system"""
    print(f"🚀 Starting FSOT 2.0 system in {mode} mode...")
    
    try:
        if mode == "web":
            subprocess.run([sys.executable, "main.py", "--web"], check=True)
        else:
            subprocess.run([sys.executable, "main.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start system: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 System stopped by user")
        return True

def main():
    """Main startup function"""
    print("🧠⚡ FSOT 2.0 NEUROMORPHIC AI SYSTEM - QUICK START")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ main.py not found")
        print("💡 Please run this script from the FSOT_Clean_System directory")
        return 1
    
    # Ask user what they want to do
    print("\nWhat would you like to do?")
    print("1. 🔧 Install dependencies only")
    print("2. 💻 Run CLI interface")
    print("3. 🌐 Run web interface")
    print("4. 🧪 Run tests")
    print("5. ❌ Exit")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    
    if choice == "1":
        install_dependencies()
    elif choice == "2":
        if install_dependencies():
            run_system("cli")
    elif choice == "3":
        if install_dependencies():
            run_system("web")
    elif choice == "4":
        print("🧪 Running tests...")
        try:
            subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Tests failed or pytest not installed")
            print("💡 Install pytest: pip install pytest pytest-asyncio")
    elif choice == "5":
        print("👋 Goodbye!")
        return 0
    else:
        print("❌ Invalid choice")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
