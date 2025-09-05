#!/usr/bin/env python3
"""
Enhanced FSOT 2.0 - Dependency Installation Script
=================================================

Automated installation script for all Enhanced FSOT 2.0 dependencies.
Handles core dependencies, optional components, and Windows-specific packages.

Author: GitHub Copilot
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   Enhanced FSOT 2.0 requires Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_core_dependencies():
    """Install core dependencies"""
    print("\n🔧 Installing core dependencies...")
    
    # Essential packages first (sqlite3 is built into Python)
    essential_packages = [
        "pip --upgrade",
        "setuptools --upgrade", 
        "wheel",
        "numpy",
        "requests",
        "flask",
    ]
    
    success_count = 0
    for package in essential_packages:
        if run_command(f"pip install {package}", f"Installing {package}"):
            success_count += 1
    
    print(f"📊 Essential packages: {success_count}/{len(essential_packages)} installed")
    return success_count == len(essential_packages)

def install_enhanced_requirements():
    """Install from requirements file"""
    print("\n📋 Installing Enhanced FSOT 2.0 requirements...")
    
    requirements_file = Path("requirements_enhanced.txt")
    if not requirements_file.exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    return run_command(
        f"pip install -r {requirements_file}",
        "Installing Enhanced FSOT 2.0 requirements"
    )

def install_windows_requirements():
    """Install Windows-specific requirements"""
    if platform.system() != "Windows":
        print("ℹ️ Skipping Windows-specific packages (not on Windows)")
        return True
    
    print("\n🪟 Installing Windows-specific dependencies...")
    
    windows_file = Path("requirements_windows.txt")
    if not windows_file.exists():
        print(f"❌ Windows requirements file not found: {windows_file}")
        return False
    
    return run_command(
        f"pip install -r {windows_file}",
        "Installing Windows-specific requirements"
    )

def install_optional_dependencies():
    """Install optional dependencies with fallbacks"""
    print("\n🎯 Installing optional dependencies...")
    
    optional_packages = [
        ("opencv-python", "Computer vision capabilities"),
        ("nltk", "Natural language processing"),
        ("pyautogui", "Desktop automation"),
        ("beautifulsoup4", "Web scraping for training"),
        ("matplotlib", "Data visualization"),
        ("torch", "Machine learning (this may take a while)"),
        ("transformers", "Advanced AI models"),
        ("gradio", "Web interface components")
    ]
    
    success_count = 0
    for package, description in optional_packages:
        print(f"\n🔄 Installing {package} for {description}...")
        if run_command(f"pip install {package}", f"Installing {package}"):
            success_count += 1
        else:
            print(f"⚠️ {package} installation failed - some features may not work")
    
    print(f"\n📊 Optional packages: {success_count}/{len(optional_packages)} installed")
    return success_count

def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    try:
        import nltk
        
        # Download required NLTK packages
        nltk_packages = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'wordnet']
        
        for package in nltk_packages:
            try:
                print(f"📥 Downloading NLTK {package}...")
                nltk.download(package, quiet=True)
                print(f"✅ NLTK {package} downloaded")
            except Exception as e:
                print(f"⚠️ NLTK {package} download failed: {e}")
        
        return True
        
    except ImportError:
        print("⚠️ NLTK not available - skipping data download")
        return False

def verify_installations():
    """Verify that key packages are installed and working"""
    print("\n🔍 Verifying installations...")
    
    test_imports = [
        ("numpy", "Scientific computing"),
        ("requests", "HTTP requests"),
        ("flask", "Web framework"),
        ("sqlite3", "Database"),
        ("json", "JSON processing"),
        ("threading", "Multi-threading"),
        ("pathlib", "Path handling"),
        ("datetime", "Date/time handling")
    ]
    
    optional_imports = [
        ("cv2", "OpenCV computer vision"),
        ("nltk", "Natural language processing"),
        ("pyautogui", "Desktop automation"),
        ("bs4", "Web scraping"),
        ("matplotlib", "Data visualization")
    ]
    
    print("\n✅ Core imports:")
    core_success = 0
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"   ✅ {module} - {description}")
            core_success += 1
        except ImportError:
            print(f"   ❌ {module} - {description} - REQUIRED")
    
    print(f"\n🎯 Optional imports:")
    optional_success = 0
    for module, description in optional_imports:
        try:
            __import__(module)
            print(f"   ✅ {module} - {description}")
            optional_success += 1
        except ImportError:
            print(f"   ⚠️ {module} - {description} - Optional feature disabled")
    
    print(f"\n📊 Verification Results:")
    print(f"   Core packages: {core_success}/{len(test_imports)}")
    print(f"   Optional packages: {optional_success}/{len(optional_imports)}")
    
    return core_success == len(test_imports)

def create_environment_info():
    """Create environment information file"""
    print("\n📄 Creating environment information...")
    
    try:
        import json
        from datetime import datetime
        
        env_info = {
            "installation_date": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "pip_version": subprocess.run(["pip", "--version"], capture_output=True, text=True).stdout.strip()
        }
        
        # Get installed packages
        try:
            result = subprocess.run(["pip", "list", "--format=json"], capture_output=True, text=True)
            if result.returncode == 0:
                env_info["installed_packages"] = json.loads(result.stdout)
        except:
            env_info["installed_packages"] = "Could not retrieve package list"
        
        with open("environment_info.json", "w") as f:
            json.dump(env_info, f, indent=2)
        
        print("✅ Environment info saved to environment_info.json")
        return True
        
    except Exception as e:
        print(f"⚠️ Could not create environment info: {e}")
        return False

def main():
    """Main installation process"""
    print("🚀 Enhanced FSOT 2.0 - Dependency Installation")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Installation aborted - incompatible Python version")
        return False
    
    # Install core dependencies
    if not install_core_dependencies():
        print("\n❌ Core dependency installation failed")
        return False
    
    # Install enhanced requirements
    if not install_enhanced_requirements():
        print("\n⚠️ Enhanced requirements installation had issues")
        print("   Some features may not work properly")
    
    # Install Windows requirements if on Windows
    install_windows_requirements()
    
    # Install optional dependencies
    optional_count = install_optional_dependencies()
    
    # Download NLTK data
    download_nltk_data()
    
    # Verify installations
    if not verify_installations():
        print("\n⚠️ Some core packages failed verification")
        print("   System may not work properly")
    
    # Create environment info
    create_environment_info()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 INSTALLATION COMPLETE!")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("   1. Run: python test_all_capabilities.py")
    print("   2. If tests pass, run: python main.py")
    print("   3. For web interface: python main.py --web")
    print("   4. Check environment_info.json for details")
    
    print("\n💡 Troubleshooting:")
    print("   • If imports fail, try: pip install --upgrade <package>")
    print("   • For GPU support, install CUDA-enabled PyTorch")
    print("   • For Windows issues, run as Administrator")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Installation failed with error: {e}")
        sys.exit(1)
