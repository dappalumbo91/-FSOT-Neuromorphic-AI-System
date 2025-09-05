#!/usr/bin/env python3
"""
FSOT Simple Application Demo
============================
Demonstrates basic application integration capabilities.
"""

import os
import subprocess
import time
from pathlib import Path

class FSOTSimpleDemo:
    """Simple demonstration of FSOT application integration."""
    
    def __init__(self):
        self.detected_apps = {
            "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "vscode": r"C:\Users\damia\AppData\Local\Programs\Microsoft VS Code\Code.exe",
            "git": r"C:\Program Files\Git\bin\git.exe",
            "powershell": r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
        }
    
    def test_application_detection(self):
        """Test if applications are available."""
        print("🔍 Testing Application Detection:")
        print("=" * 40)
        
        available_apps = {}
        for app_name, app_path in self.detected_apps.items():
            if os.path.exists(app_path):
                available_apps[app_name] = app_path
                print(f"✅ {app_name}: Available")
            else:
                print(f"❌ {app_name}: Not found")
        
        return available_apps
    
    def test_git_integration(self):
        """Test Git integration capabilities."""
        print("\n🔧 Testing Git Integration:")
        print("=" * 30)
        
        try:
            # Test git status
            result = subprocess.run(
                ["git", "status", "--porcelain"], 
                capture_output=True, text=True, cwd="."
            )
            
            if result.returncode == 0:
                print("✅ Git repository detected")
                print(f"📊 Modified files: {len(result.stdout.strip().split() if result.stdout.strip() else [])}")
                return True
            else:
                print("❌ Not a git repository or git not available")
                return False
                
        except Exception as e:
            print(f"❌ Git test failed: {e}")
            return False
    
    def test_powershell_integration(self):
        """Test PowerShell integration capabilities."""
        print("\n⚡ Testing PowerShell Integration:")
        print("=" * 35)
        
        try:
            # Test basic PowerShell command
            result = subprocess.run([
                "powershell", "-Command", "Get-ComputerInfo | Select-Object WindowsProductName"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("✅ PowerShell integration working")
                print(f"📋 System info retrieved: {len(result.stdout.strip())} characters")
                return True
            else:
                print("❌ PowerShell command failed")
                return False
                
        except Exception as e:
            print(f"❌ PowerShell test failed: {e}")
            return False
    
    def test_backup_dependencies(self):
        """Test if backup dependencies are available."""
        print("\n📦 Testing Backup Dependencies:")
        print("=" * 32)
        
        backup_packages = [
            "selenium", "requests", "beautifulsoup4", "pandas", 
            "numpy", "matplotlib", "torch", "sklearn"
        ]
        
        available_packages = []
        for package in backup_packages:
            try:
                __import__(package)
                available_packages.append(package)
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package}: Not installed")
        
        print(f"\n📊 Backup Dependencies: {len(available_packages)}/{len(backup_packages)} available")
        return available_packages
    
    def run_comprehensive_demo(self):
        """Run comprehensive demonstration."""
        print("🚀 FSOT Application Integration Demo")
        print("=" * 50)
        
        # Test application detection
        available_apps = self.test_application_detection()
        
        # Test integrations
        git_working = self.test_git_integration()
        powershell_working = self.test_powershell_integration()
        backup_deps = self.test_backup_dependencies()
        
        # Generate summary
        print("\n📊 DEMO SUMMARY:")
        print("=" * 20)
        print(f"🖥️ Applications Detected: {len(available_apps)}/4")
        print(f"🔧 Git Integration: {'✅ Working' if git_working else '❌ Issues'}")
        print(f"⚡ PowerShell Integration: {'✅ Working' if powershell_working else '❌ Issues'}")
        print(f"📦 Backup Dependencies: {len(backup_deps)}/8 available")
        
        # Overall status
        total_score = len(available_apps) + (2 if git_working else 0) + (2 if powershell_working else 0) + len(backup_deps)
        max_score = 4 + 2 + 2 + 8  # 16 total
        percentage = (total_score / max_score) * 100
        
        print(f"\n🎯 Overall Integration Score: {total_score}/{max_score} ({percentage:.1f}%)")
        
        if percentage >= 80:
            print("🎉 EXCELLENT: System ready for advanced automation!")
        elif percentage >= 60:
            print("✅ GOOD: System ready for basic automation")
        else:
            print("⚠️ NEEDS WORK: Install missing dependencies")
        
        return {
            "applications": available_apps,
            "git_working": git_working,
            "powershell_working": powershell_working,
            "backup_dependencies": backup_deps,
            "score": percentage
        }

def main():
    """Main demo execution."""
    demo = FSOTSimpleDemo()
    results = demo.run_comprehensive_demo()
    
    print(f"\n💾 Demo completed with {results['score']:.1f}% integration score")
    return results

if __name__ == "__main__":
    main()
