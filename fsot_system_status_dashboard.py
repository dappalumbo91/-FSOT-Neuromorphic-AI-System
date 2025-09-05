#!/usr/bin/env python3
"""
FSOT System Status Dashboard
===========================
Comprehensive status of all FSOT components and capabilities.
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path

class FSOTSystemDashboard:
    """Comprehensive system status dashboard for FSOT."""
    
    def __init__(self):
        self.workspace_path = Path.cwd()
        self.system_status = {}
        
    def check_file_availability(self) -> dict:
        """Check availability of all FSOT system files."""
        core_files = {
            "fsot_comprehensive_integration_manager.py": "Application Detection & Backup Management",
            "fsot_web_training_pipeline.py": "Web Automation Training System",
            "fsot_desktop_training_pipeline.py": "Desktop Application Control",
            "fsot_backup_dependencies_installer.py": "Backup Package Installation",
            "fsot_fixed_application_coordinator.py": "Application Coordination System",
            "fsot_simple_application_demo.py": "Simple Integration Demo",
            "brain_system.py": "Core Neural System",
            "neural_network.py": "Neural Network Framework",
            "comprehensive_ai_demo.py": "AI Demonstration System"
        }
        
        file_status = {}
        for filename, description in core_files.items():
            file_path = self.workspace_path / filename
            file_status[filename] = {
                "exists": file_path.exists(),
                "description": description,
                "size": file_path.stat().st_size if file_path.exists() else 0,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None
            }
        
        return file_status
    
    def check_python_packages(self) -> dict:
        """Check critical Python package availability."""
        critical_packages = [
            "numpy", "pandas", "matplotlib", "torch", "sklearn",
            "selenium", "requests", "psutil", "pyautogui"
        ]
        
        package_status = {}
        for package in critical_packages:
            try:
                __import__(package)
                package_status[package] = {"available": True, "error": None}
            except ImportError as e:
                package_status[package] = {"available": False, "error": str(e)}
        
        return package_status
    
    def check_application_integration(self) -> dict:
        """Check application integration status."""
        app_paths = {
            "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "firefox": r"C:\Program Files\Mozilla Firefox\firefox.exe", 
            "edge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            "vscode": r"C:\Users\damia\AppData\Local\Programs\Microsoft VS Code\Code.exe",
            "git": r"C:\Program Files\Git\bin\git.exe",
            "powershell": r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
            "cmd": r"C:\Windows\System32\cmd.exe"
        }
        
        app_status = {}
        for app_name, app_path in app_paths.items():
            app_status[app_name] = {
                "available": os.path.exists(app_path),
                "path": app_path,
                "integration_ready": os.path.exists(app_path)
            }
        
        return app_status
    
    def check_system_capabilities(self) -> dict:
        """Check overall system capabilities."""
        capabilities = {
            "web_automation": {
                "description": "Selenium-based web browser automation",
                "requirements": ["selenium", "chrome", "firefox", "edge"],
                "status": "unknown"
            },
            "desktop_automation": {
                "description": "Desktop application control and UI automation", 
                "requirements": ["pyautogui", "psutil"],
                "status": "unknown"
            },
            "development_integration": {
                "description": "Integration with development tools",
                "requirements": ["vscode", "git"],
                "status": "unknown"
            },
            "system_administration": {
                "description": "System-level automation capabilities",
                "requirements": ["powershell", "cmd"],
                "status": "unknown"
            },
            "ai_processing": {
                "description": "AI and machine learning capabilities",
                "requirements": ["torch", "sklearn", "numpy"],
                "status": "unknown"
            }
        }
        
        # Check each capability
        package_status = self.check_python_packages()
        app_status = self.check_application_integration()
        
        for cap_name, cap_info in capabilities.items():
            requirements_met = 0
            total_requirements = len(cap_info["requirements"])
            
            for requirement in cap_info["requirements"]:
                if requirement in package_status and package_status[requirement]["available"]:
                    requirements_met += 1
                elif requirement in app_status and app_status[requirement]["available"]:
                    requirements_met += 1
            
            if requirements_met == total_requirements:
                cap_info["status"] = "fully_ready"
            elif requirements_met > total_requirements * 0.5:
                cap_info["status"] = "partially_ready"
            else:
                cap_info["status"] = "not_ready"
            
            cap_info["readiness_percentage"] = (requirements_met / total_requirements) * 100
        
        return capabilities
    
    def generate_comprehensive_status(self) -> dict:
        """Generate comprehensive system status."""
        print("🔍 Analyzing FSOT System Status...")
        
        status = {
            "timestamp": datetime.now().isoformat(),
            "workspace_path": str(self.workspace_path),
            "files": self.check_file_availability(),
            "packages": self.check_python_packages(),
            "applications": self.check_application_integration(),
            "capabilities": self.check_system_capabilities()
        }
        
        # Calculate overall system health
        file_health = sum(1 for f in status["files"].values() if f["exists"]) / len(status["files"]) * 100
        package_health = sum(1 for p in status["packages"].values() if p["available"]) / len(status["packages"]) * 100
        app_health = sum(1 for a in status["applications"].values() if a["available"]) / len(status["applications"]) * 100
        cap_health = sum(c["readiness_percentage"] for c in status["capabilities"].values()) / len(status["capabilities"])
        
        status["overall_health"] = {
            "file_availability": file_health,
            "package_availability": package_health, 
            "application_integration": app_health,
            "capability_readiness": cap_health,
            "overall_score": (file_health + package_health + app_health + cap_health) / 4
        }
        
        return status
    
    def display_status_dashboard(self):
        """Display comprehensive status dashboard."""
        status = self.generate_comprehensive_status()
        
        print("\n" + "="*60)
        print("🚀 FSOT NEUROMORPHIC AI SYSTEM - STATUS DASHBOARD")
        print("="*60)
        print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Workspace: {self.workspace_path}")
        
        # Overall Health
        health = status["overall_health"]
        overall_score = health["overall_score"]
        
        if overall_score >= 90:
            status_icon = "🎉 EXCELLENT"
            status_color = "GREEN"
        elif overall_score >= 75:
            status_icon = "✅ GOOD"
            status_color = "YELLOW"
        elif overall_score >= 50:
            status_icon = "⚠️ NEEDS ATTENTION"
            status_color = "ORANGE"
        else:
            status_icon = "❌ CRITICAL"
            status_color = "RED"
        
        print(f"\n📊 OVERALL SYSTEM HEALTH: {overall_score:.1f}% - {status_icon}")
        print("-" * 40)
        print(f"📄 Files Available: {health['file_availability']:.1f}%")
        print(f"📦 Packages Ready: {health['package_availability']:.1f}%")
        print(f"🖥️ Apps Integrated: {health['application_integration']:.1f}%")
        print(f"⚡ Capabilities: {health['capability_readiness']:.1f}%")
        
        # File Status
        print(f"\n📄 CORE FILES STATUS:")
        print("-" * 25)
        for filename, info in status["files"].items():
            icon = "✅" if info["exists"] else "❌"
            size_kb = info["size"] / 1024 if info["size"] > 0 else 0
            print(f"{icon} {filename} ({size_kb:.1f}KB)")
        
        # Package Status
        print(f"\n📦 PACKAGE AVAILABILITY:")
        print("-" * 25)
        for package, info in status["packages"].items():
            icon = "✅" if info["available"] else "❌"
            print(f"{icon} {package}")
        
        # Application Status
        print(f"\n🖥️ APPLICATION INTEGRATION:")
        print("-" * 30)
        for app, info in status["applications"].items():
            icon = "✅" if info["available"] else "❌"
            print(f"{icon} {app}")
        
        # Capability Status
        print(f"\n⚡ SYSTEM CAPABILITIES:")
        print("-" * 25)
        for cap_name, cap_info in status["capabilities"].items():
            if cap_info["status"] == "fully_ready":
                icon = "🎯 READY"
            elif cap_info["status"] == "partially_ready":
                icon = "⚡ PARTIAL"
            else:
                icon = "❌ NOT READY"
            
            print(f"{icon} {cap_name.replace('_', ' ').title()}: {cap_info['readiness_percentage']:.0f}%")
            print(f"    {cap_info['description']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 20)
        
        missing_packages = [p for p, info in status["packages"].items() if not info["available"]]
        if missing_packages:
            print(f"📦 Install missing packages: {', '.join(missing_packages)}")
        
        missing_files = [f for f, info in status["files"].items() if not info["exists"]]
        if missing_files:
            print(f"📄 Check missing files: {', '.join(missing_files)}")
        
        not_ready_caps = [c for c, info in status["capabilities"].items() if info["status"] == "not_ready"]
        if not_ready_caps:
            print(f"⚡ Focus on capabilities: {', '.join(not_ready_caps)}")
        
        if overall_score >= 80:
            print("🚀 System ready for advanced automation and training!")
        elif overall_score >= 60:
            print("✅ System ready for basic operations with some limitations")
        else:
            print("⚠️ System needs setup before full operation")
        
        print("\n" + "="*60)
        
        # Save detailed status to JSON
        status_file = f"FSOT_System_Status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Detailed status saved to: {status_file}")
        
        return status

def main():
    """Main dashboard execution."""
    dashboard = FSOTSystemDashboard()
    return dashboard.display_status_dashboard()

if __name__ == "__main__":
    main()
