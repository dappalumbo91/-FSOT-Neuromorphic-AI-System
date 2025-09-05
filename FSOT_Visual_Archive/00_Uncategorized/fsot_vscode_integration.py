#!/usr/bin/env python3
"""
FSOT VS Code Integration - One-Click AI Code Analysis
===================================================

Quick commands for VS Code integration:
1. Analyze code and generate report
2. Apply automatic fixes  
3. Open results in VS Code

Usage in VS Code Terminal:
python fsot_vscode_integration.py analyze
python fsot_vscode_integration.py fix
python fsot_vscode_integration.py report
"""

import sys
import subprocess
import json
from pathlib import Path

def run_analysis():
    """Run the AI self-analysis"""
    print("🧠 Running FSOT AI Self-Analysis...")
    
    try:
        result = subprocess.run([
            sys.executable, "fsot_self_analyzer.py"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def apply_fixes():
    """Apply automatic fixes"""
    print("🔧 Applying AI-recommended fixes...")
    
    try:
        result = subprocess.run([
            sys.executable, "fsot_auto_fixer.py"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Auto-fix failed: {e}")
        return False

def show_report():
    """Show the analysis report"""
    report_file = Path("FSOT_Self_Analysis_Report.md")
    
    if report_file.exists():
        print("📄 Opening analysis report...")
        
        # Try to open in VS Code
        try:
            subprocess.run(["code", str(report_file)])
            print(f"✅ Report opened in VS Code: {report_file}")
        except:
            # Fallback to printing
            with open(report_file, 'r') as f:
                print(f.read())
    else:
        print("❌ No report found. Run analysis first!")

def show_stats():
    """Show quick stats from JSON report"""
    json_report = Path("FSOT_Self_Analysis_Report.json")
    
    if json_report.exists():
        with open(json_report, 'r') as f:
            data = json.load(f)
        
        print("📊 FSOT System Health Dashboard")
        print("=" * 40)
        print(f"🏥 Health Score: {data['system_health_score']}%")
        print(f"📁 Files Analyzed: {data['total_files_analyzed']}")
        print(f"🚨 Critical Issues: {data['critical_issues']}")
        print(f"📝 Total Issues: {len(data['issues'])}")
        print(f"⚡ Analysis Time: {data['performance_metrics']['analysis_time']:.2f}s")
        
        if data['high_priority_fixes']:
            print(f"\n🎯 Priority Fixes ({len(data['high_priority_fixes'])}):")
            for fix in data['high_priority_fixes'][:3]:
                print(f"  • {fix}")
    else:
        print("❌ No report data found. Run analysis first!")

def main():
    """Main VS Code integration entry point"""
    
    if len(sys.argv) < 2:
        print("🧠 FSOT AI VS Code Integration")
        print("=" * 40)
        print("Commands:")
        print("  analyze  - Run AI self-analysis")
        print("  fix     - Apply automatic fixes")
        print("  report  - Open detailed report")
        print("  stats   - Show quick statistics")
        print("  all     - Run analysis + fixes + report")
        print()
        print("Example: python fsot_vscode_integration.py analyze")
        return
    
    command = sys.argv[1].lower()
    
    if command == "analyze":
        success = run_analysis()
        if success:
            show_stats()
    
    elif command == "fix":
        apply_fixes()
    
    elif command == "report":
        show_report()
    
    elif command == "stats":
        show_stats()
    
    elif command == "all":
        print("🚀 Running complete AI analysis and fix cycle...")
        
        # Run analysis
        if run_analysis():
            print("\n" + "="*50)
            
            # Show stats
            show_stats()
            print("\n" + "="*50)
            
            # Apply fixes
            apply_fixes()
            print("\n" + "="*50)
            
            # Show report
            show_report()
            
            print("\n🎉 Complete AI analysis cycle finished!")
        else:
            print("❌ Analysis failed, skipping remaining steps")
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Available: analyze, fix, report, stats, all")

if __name__ == "__main__":
    main()
