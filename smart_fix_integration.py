#!/usr/bin/env python3
"""
Smart Code Fix Integration System
================================

This system automatically runs error assessment and provides intelligent
fix recommendations after every code update.

Usage:
    python smart_fix_integration.py
    
Features:
- Automatic error detection after code changes
- Intelligent fix recommendations
- Progress tracking
- Next action suggestions
- Integration with development workflow
"""

import os
import sys
import time
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import our assessment systems
try:
    from post_update_assessment import PostUpdateAssessment
    from automated_error_assessment import AutomatedErrorAssessment
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Please ensure both assessment scripts are in the same directory")
    sys.exit(1)

class SmartFixIntegration:
    """Intelligent code fix integration system."""
    
    def __init__(self, workspace_path: Optional[str] = None):
        """Initialize the smart fix integration."""
        self.workspace_path = workspace_path or os.getcwd()
        self.post_update_assessor = PostUpdateAssessment(self.workspace_path)
        self.last_assessment_time = 0
        self.fix_session_active = False
        
    def run_post_fix_assessment(self, fix_type: str = "general") -> Dict[str, Any]:
        """Run assessment after applying fixes."""
        print(f"\n🔧 SMART FIX INTEGRATION - POST-FIX ASSESSMENT")
        print(f"{'='*55}")
        
        # Run the assessment
        report = self.post_update_assessor.run_post_update_assessment(fix_type)
        
        # Generate next fix recommendations
        next_fixes = self._generate_next_fix_recommendations(report)
        
        # Create summary
        summary = {
            "timestamp": report.timestamp,
            "total_errors": report.total_errors,
            "total_warnings": report.total_warnings,
            "fix_type_applied": fix_type,
            "next_recommended_fixes": next_fixes,
            "progress_assessment": self._assess_fix_progress(report),
            "should_continue_fixing": report.total_errors > 0
        }
        
        self._display_fix_summary(summary)
        
        return summary
    
    def _generate_next_fix_recommendations(self, report) -> List[Dict[str, Any]]:
        """Generate specific next fix recommendations."""
        recommendations = []
        
        categories = report.errors_by_category
        
        # Priority-based recommendations
        if categories.get("syntax", 0) > 0:
            recommendations.append({
                "priority": 10,
                "category": "syntax",
                "action": "Fix syntax errors immediately",
                "command": "Check Python syntax in affected files",
                "count": categories["syntax"]
            })
        
        if categories.get("import", 0) > 0:
            recommendations.append({
                "priority": 9,
                "category": "import", 
                "action": "Resolve import issues",
                "command": "pip install missing packages or fix import paths",
                "count": categories["import"]
            })
        
        if categories.get("attribute", 0) > 0:
            recommendations.append({
                "priority": 8,
                "category": "attribute",
                "action": "Fix attribute access errors", 
                "command": "Add missing methods or use setattr()",
                "count": categories["attribute"]
            })
        
        if categories.get("type", 0) > 0:
            recommendations.append({
                "priority": 6,
                "category": "type",
                "action": "Address type mismatches",
                "command": "Add type casting or update annotations", 
                "count": categories["type"]
            })
        
        if categories.get("declaration", 0) > 0:
            recommendations.append({
                "priority": 4,
                "category": "declaration",
                "action": "Clean up declaration conflicts",
                "command": "Rename classes or use unique namespaces",
                "count": categories["declaration"]
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _assess_fix_progress(self, report) -> Dict[str, Any]:
        """Assess progress of fixing efforts."""
        progress = {
            "status": "unknown",
            "message": "",
            "improvement_percentage": 0,
            "errors_remaining": report.total_errors,
            "next_target": ""
        }
        
        if report.progress_since_last:
            change = report.progress_since_last["errors_change"]
            improvement = report.progress_since_last["improvement_percentage"]
            
            if change < 0:
                progress["status"] = "improving"
                progress["message"] = f"Excellent! {abs(change)} fewer errors ({improvement:.1f}% improvement)"
                progress["improvement_percentage"] = improvement
            elif change > 0:
                progress["status"] = "regressing" 
                progress["message"] = f"Warning: {change} more errors detected"
            else:
                progress["status"] = "stable"
                progress["message"] = "No change in error count"
        else:
            progress["status"] = "baseline"
            progress["message"] = "First assessment - establishing baseline"
        
        # Set next target
        if report.total_errors > 10:
            progress["next_target"] = "Reduce errors to under 10"
        elif report.total_errors > 5:
            progress["next_target"] = "Reduce errors to under 5"
        elif report.total_errors > 0:
            progress["next_target"] = "Achieve zero errors"
        else:
            progress["next_target"] = "Optimize warnings and code quality"
        
        return progress
    
    def _display_fix_summary(self, summary: Dict[str, Any]):
        """Display comprehensive fix summary and recommendations."""
        print(f"\n📋 SMART FIX SUMMARY")
        print(f"{'='*30}")
        
        # Current status
        print(f"🎯 Current Status:")
        print(f"   • Errors: {summary['total_errors']}")
        print(f"   • Warnings: {summary['total_warnings']}")
        print(f"   • Fix Applied: {summary['fix_type_applied']}")
        
        # Progress assessment
        progress = summary["progress_assessment"]
        status_emoji = {
            "improving": "📈", "regressing": "📉", 
            "stable": "➡️", "baseline": "📊", "unknown": "❓"
        }.get(progress["status"], "❓")
        
        print(f"\n{status_emoji} Progress: {progress['message']}")
        print(f"🎯 Next Target: {progress['next_target']}")
        
        # Next recommendations
        if summary["next_recommended_fixes"]:
            print(f"\n🔧 NEXT RECOMMENDED FIXES:")
            for i, fix in enumerate(summary["next_recommended_fixes"][:3], 1):
                print(f"   {i}. 🏷️ {fix['category'].upper()} ({fix['count']} issues)")
                print(f"      ⚡ Action: {fix['action']}")
                print(f"      💻 Command: {fix['command']}")
        
        # Continue fixing?
        if summary["should_continue_fixing"]:
            print(f"\n🚀 RECOMMENDATION: Continue fixing session")
            print(f"   Run: python smart_fix_integration.py after next fix")
        else:
            print(f"\n🎉 EXCELLENT: No errors remaining!")
            print(f"   Consider monitoring for new issues during development")
    
    def create_fix_workflow_commands(self) -> List[str]:
        """Create executable commands for common fixes."""
        commands = [
            "# Smart Fix Integration Workflow Commands",
            "",
            "# After importing fixes:",
            "python smart_fix_integration.py --type=import_fix",
            "",
            "# After type fixes:", 
            "python smart_fix_integration.py --type=type_fix",
            "",
            "# After attribute fixes:",
            "python smart_fix_integration.py --type=attribute_fix",
            "",
            "# After dependency updates:",
            "python smart_fix_integration.py --type=dependency_update",
            "",
            "# General assessment:",
            "python smart_fix_integration.py",
            "",
            "# Setup automatic monitoring:",
            "python post_update_assessment.py --setup-hooks"
        ]
        
        return commands
    
    def save_workflow_commands(self):
        """Save workflow commands to file."""
        commands = self.create_fix_workflow_commands()
        
        workflow_file = os.path.join(self.workspace_path, "fix_workflow_commands.txt")
        with open(workflow_file, 'w') as f:
            f.write('\n'.join(commands))
        
        print(f"📄 Workflow commands saved to: {workflow_file}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Code Fix Integration")
    parser.add_argument("--workspace", "-w", help="Workspace path", default=".")
    parser.add_argument("--type", "-t", help="Fix type applied", default="general",
                       choices=["general", "import_fix", "type_fix", "attribute_fix", 
                               "dependency_update", "syntax_fix", "declaration_fix"])
    parser.add_argument("--save-workflow", action="store_true", help="Save workflow commands")
    
    args = parser.parse_args()
    
    # Initialize smart fix integration
    smart_fix = SmartFixIntegration(args.workspace)
    
    if args.save_workflow:
        smart_fix.save_workflow_commands()
        return
    
    # Run post-fix assessment
    summary = smart_fix.run_post_fix_assessment(args.type)
    
    # Generate specific recommendations based on results
    print(f"\n🎯 AUTOMATED NEXT STEPS:")
    
    if summary["should_continue_fixing"]:
        next_fixes = summary["next_recommended_fixes"]
        if next_fixes:
            next_fix = next_fixes[0]  # Highest priority
            print(f"   1. Focus on {next_fix['category']} errors ({next_fix['count']} issues)")
            print(f"   2. {next_fix['command']}")
            print(f"   3. Run: python smart_fix_integration.py --type={next_fix['category']}_fix")
        
        print(f"\n💡 TIP: Each fix iteration should reduce errors by 20-50%")
    else:
        print(f"   1. Excellent work! All errors resolved")
        print(f"   2. Consider setting up continuous monitoring")
        print(f"   3. Focus on code optimization and documentation")
    
    # Return appropriate exit code
    if summary["total_errors"] > 0:
        print(f"\n⚠️ {summary['total_errors']} errors remain - continue fixing session")
        sys.exit(summary["total_errors"])  # Exit code = number of errors
    else:
        print(f"\n✅ All errors resolved! System is ready for production")
        sys.exit(0)

if __name__ == "__main__":
    main()
