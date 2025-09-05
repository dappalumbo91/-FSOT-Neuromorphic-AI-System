#!/usr/bin/env python3
"""
Post-Update Error Assessment Integration
======================================

This script automatically runs comprehensive error assessment after any code update,
providing immediate feedback and fix recommendations.

Usage:
    python post_update_assessment.py
    
Features:
- Automatic triggering after code changes
- Integration with git hooks
- Immediate error analysis and reporting
- Smart fix prioritization
- Progress tracking across updates
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import our automated assessment system
from automated_error_assessment import AutomatedErrorAssessment, AssessmentReport

class PostUpdateAssessment:
    """Manages post-update error assessment workflow."""
    
    def __init__(self, workspace_path: Optional[str] = None):
        """Initialize post-update assessment."""
        self.workspace_path = workspace_path or os.getcwd()
        self.assessor = AutomatedErrorAssessment(self.workspace_path)
        self.config_file = os.path.join(self.workspace_path, ".assessment_config.json")
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load or create assessment configuration."""
        default_config = {
            "auto_run_on_save": True,
            "auto_run_on_commit": True,
            "error_threshold_critical": 0,  # Fail if critical errors > this
            "error_threshold_total": 10,    # Warning if total errors > this
            "assessment_history_limit": 20,
            "notification_settings": {
                "show_progress": True,
                "show_recommendations": True,
                "show_next_actions": True
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
            except Exception as e:
                print(f"⚠️ Error loading config: {e}, using defaults")
        
        return default_config
    
    def _save_config(self):
        """Save current configuration."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving config: {e}")
    
    def run_post_update_assessment(self, update_type: str = "manual") -> AssessmentReport:
        """Run comprehensive assessment after code update."""
        print(f"\n🔄 POST-UPDATE ERROR ASSESSMENT")
        print(f"{'='*50}")
        print(f"📝 Update Type: {update_type}")
        print(f"🕒 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Workspace: {self.workspace_path}")
        
        # Run the assessment
        report = self.assessor.analyze_workspace()
        
        # Analyze results and provide recommendations
        self._analyze_assessment_results(report, update_type)
        
        # Check thresholds and provide warnings/failures
        self._check_error_thresholds(report)
        
        # Generate next action plan
        self._generate_action_plan(report)
        
        return report
    
    def _analyze_assessment_results(self, report: AssessmentReport, update_type: str):
        """Analyze assessment results and provide context-aware feedback."""
        print(f"\n📊 ASSESSMENT RESULTS ANALYSIS:")
        
        # Progress analysis
        if report.progress_since_last:
            progress = report.progress_since_last
            if progress["errors_change"] < 0:
                print(f"✅ IMPROVEMENT: {abs(progress['errors_change'])} fewer errors ({progress['improvement_percentage']:.1f}% improvement)")
            elif progress["errors_change"] > 0:
                print(f"⚠️ REGRESSION: {progress['errors_change']} more errors")
            else:
                print(f"➡️ STABLE: No change in error count")
        
        # Context-specific recommendations
        if update_type == "import_fix":
            self._analyze_import_fixes(report)
        elif update_type == "type_fix":
            self._analyze_type_fixes(report)
        elif update_type == "attribute_fix":
            self._analyze_attribute_fixes(report)
        elif update_type == "dependency_update":
            self._analyze_dependency_updates(report)
        
        # Overall health assessment
        self._assess_code_health(report)
    
    def _analyze_import_fixes(self, report: AssessmentReport):
        """Analyze results of import-related fixes."""
        import_errors = report.errors_by_category.get("import", 0)
        if import_errors == 0:
            print("🎉 IMPORT FIXES: All import errors resolved!")
        elif import_errors < 3:
            print(f"📈 IMPORT FIXES: Good progress, {import_errors} import errors remaining")
        else:
            print(f"🔧 IMPORT FIXES: Need more work, {import_errors} import errors still present")
    
    def _analyze_type_fixes(self, report: AssessmentReport):
        """Analyze results of type-related fixes."""
        type_errors = report.errors_by_category.get("type", 0)
        if type_errors == 0:
            print("🎉 TYPE FIXES: All type errors resolved!")
        elif type_errors < 5:
            print(f"📈 TYPE FIXES: Good progress, {type_errors} type errors remaining")
        else:
            print(f"🔧 TYPE FIXES: Need more work, {type_errors} type errors still present")
    
    def _analyze_attribute_fixes(self, report: AssessmentReport):
        """Analyze results of attribute-related fixes."""
        attr_errors = report.errors_by_category.get("attribute", 0)
        if attr_errors == 0:
            print("🎉 ATTRIBUTE FIXES: All attribute errors resolved!")
        else:
            print(f"🔧 ATTRIBUTE FIXES: {attr_errors} attribute errors remaining")
    
    def _analyze_dependency_updates(self, report: AssessmentReport):
        """Analyze results of dependency updates."""
        import_errors = report.errors_by_category.get("import", 0)
        if import_errors == 0:
            print("🎉 DEPENDENCY UPDATE: No import issues detected!")
        else:
            print(f"📦 DEPENDENCY UPDATE: {import_errors} import issues need attention")
    
    def _assess_code_health(self, report: AssessmentReport):
        """Provide overall code health assessment."""
        total_issues = report.total_errors + report.total_warnings
        
        if total_issues == 0:
            health_status = "🟢 EXCELLENT"
            health_message = "Code is clean and error-free!"
        elif report.total_errors == 0 and report.total_warnings < 5:
            health_status = "🟢 VERY GOOD"
            health_message = "No errors, minimal warnings"
        elif report.total_errors < 3 and total_issues < 10:
            health_status = "🟡 GOOD"
            health_message = "Few errors, manageable issues"
        elif report.total_errors < 10:
            health_status = "🟠 NEEDS WORK"
            health_message = "Several errors need attention"
        else:
            health_status = "🔴 CRITICAL"
            health_message = "High error count, immediate action needed"
        
        print(f"\n🏥 CODE HEALTH: {health_status}")
        print(f"   📋 Status: {health_message}")
        print(f"   📊 Errors: {report.total_errors}, Warnings: {report.total_warnings}")
    
    def _check_error_thresholds(self, report: AssessmentReport):
        """Check if error counts exceed configured thresholds."""
        critical_errors = report.errors_by_severity.get("critical", 0)
        total_errors = report.total_errors
        
        if critical_errors > self.config["error_threshold_critical"]:
            print(f"\n🚨 CRITICAL THRESHOLD EXCEEDED!")
            print(f"   Critical errors: {critical_errors} (threshold: {self.config['error_threshold_critical']})")
            print(f"   ⚠️ Immediate action required!")
        
        if total_errors > self.config["error_threshold_total"]:
            print(f"\n⚠️ ERROR THRESHOLD WARNING!")
            print(f"   Total errors: {total_errors} (threshold: {self.config['error_threshold_total']})")
            print(f"   📝 Consider focused error reduction session")
    
    def _generate_action_plan(self, report: AssessmentReport):
        """Generate specific action plan based on current state."""
        print(f"\n🎯 RECOMMENDED ACTION PLAN:")
        
        # Prioritize actions based on error types and counts
        actions = []
        
        # Critical errors first
        critical_count = report.errors_by_severity.get("critical", 0)
        if critical_count > 0:
            actions.append(f"🔴 URGENT: Fix {critical_count} critical errors immediately")
        
        # Then by category priority
        categories = report.errors_by_category
        if categories.get("import", 0) > 0:
            actions.append(f"📦 HIGH: Resolve {categories['import']} import issues")
        if categories.get("attribute", 0) > 0:
            actions.append(f"🔧 HIGH: Fix {categories['attribute']} attribute errors")
        if categories.get("type", 0) > 3:
            actions.append(f"🏷️ MEDIUM: Address {categories['type']} type errors")
        if categories.get("declaration", 0) > 0:
            actions.append(f"📝 LOW: Clean up {categories['declaration']} declaration warnings")
        
        # If no errors, focus on optimization
        if report.total_errors == 0:
            if report.total_warnings > 0:
                actions.append(f"✨ OPTIMIZE: Clean up {report.total_warnings} warnings for code quality")
            else:
                actions.append("🎉 MAINTAIN: Code is excellent! Continue monitoring")
        
        # Display action plan
        for i, action in enumerate(actions[:5], 1):
            print(f"   {i}. {action}")
        
        # Provide next steps
        print(f"\n🚀 IMMEDIATE NEXT STEPS:")
        if report.total_errors > 0:
            top_category = max(categories.items(), key=lambda x: x[1])[0] if categories else "general"
            print(f"   1. Focus on {top_category} errors (highest count)")
            print(f"   2. Run: python post_update_assessment.py after each fix")
            print(f"   3. Target: Reduce errors by 50% in next iteration")
        else:
            print(f"   1. Monitor for new issues during development")
            print(f"   2. Consider enabling continuous monitoring")
            print(f"   3. Focus on code optimization and documentation")

def create_git_hooks():
    """Create git hooks for automatic assessment."""
    git_dir = os.path.join(os.getcwd(), ".git")
    if not os.path.exists(git_dir):
        print("⚠️ Not a git repository, skipping git hook creation")
        return
    
    hooks_dir = os.path.join(git_dir, "hooks")
    os.makedirs(hooks_dir, exist_ok=True)
    
    # Pre-commit hook
    pre_commit_hook = os.path.join(hooks_dir, "pre-commit")
    hook_content = """#!/bin/sh
# Automated error assessment pre-commit hook
echo "🔍 Running pre-commit error assessment..."
python post_update_assessment.py --type="pre_commit"
"""
    
    with open(pre_commit_hook, 'w') as f:
        f.write(hook_content)
    
    # Make executable (Unix/Linux)
    try:
        os.chmod(pre_commit_hook, 0o755)
        print(f"✅ Created git pre-commit hook: {pre_commit_hook}")
    except:
        print(f"📝 Created git pre-commit hook: {pre_commit_hook} (manual chmod needed)")

def main():
    """Main entry point for post-update assessment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Post-Update Error Assessment")
    parser.add_argument("--workspace", "-w", help="Workspace path", default=".")
    parser.add_argument("--type", "-t", help="Update type", default="manual", 
                       choices=["manual", "import_fix", "type_fix", "attribute_fix", 
                               "dependency_update", "pre_commit", "post_commit"])
    parser.add_argument("--setup-hooks", action="store_true", help="Setup git hooks")
    
    args = parser.parse_args()
    
    if args.setup_hooks:
        create_git_hooks()
        return
    
    # Run post-update assessment
    assessor = PostUpdateAssessment(args.workspace)
    report = assessor.run_post_update_assessment(args.type)
    
    # Return exit code based on results
    if report.errors_by_severity.get("critical", 0) > 0:
        print("\n🚨 Exiting with error code due to critical issues")
        sys.exit(1)
    elif report.total_errors > 10:
        print("\n⚠️ Exiting with warning code due to high error count")
        sys.exit(2)
    else:
        print("\n✅ Assessment completed successfully")
        sys.exit(0)

if __name__ == "__main__":
    main()
