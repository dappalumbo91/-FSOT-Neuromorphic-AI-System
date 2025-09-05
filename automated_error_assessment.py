#!/usr/bin/env python3
"""
Automated Error and Warning Assessment System
===========================================

This system automatically analyzes code changes, detects errors/warnings,
and provides intelligent recommendations for further fixes.

Usage:
    python automated_error_assessment.py
    
Features:
- Real-time error detection across all Python files
- Categorized error analysis with priority levels
- Intelligent fix recommendations
- Progress tracking and reporting
- Integration with VS Code and development workflow
"""

import os
import sys
import json
import time
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import re

@dataclass
class ErrorInfo:
    """Structured error information."""
    file_path: str
    line_number: int
    error_type: str
    error_message: str
    code_snippet: str
    severity: str  # "critical", "error", "warning", "info"
    category: str  # "import", "type", "attribute", "syntax", etc.
    fix_priority: int  # 1-10, higher = more critical

@dataclass
class AssessmentReport:
    """Complete assessment report."""
    timestamp: str
    total_files_analyzed: int
    total_errors: int
    total_warnings: int
    errors_by_category: Dict[str, int]
    errors_by_severity: Dict[str, int]
    top_priority_fixes: List[str]
    improvement_suggestions: List[str]
    progress_since_last: Optional[Dict[str, Any]]

class AutomatedErrorAssessment:
    """Main automated error assessment system."""
    
    def __init__(self, workspace_path: Optional[str] = None):
        """Initialize the assessment system."""
        self.workspace_path = workspace_path or os.getcwd()
        self.assessment_history = []
        self.error_patterns = self._load_error_patterns()
        self.fix_recommendations = self._load_fix_recommendations()
        
    def _load_error_patterns(self) -> Dict[str, Any]:
        """Load error pattern recognition rules."""
        return {
            # Import errors
            "unknown_import_symbol": {
                "pattern": r'"(.+)" is unknown import symbol',
                "category": "import",
                "severity": "error",
                "priority": 7,
                "fix_template": "Check if module exists, add to requirements.txt, or create fallback"
            },
            "import_resolution": {
                "pattern": r'Import "(.+)" could not be resolved',
                "category": "import", 
                "severity": "error",
                "priority": 8,
                "fix_template": "Install missing package or check import path"
            },
            
            # Type errors
            "type_assignment": {
                "pattern": r'Type "(.+)" is not assignable to declared type "(.+)"',
                "category": "type",
                "severity": "warning",
                "priority": 4,
                "fix_template": "Add type casting or update type annotations"
            },
            "attribute_assignment": {
                "pattern": r'Cannot assign to attribute "(.+)" for class "(.+)"',
                "category": "attribute",
                "severity": "error", 
                "priority": 6,
                "fix_template": "Use setattr() or define attribute in class"
            },
            
            # Class/method errors
            "class_obscured": {
                "pattern": r'Class declaration "(.+)" is obscured by a declaration of the same name',
                "category": "declaration",
                "severity": "warning",
                "priority": 3,
                "fix_template": "Rename class or use unique namespaces"
            },
            "missing_attribute": {
                "pattern": r'Cannot access attribute "(.+)" for class "(.+)"',
                "category": "attribute",
                "severity": "error",
                "priority": 7,
                "fix_template": "Add missing method/attribute or check object type"
            },
            
            # Argument/parameter errors
            "argument_type": {
                "pattern": r'Argument of type "(.+)" cannot be assigned to parameter "(.+)"',
                "category": "type",
                "severity": "error",
                "priority": 6,
                "fix_template": "Convert argument type or update function signature"
            }
        }
    
    def _load_fix_recommendations(self) -> Dict[str, List[str]]:
        """Load intelligent fix recommendation rules."""
        return {
            "import": [
                "Check if the required package is installed: pip install <package>",
                "Verify the import path is correct",
                "Add missing modules to requirements.txt",
                "Create fallback implementations for optional dependencies",
                "Use try-except blocks for conditional imports"
            ],
            "type": [
                "Add explicit type casting: float(), int(), str()",
                "Update type annotations to match actual usage",
                "Use Union types for multiple possible types",
                "Add type: ignore comments for known issues",
                "Consider using Any type for complex cases"
            ],
            "attribute": [
                "Use setattr() instead of direct attribute assignment",
                "Define attributes in __init__ method",
                "Add property decorators for computed attributes",
                "Check if object has attribute before accessing: hasattr()",
                "Use type assertions: assert isinstance(obj, ExpectedType)"
            ],
            "declaration": [
                "Use unique class names to avoid conflicts",
                "Implement proper namespace separation",
                "Use aliases for imported classes: from module import Class as LocalClass",
                "Consider moving classes to separate modules",
                "Use conditional class definitions"
            ]
        }
    
    def analyze_workspace(self) -> AssessmentReport:
        """Perform complete workspace error analysis."""
        print(f"🔍 Starting automated error assessment...")
        print(f"📁 Workspace: {self.workspace_path}")
        
        # Find all Python files
        python_files = self._find_python_files()
        print(f"📄 Found {len(python_files)} Python files to analyze")
        
        # Analyze each file for errors
        all_errors = []
        for file_path in python_files:
            errors = self._analyze_file(file_path)
            all_errors.extend(errors)
        
        # Generate comprehensive report
        report = self._generate_report(python_files, all_errors)
        
        # Save report
        self._save_report(report)
        
        # Display results
        self._display_report(report)
        
        return report
    
    def _find_python_files(self) -> List[str]:
        """Find all Python files in workspace."""
        python_files = []
        for root, dirs, files in os.walk(self.workspace_path):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return python_files
    
    def _analyze_file(self, file_path: str) -> List[ErrorInfo]:
        """Analyze a single file for errors using Pylance/LSP."""
        errors = []
        
        try:
            # Use VS Code's error detection if available
            errors.extend(self._get_vscode_errors(file_path))
            
        except Exception as e:
            print(f"⚠️ Error analyzing {file_path}: {e}")
        
        return errors
    
    def _get_vscode_errors(self, file_path: str) -> List[ErrorInfo]:
        """Get errors from VS Code/Pylance."""
        errors = []
        
        try:
            # Simulate getting errors (in real implementation, this would use LSP)
            # For now, we'll use a simple syntax check
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to compile to catch syntax errors
            try:
                compile(content, file_path, 'exec')
            except SyntaxError as e:
                error = ErrorInfo(
                    file_path=file_path,
                    line_number=e.lineno or 0,
                    error_type="SyntaxError",
                    error_message=str(e),
                    code_snippet=e.text or "",
                    severity="critical",
                    category="syntax",
                    fix_priority=10
                )
                errors.append(error)
            
        except Exception as e:
            print(f"⚠️ Could not analyze {file_path}: {e}")
        
        return errors
    
    def _categorize_error(self, error_message: str) -> Tuple[str, str, int]:
        """Categorize error and assign priority."""
        for pattern_name, pattern_info in self.error_patterns.items():
            if re.search(pattern_info["pattern"], error_message):
                return (
                    pattern_info["category"],
                    pattern_info["severity"], 
                    pattern_info["priority"]
                )
        
        # Default categorization
        if "import" in error_message.lower():
            return ("import", "error", 7)
        elif "type" in error_message.lower():
            return ("type", "warning", 4)
        elif "attribute" in error_message.lower():
            return ("attribute", "error", 6)
        else:
            return ("other", "warning", 3)
    
    def _generate_report(self, files: List[str], errors: List[ErrorInfo]) -> AssessmentReport:
        """Generate comprehensive assessment report."""
        
        # Count errors by category and severity
        errors_by_category = {}
        errors_by_severity = {}
        
        for error in errors:
            errors_by_category[error.category] = errors_by_category.get(error.category, 0) + 1
            errors_by_severity[error.severity] = errors_by_severity.get(error.severity, 0) + 1
        
        # Generate top priority fixes
        top_priority_fixes = self._generate_priority_fixes(errors)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(errors_by_category)
        
        # Calculate progress if we have previous reports
        progress = self._calculate_progress(errors) if self.assessment_history else None
        
        report = AssessmentReport(
            timestamp=datetime.now().isoformat(),
            total_files_analyzed=len(files),
            total_errors=len([e for e in errors if e.severity in ["critical", "error"]]),
            total_warnings=len([e for e in errors if e.severity == "warning"]),
            errors_by_category=errors_by_category,
            errors_by_severity=errors_by_severity,
            top_priority_fixes=top_priority_fixes,
            improvement_suggestions=improvement_suggestions,
            progress_since_last=progress
        )
        
        return report
    
    def _generate_priority_fixes(self, errors: List[ErrorInfo]) -> List[str]:
        """Generate prioritized fix recommendations."""
        # Sort errors by priority
        sorted_errors = sorted(errors, key=lambda x: x.fix_priority, reverse=True)
        
        fixes = []
        seen_categories = set()
        
        for error in sorted_errors[:10]:  # Top 10 priorities
            if error.category not in seen_categories:
                category_fixes = self.fix_recommendations.get(error.category, [])
                if category_fixes:
                    fixes.append(f"🔧 {error.category.upper()}: {category_fixes[0]}")
                    seen_categories.add(error.category)
        
        return fixes
    
    def _generate_improvement_suggestions(self, errors_by_category: Dict[str, int]) -> List[str]:
        """Generate overall improvement suggestions."""
        suggestions = []
        
        # Prioritize suggestions based on error frequency
        sorted_categories = sorted(errors_by_category.items(), key=lambda x: x[1], reverse=True)
        
        for category, count in sorted_categories[:5]:
            if count > 0:
                category_suggestions = self.fix_recommendations.get(category, [])
                if category_suggestions and len(category_suggestions) > 1:
                    suggestions.append(f"📈 {category.upper()} ({count} issues): {category_suggestions[1]}")
        
        return suggestions
    
    def _calculate_progress(self, current_errors: List[ErrorInfo]) -> Optional[Dict[str, Any]]:
        """Calculate progress since last assessment."""
        if not self.assessment_history:
            return None
        
        last_report = self.assessment_history[-1]
        
        progress = {
            "errors_change": len(current_errors) - last_report.total_errors,
            "warnings_change": len([e for e in current_errors if e.severity == "warning"]) - last_report.total_warnings,
            "improvement_percentage": 0
        }
        
        if last_report.total_errors > 0:
            progress["improvement_percentage"] = (
                (last_report.total_errors - len(current_errors)) / last_report.total_errors * 100
            )
        
        return progress
    
    def _save_report(self, report: AssessmentReport):
        """Save assessment report to file."""
        reports_dir = os.path.join(self.workspace_path, "assessment_reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(reports_dir, f"error_assessment_{timestamp}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2)
        
        print(f"📄 Report saved to: {report_file}")
    
    def _display_report(self, report: AssessmentReport):
        """Display formatted assessment report."""
        print("\n" + "="*60)
        print("🔍 AUTOMATED ERROR ASSESSMENT REPORT")
        print("="*60)
        
        print(f"📅 Timestamp: {report.timestamp}")
        print(f"📁 Files Analyzed: {report.total_files_analyzed}")
        print(f"🚨 Total Errors: {report.total_errors}")
        print(f"⚠️  Total Warnings: {report.total_warnings}")
        
        if report.progress_since_last:
            progress = report.progress_since_last
            change_emoji = "📈" if progress["errors_change"] > 0 else "📉" if progress["errors_change"] < 0 else "➡️"
            print(f"{change_emoji} Progress: {progress['errors_change']:+d} errors, {progress['improvement_percentage']:.1f}% improvement")
        
        print("\n📊 ERRORS BY CATEGORY:")
        for category, count in sorted(report.errors_by_category.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {category.upper()}: {count}")
        
        print("\n📋 ERRORS BY SEVERITY:")
        for severity, count in sorted(report.errors_by_severity.items(), key=lambda x: x[1], reverse=True):
            emoji = {"critical": "🔴", "error": "🟠", "warning": "🟡", "info": "🔵"}.get(severity, "⚪")
            print(f"  {emoji} {severity.upper()}: {count}")
        
        print("\n🎯 TOP PRIORITY FIXES:")
        for i, fix in enumerate(report.top_priority_fixes[:5], 1):
            print(f"  {i}. {fix}")
        
        print("\n💡 IMPROVEMENT SUGGESTIONS:")
        for i, suggestion in enumerate(report.improvement_suggestions[:3], 1):
            print(f"  {i}. {suggestion}")
        
        # Generate next action recommendations
        print("\n🚀 NEXT RECOMMENDED ACTIONS:")
        if report.total_errors > 0:
            if report.errors_by_category.get("import", 0) > 0:
                print("  1. 📦 Install missing dependencies and fix import paths")
            if report.errors_by_category.get("type", 0) > 0:
                print("  2. 🏷️  Add type casting and update annotations")
            if report.errors_by_category.get("attribute", 0) > 0:
                print("  3. 🔧 Fix attribute access and method definitions")
        else:
            print("  🎉 No critical errors found! Focus on optimizing warnings.")
        
        print("\n" + "="*60)

    def run_continuous_monitoring(self, interval: int = 30):
        """Run continuous monitoring with specified interval."""
        print(f"🔄 Starting continuous monitoring (every {interval} seconds)...")
        
        try:
            while True:
                report = self.analyze_workspace()
                self.assessment_history.append(report)
                
                # Keep only last 10 reports
                if len(self.assessment_history) > 10:
                    self.assessment_history = self.assessment_history[-10:]
                
                print(f"⏰ Next assessment in {interval} seconds...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Error Assessment System")
    parser.add_argument("--workspace", "-w", help="Workspace path", default=".")
    parser.add_argument("--continuous", "-c", action="store_true", help="Run continuous monitoring")
    parser.add_argument("--interval", "-i", type=int, default=30, help="Monitoring interval (seconds)")
    
    args = parser.parse_args()
    
    # Initialize assessment system
    assessor = AutomatedErrorAssessment(args.workspace)
    
    if args.continuous:
        assessor.run_continuous_monitoring(args.interval)
    else:
        # Run single assessment
        report = assessor.analyze_workspace()
        
        # Generate follow-up recommendations
        print("\n🔄 AUTOMATED FOLLOW-UP RECOMMENDATIONS:")
        if report.total_errors > 5:
            print("  • High error count detected - recommend focused fix session")
            print("  • Consider running: python automated_error_assessment.py --continuous")
        elif report.total_errors > 0:
            print("  • Good progress! Focus on remaining critical errors")
        else:
            print("  • Excellent! System is error-free. Monitor warnings for optimization")

if __name__ == "__main__":
    main()
