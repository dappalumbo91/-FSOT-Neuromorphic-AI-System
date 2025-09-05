#!/usr/bin/env python3
"""
Enhanced Automated Error and Warning Assessment System
====================================================

Combines automated error detection with VS Code integration to provide
comprehensive analysis matching what users see in their development environment.

Features:
- Integration with enhanced error detector
- VS Code Pylance error matching
- Virtual environment error filtering
- Real-time assessment with intelligent recommendations
- Progress tracking and automated fixes
"""

import os
import sys
import json
import time
import ast
import re
import subprocess
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

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
    source: str = "pylance"  # "pylance", "ast", "custom"

@dataclass
class AssessmentReport:
    """Complete assessment report."""
    timestamp: str
    total_files_analyzed: int
    project_files_analyzed: int
    total_errors: int
    total_warnings: int
    project_errors: int
    project_warnings: int
    errors_by_category: Dict[str, int]
    errors_by_severity: Dict[str, int]
    vscode_matching_errors: int
    top_priority_fixes: List[str]
    improvement_suggestions: List[str]
    progress_since_last: Optional[Dict[str, Any]]
    detailed_errors: List[Dict[str, Any]]

class EnhancedAutomatedAssessment:
    """Enhanced automated error assessment system with VS Code integration."""
    
    def __init__(self, workspace_path: Optional[str] = None):
        """Initialize the enhanced assessment system."""
        self.workspace_path = workspace_path or os.getcwd()
        self.assessment_history = []
        self.error_patterns = self._load_enhanced_error_patterns()
        self.fix_recommendations = self._load_enhanced_fix_recommendations()
        
        # Virtual environment detection
        self.venv_patterns = [
            r'\.venv',
            r'venv',
            r'site-packages',
            r'Scripts\\python',
            r'bin/python',
            r'__pycache__'
        ]
    
    def _is_project_file(self, file_path: str) -> bool:
        """Check if file is a project file (not virtual environment)."""
        for pattern in self.venv_patterns:
            if re.search(pattern, file_path, re.IGNORECASE):
                return False
        return True
    
    def _load_enhanced_error_patterns(self) -> Dict[str, Any]:
        """Load enhanced error pattern recognition rules."""
        return {
            # VS Code specific syntax errors
            "invalid_character": {
                "pattern": r"Invalid character.*in token|unexpected character",
                "category": "syntax",
                "severity": "critical",
                "priority": 10,
                "fix_template": "Remove invalid escape characters or fix string literals"
            },
            "unclosed_parentheses": {
                "pattern": r'".*" was not closed|Expected expression|Statements must be separated',
                "category": "syntax", 
                "severity": "critical",
                "priority": 10,
                "fix_template": "Close parentheses, brackets, or fix statement separation"
            },
            
            # Import errors
            "unknown_import_symbol": {
                "pattern": r'"(.+)" is unknown import symbol',
                "category": "import",
                "severity": "error",
                "priority": 8,
                "fix_template": "Check if module exists, add to requirements.txt, or create fallback"
            },
            "import_resolution": {
                "pattern": r'Import "(.+)" could not be resolved',
                "category": "import", 
                "severity": "error",
                "priority": 8,
                "fix_template": "Install missing package or check import path"
            },
            "circular_import": {
                "pattern": r'Circular import detected|cyclic import',
                "category": "import",
                "severity": "error", 
                "priority": 7,
                "fix_template": "Reorganize imports or use lazy loading"
            },
            
            # Type errors
            "type_assignment": {
                "pattern": r'Type "(.+)" is not assignable to declared type "(.+)"',
                "category": "type",
                "severity": "warning",
                "priority": 5,
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
                "priority": 4,
                "fix_template": "Rename class or use unique namespaces"
            },
            "missing_attribute": {
                "pattern": r'Cannot access attribute "(.+)" for class "(.+)"',
                "category": "attribute",
                "severity": "error",
                "priority": 7,
                "fix_template": "Add missing method/attribute or check object type"
            },
            "function_override": {
                "pattern": r'Function "(.+)" overrides class "(.+)"',
                "category": "declaration",
                "severity": "error",
                "priority": 6,
                "fix_template": "Rename function or class to avoid conflicts"
            },
            
            # Variable/usage errors
            "undefined_variable": {
                "pattern": r'"(.+)" is not defined|Undefined variable',
                "category": "variable",
                "severity": "error", 
                "priority": 8,
                "fix_template": "Define variable before use or check scope"
            },
            "unused_import": {
                "pattern": r'"(.+)" is not accessed|unused import',
                "category": "cleanup",
                "severity": "warning",
                "priority": 2,
                "fix_template": "Remove unused import statements"
            }
        }
    
    def _load_enhanced_fix_recommendations(self) -> Dict[str, List[str]]:
        """Load enhanced intelligent fix recommendation rules."""
        return {
            "syntax": [
                "Fix invalid escape characters in strings (use raw strings r'' or double backslashes)",
                "Ensure all parentheses, brackets, and quotes are properly closed",
                "Separate statements with newlines or semicolons",
                "Check for missing commas in lists/dictionaries",
                "Validate indentation consistency"
            ],
            "import": [
                "Install missing packages: pip install <package>",
                "Verify import paths and module names", 
                "Add missing modules to requirements.txt",
                "Create fallback implementations for optional dependencies",
                "Use try-except blocks for conditional imports",
                "Reorganize imports to avoid circular dependencies"
            ],
            "type": [
                "Add explicit type casting: float(), int(), str()",
                "Update type annotations to match actual usage",
                "Use Union types for multiple possible types",
                "Add type: ignore comments for known safe issues",
                "Consider using Any type for complex dynamic cases"
            ],
            "attribute": [
                "Define attributes in __init__ method",
                "Use setattr() for dynamic attribute assignment",
                "Add property decorators for computed attributes", 
                "Check if object has attribute: hasattr(obj, 'attr')",
                "Use type assertions: assert isinstance(obj, ExpectedType)"
            ],
            "declaration": [
                "Use unique class/function names to avoid conflicts",
                "Implement proper namespace separation",
                "Use aliases for imports: from module import Class as LocalClass",
                "Consider moving conflicting classes to separate modules",
                "Use conditional definitions with try-except"
            ],
            "variable": [
                "Define variables before use",
                "Check variable scope and initialization",
                "Use global/nonlocal keywords when needed", 
                "Initialize variables with default values",
                "Consider using dataclasses for structured data"
            ],
            "cleanup": [
                "Remove unused import statements",
                "Clean up commented-out code",
                "Remove unused variables and functions",
                "Organize imports alphabetically",
                "Use tools like isort and black for formatting"
            ]
        }
    
    def analyze_workspace(self) -> AssessmentReport:
        """Perform enhanced workspace error analysis."""
        print(f"🔍 Starting enhanced automated error assessment...")
        print(f"📁 Workspace: {self.workspace_path}")
        
        # Find all Python files
        all_python_files = self._find_python_files()
        project_files = [f for f in all_python_files if self._is_project_file(f)]
        
        print(f"📄 Found {len(all_python_files)} total Python files")
        print(f"🎯 Analyzing {len(project_files)} project files (excluding virtual environment)")
        
        # Run enhanced error detection
        all_errors = self._run_enhanced_detection(all_python_files)
        project_errors = [e for e in all_errors if self._is_project_file(e.file_path)]
        
        print(f"📊 Found {len(all_errors)} total errors, {len(project_errors)} in project files")
        
        # Generate comprehensive report
        report = self._generate_enhanced_report(all_python_files, project_files, all_errors, project_errors)
        
        # Save and display report
        self._save_report(report)
        self._display_enhanced_report(report)
        
        return report
    
    def _find_python_files(self) -> List[str]:
        """Find all Python files in workspace."""
        python_files = []
        for root, dirs, files in os.walk(self.workspace_path):
            # Skip hidden directories but include .venv for full analysis
            dirs[:] = [d for d in dirs if not d.startswith('.git')]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return python_files
    
    def _run_enhanced_detection(self, files: List[str]) -> List[ErrorInfo]:
        """Run enhanced error detection combining multiple methods."""
        all_errors = []
        
        print("🔍 Running enhanced error detection...")
        
        for file_path in files:
            try:
                # AST-based syntax checking
                errors = self._check_syntax_errors(file_path)
                all_errors.extend(errors)
                
                # Content-based error detection
                errors = self._check_content_errors(file_path)
                all_errors.extend(errors)
                
                # Pattern-based detection
                errors = self._check_pattern_errors(file_path)
                all_errors.extend(errors)
                
            except Exception as e:
                if self._is_project_file(file_path):
                    print(f"⚠️ Error analyzing {file_path}: {e}")
        
        return all_errors
    
    def _check_syntax_errors(self, file_path: str) -> List[ErrorInfo]:
        """Check for syntax errors using AST."""
        errors = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Try to parse with AST
            try:
                ast.parse(content)
            except SyntaxError as e:
                error = ErrorInfo(
                    file_path=file_path,
                    line_number=e.lineno or 1,
                    error_type="SyntaxError",
                    error_message=str(e),
                    code_snippet=e.text or "",
                    severity="critical",
                    category="syntax",
                    fix_priority=10,
                    source="ast"
                )
                errors.append(error)
            
        except Exception:
            pass  # Skip files that can't be read
        
        return errors
    
    def _check_content_errors(self, file_path: str) -> List[ErrorInfo]:
        """Check for content-based errors (invalid characters, etc.)."""
        errors = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                # Check for invalid escape characters
                if re.search(r'\\[^rntbfav\\\'"0-7xuUN]', line):
                    error = ErrorInfo(
                        file_path=file_path,
                        line_number=line_num,
                        error_type="InvalidCharacter",
                        error_message=f"Invalid escape character found in line {line_num}",
                        code_snippet=line.strip(),
                        severity="critical",
                        category="syntax", 
                        fix_priority=9,
                        source="custom"
                    )
                    errors.append(error)
                
                # Check for unclosed quotes/parentheses (simple heuristic)
                quote_count = line.count('"') + line.count("'")
                paren_open = line.count('(') + line.count('[') + line.count('{')
                paren_close = line.count(')') + line.count(']') + line.count('}')
                
                if quote_count % 2 != 0 and not line.strip().startswith('#'):
                    error = ErrorInfo(
                        file_path=file_path,
                        line_number=line_num,
                        error_type="UnclosedQuote",
                        error_message=f"Potentially unclosed quote in line {line_num}",
                        code_snippet=line.strip(),
                        severity="warning",
                        category="syntax",
                        fix_priority=6,
                        source="custom"
                    )
                    errors.append(error)
        
        except Exception:
            pass
        
        return errors
    
    def _check_pattern_errors(self, file_path: str) -> List[ErrorInfo]:
        """Check for pattern-based errors."""
        errors = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Look for common problematic patterns
            patterns = [
                (r'class\s+(\w+).*:\s*.*class\s+\1', "duplicate_class", "Duplicate class definition"),
                (r'def\s+(\w+).*:\s*.*def\s+\1', "duplicate_function", "Duplicate function definition"),
                (r'from\s+\.\.\.', "invalid_import", "Invalid relative import"),
                (r'except\s*:', "bare_except", "Bare except clause")
            ]
            
            for pattern, error_type, message in patterns:
                matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    error = ErrorInfo(
                        file_path=file_path,
                        line_number=line_num,
                        error_type=error_type,
                        error_message=message,
                        code_snippet=match.group(0)[:100],
                        severity="warning",
                        category="code_quality",
                        fix_priority=3,
                        source="custom"
                    )
                    errors.append(error)
        
        except Exception:
            pass
        
        return errors
    
    def _categorize_error(self, error_message: str) -> Tuple[str, str, int]:
        """Enhanced error categorization."""
        for pattern_name, pattern_info in self.error_patterns.items():
            if re.search(pattern_info["pattern"], error_message, re.IGNORECASE):
                return (
                    pattern_info["category"],
                    pattern_info["severity"], 
                    pattern_info["priority"]
                )
        
        # Enhanced default categorization
        msg_lower = error_message.lower()
        if any(word in msg_lower for word in ["syntax", "invalid", "unexpected"]):
            return ("syntax", "critical", 9)
        elif any(word in msg_lower for word in ["import", "module", "package"]):
            return ("import", "error", 7)
        elif any(word in msg_lower for word in ["type", "assign"]):
            return ("type", "warning", 4)
        elif any(word in msg_lower for word in ["attribute", "method"]):
            return ("attribute", "error", 6)
        else:
            return ("other", "warning", 3)
    
    def _generate_enhanced_report(self, all_files: List[str], project_files: List[str], 
                                all_errors: List[ErrorInfo], project_errors: List[ErrorInfo]) -> AssessmentReport:
        """Generate enhanced assessment report."""
        
        # Count errors by category and severity
        errors_by_category = {}
        errors_by_severity = {}
        
        for error in project_errors:  # Focus on project errors for categorization
            errors_by_category[error.category] = errors_by_category.get(error.category, 0) + 1
            errors_by_severity[error.severity] = errors_by_severity.get(error.severity, 0) + 1
        
        # Generate top priority fixes
        top_priority_fixes = self._generate_priority_fixes(project_errors)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(errors_by_category)
        
        # Calculate progress
        progress = self._calculate_progress(project_errors) if self.assessment_history else None
        
        # Detailed errors for report
        detailed_errors = [
            {
                "file": error.file_path,
                "line": error.line_number,
                "type": error.error_type,
                "message": error.error_message,
                "category": error.category,
                "severity": error.severity,
                "priority": error.fix_priority,
                "source": error.source
            }
            for error in project_errors[:20]  # Top 20 project errors
        ]
        
        report = AssessmentReport(
            timestamp=datetime.now().isoformat(),
            total_files_analyzed=len(all_files),
            project_files_analyzed=len(project_files),
            total_errors=len([e for e in all_errors if e.severity in ["critical", "error"]]),
            total_warnings=len([e for e in all_errors if e.severity == "warning"]),
            project_errors=len([e for e in project_errors if e.severity in ["critical", "error"]]),
            project_warnings=len([e for e in project_errors if e.severity == "warning"]),
            errors_by_category=errors_by_category,
            errors_by_severity=errors_by_severity,
            vscode_matching_errors=len([e for e in project_errors if e.severity == "critical"]),
            top_priority_fixes=top_priority_fixes,
            improvement_suggestions=improvement_suggestions,
            progress_since_last=progress,
            detailed_errors=detailed_errors
        )
        
        return report
    
    def _generate_priority_fixes(self, errors: List[ErrorInfo]) -> List[str]:
        """Generate enhanced prioritized fix recommendations."""
        # Sort errors by priority
        sorted_errors = sorted(errors, key=lambda x: x.fix_priority, reverse=True)
        
        fixes = []
        seen_categories = set()
        
        for error in sorted_errors[:15]:  # Top 15 priorities
            if error.category not in seen_categories:
                category_fixes = self.fix_recommendations.get(error.category, [])
                if category_fixes:
                    priority_emoji = "🔴" if error.fix_priority >= 8 else "🟠" if error.fix_priority >= 6 else "🟡"
                    fixes.append(f"{priority_emoji} {error.category.upper()}: {category_fixes[0]}")
                    seen_categories.add(error.category)
        
        return fixes
    
    def _generate_improvement_suggestions(self, errors_by_category: Dict[str, int]) -> List[str]:
        """Generate enhanced improvement suggestions."""
        suggestions = []
        
        # Prioritize suggestions based on error frequency and impact
        sorted_categories = sorted(errors_by_category.items(), key=lambda x: x[1], reverse=True)
        
        improvement_map = {
            "syntax": "🔧 Focus on syntax errors first - they prevent code execution",
            "import": "📦 Resolve import issues to enable proper module loading", 
            "type": "🏷️ Add type annotations and casting for better code reliability",
            "attribute": "🔗 Fix attribute access issues to prevent runtime errors",
            "declaration": "📝 Resolve naming conflicts for cleaner code structure"
        }
        
        for category, count in sorted_categories[:5]:
            if count > 0 and category in improvement_map:
                suggestions.append(f"{improvement_map[category]} ({count} issues)")
        
        # Add general suggestions based on error patterns
        if sum(errors_by_category.values()) > 10:
            suggestions.append("🎯 Consider implementing automated pre-commit hooks")
        if errors_by_category.get("syntax", 0) > 3:
            suggestions.append("⚡ Use an IDE with real-time syntax checking")
        
        return suggestions
    
    def _calculate_progress(self, current_errors: List[ErrorInfo]) -> Optional[Dict[str, Any]]:
        """Calculate progress since last assessment."""
        if not self.assessment_history:
            return None
        
        last_report = self.assessment_history[-1]
        current_error_count = len([e for e in current_errors if e.severity in ["critical", "error"]])
        current_warning_count = len([e for e in current_errors if e.severity == "warning"])
        
        progress = {
            "errors_change": current_error_count - last_report.project_errors,
            "warnings_change": current_warning_count - last_report.project_warnings,
            "improvement_percentage": 0,
            "trend": "stable"
        }
        
        if last_report.project_errors > 0:
            progress["improvement_percentage"] = (
                (last_report.project_errors - current_error_count) / last_report.project_errors * 100
            )
        
        # Determine trend
        if progress["errors_change"] < -2:
            progress["trend"] = "improving"
        elif progress["errors_change"] > 2:
            progress["trend"] = "declining"
        
        return progress
    
    def _save_report(self, report: AssessmentReport):
        """Save enhanced assessment report."""
        reports_dir = os.path.join(self.workspace_path, "assessment_reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(reports_dir, f"enhanced_assessment_{timestamp}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(report), f, indent=2)
        
        print(f"📄 Enhanced report saved to: {report_file}")
    
    def _display_enhanced_report(self, report: AssessmentReport):
        """Display enhanced formatted assessment report."""
        print("\n" + "="*70)
        print("🔍 ENHANCED AUTOMATED ERROR ASSESSMENT REPORT")
        print("="*70)
        
        print(f"📅 Timestamp: {report.timestamp}")
        print(f"📁 Total Files: {report.total_files_analyzed} | Project Files: {report.project_files_analyzed}")
        print(f"🚨 Project Errors: {report.project_errors} | Warnings: {report.project_warnings}")
        print(f"📊 Total (Including Dependencies): {report.total_errors} errors, {report.total_warnings} warnings")
        print(f"🎯 VS Code Matching Errors: {report.vscode_matching_errors}")
        
        if report.progress_since_last:
            progress = report.progress_since_last
            trend_emoji = {"improving": "📈", "declining": "📉", "stable": "➡️"}[progress["trend"]]
            print(f"{trend_emoji} Progress: {progress['errors_change']:+d} errors, {progress['improvement_percentage']:.1f}% improvement")
        
        print("\n📊 PROJECT ERRORS BY CATEGORY:")
        for category, count in sorted(report.errors_by_category.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {category.upper()}: {count}")
        
        print("\n📋 PROJECT ERRORS BY SEVERITY:")
        for severity, count in sorted(report.errors_by_severity.items(), key=lambda x: x[1], reverse=True):
            emoji = {"critical": "🔴", "error": "🟠", "warning": "🟡", "info": "🔵"}.get(severity, "⚪")
            print(f"  {emoji} {severity.upper()}: {count}")
        
        print("\n🎯 TOP PRIORITY FIXES:")
        for i, fix in enumerate(report.top_priority_fixes[:7], 1):
            print(f"  {i}. {fix}")
        
        print("\n💡 IMPROVEMENT SUGGESTIONS:")
        for i, suggestion in enumerate(report.improvement_suggestions[:5], 1):
            print(f"  {i}. {suggestion}")
        
        if report.detailed_errors:
            print("\n📝 DETAILED ERROR SAMPLE (Top Project Issues):")
            for i, error in enumerate(report.detailed_errors[:5], 1):
                file_short = os.path.basename(error["file"])
                print(f"  {i}. {file_short}:{error['line']} - {error['type']} ({error['severity']})")
                print(f"     {error['message'][:80]}...")
        
        # Generate action plan
        print("\n🚀 RECOMMENDED ACTION PLAN:")
        if report.project_errors > 0:
            if report.errors_by_category.get("syntax", 0) > 0:
                print("  1. 🔴 CRITICAL: Fix syntax errors immediately")
            if report.errors_by_category.get("import", 0) > 0:
                print("  2. 🟠 HIGH: Resolve import and dependency issues")
            if report.errors_by_category.get("attribute", 0) > 0:
                print("  3. 🟡 MEDIUM: Fix attribute access and method issues")
            if report.errors_by_category.get("type", 0) > 0:
                print("  4. 🔵 LOW: Address type annotation warnings")
        else:
            print("  🎉 Excellent! No project errors found.")
            if report.project_warnings > 0:
                print("  🔧 Focus on optimizing remaining warnings for code quality")
            else:
                print("  ✨ Perfect code quality! Consider adding automated testing")
        
        print("\n" + "="*70)

def main():
    """Main entry point for enhanced assessment."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Automated Error Assessment System")
    parser.add_argument("--workspace", "-w", help="Workspace path", default=".")
    parser.add_argument("--continuous", "-c", action="store_true", help="Run continuous monitoring")
    parser.add_argument("--interval", "-i", type=int, default=60, help="Monitoring interval (seconds)")
    parser.add_argument("--project-only", "-p", action="store_true", help="Show only project errors (exclude venv)")
    
    args = parser.parse_args()
    
    # Initialize enhanced assessment system
    assessor = EnhancedAutomatedAssessment(args.workspace)
    
    if args.continuous:
        print(f"🔄 Starting enhanced continuous monitoring (every {args.interval} seconds)...")
        try:
            while True:
                report = assessor.analyze_workspace()
                assessor.assessment_history.append(report)
                
                # Keep only last 20 reports for trend analysis
                if len(assessor.assessment_history) > 20:
                    assessor.assessment_history = assessor.assessment_history[-20:]
                
                print(f"⏰ Next assessment in {args.interval} seconds...")
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Enhanced monitoring stopped by user")
    else:
        # Run single enhanced assessment
        report = assessor.analyze_workspace()
        
        # Generate intelligent follow-up recommendations
        print("\n🤖 INTELLIGENT FOLLOW-UP RECOMMENDATIONS:")
        if report.project_errors > 10:
            print("  📈 High error count - recommend immediate fix session")
            print("  🔄 Consider: python enhanced_automated_assessment.py --continuous")
        elif report.project_errors > 0:
            print("  ✅ Manageable error count - focus on critical issues first")
            print("  🎯 Target syntax and import errors for maximum impact")
        else:
            print("  🌟 Outstanding! Zero project errors detected")
            if report.project_warnings > 0:
                print("  📝 Optional: Address warnings for perfect code quality")

if __name__ == "__main__":
    main()
