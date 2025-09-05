#!/usr/bin/env python3
"""
Integrated Enhanced Error Assessment System
===========================================
Combines standard error detection with enhanced character-level analysis.
Filters out virtual environment errors to focus on project issues.
"""

import os
import ast
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class ErrorInfo:
    """Enhanced error information structure."""
    file_path: str
    line_number: int
    column: Optional[int]
    error_type: str
    message: str
    severity: str
    category: str
    source: str = "enhanced_detector"

class IntegratedErrorAssessment:
    """Integrated error assessment combining multiple detection methods."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.exclude_patterns = [
            '.venv',
            '__pycache__',
            '.git',
            'node_modules',
            '.pytest_cache'
        ]
        
    def is_project_file(self, file_path: str) -> bool:
        """Check if file is part of the project (not dependencies)."""
        path = Path(file_path)
        return not any(pattern in str(path) for pattern in self.exclude_patterns)
    
    def enhanced_file_scan(self, filepath: str) -> List[ErrorInfo]:
        """Enhanced scan of a single file for all error types."""
        errors = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Standard syntax check
            try:
                ast.parse(content)
            except SyntaxError as e:
                errors.append(ErrorInfo(
                    file_path=filepath,
                    line_number=e.lineno or 1,
                    column=e.offset,
                    error_type="SyntaxError",
                    message=str(e.msg),
                    severity="critical",
                    category="SYNTAX"
                ))
            
            # Enhanced character-level checks
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                # Invalid escape characters (like VS Code reports)
                if '\\u5c' in repr(line) or '\"\"\"\\' in line:
                    errors.append(ErrorInfo(
                        file_path=filepath,
                        line_number=i,
                        column=1,
                        error_type="InvalidCharacter",
                        message='Invalid character sequence in token',
                        severity="critical",
                        category="SYNTAX"
                    ))
                
                # Unclosed parentheses
                if 'print(' in line:
                    open_parens = line.count('(')
                    close_parens = line.count(')')
                    if open_parens > close_parens:
                        errors.append(ErrorInfo(
                            file_path=filepath,
                            line_number=i,
                            column=line.find('print(') + 6,
                            error_type="UnclosedParen",
                            message='"(" was not closed',
                            severity="critical",
                            category="SYNTAX"
                        ))
                
                # Expected expression (line continuation issues)
                if line.rstrip().endswith('\\') and not line.rstrip().endswith('\\\\'):
                    errors.append(ErrorInfo(
                        file_path=filepath,
                        line_number=i,
                        column=len(line.rstrip()),
                        error_type="ExpectedExpression",
                        message="Expected expression",
                        severity="critical",
                        category="SYNTAX"
                    ))
                
                # Malformed docstrings
                if line.strip().startswith('"""') and line.strip().endswith('\\'):
                    errors.append(ErrorInfo(
                        file_path=filepath,
                        line_number=i,
                        column=len(line.rstrip()) - 1,
                        error_type="MalformedDocstring",
                        message="Invalid character sequence in docstring",
                        severity="critical",
                        category="SYNTAX"
                    ))
        
        except Exception as e:
            errors.append(ErrorInfo(
                file_path=filepath,
                line_number=1,
                column=1,
                error_type="FileReadError",
                message=f"Error reading file: {str(e)}",
                severity="error",
                category="IO"
            ))
        
        return errors
    
    def scan_workspace(self) -> Dict[str, Any]:
        """Comprehensive workspace scan with filtering."""
        all_errors = []
        files_scanned = 0
        project_files_with_errors = []
        
        # Scan all Python files
        for root, dirs, files in os.walk('.'):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.exclude_patterns)]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    files_scanned += 1
                    
                    file_errors = self.enhanced_file_scan(filepath)
                    
                    # Only include project files in main results
                    if self.is_project_file(filepath) and file_errors:
                        project_files_with_errors.append({
                            'filepath': filepath,
                            'errors': [asdict(error) for error in file_errors],
                            'error_count': len(file_errors)
                        })
                    
                    all_errors.extend(file_errors)
        
        # Filter project errors
        project_errors = [e for e in all_errors if self.is_project_file(e.file_path)]
        
        # Categorize errors
        error_categories = {}
        severity_counts = {}
        
        for error in project_errors:
            error_categories[error.category] = error_categories.get(error.category, 0) + 1
            severity_counts[error.severity] = severity_counts.get(error.severity, 0) + 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'scan_summary': {
                'total_files_scanned': files_scanned,
                'project_files_with_errors': len(project_files_with_errors),
                'total_project_errors': len(project_errors),
                'total_all_errors': len(all_errors),
                'venv_errors_filtered': len(all_errors) - len(project_errors)
            },
            'project_errors': {
                'total': len(project_errors),
                'by_category': error_categories,
                'by_severity': severity_counts,
                'files': project_files_with_errors
            },
            'error_types_found': list(set(e.error_type for e in project_errors))
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive error report."""
        results = self.scan_workspace()
        
        report = []
        report.append("🔍 INTEGRATED ENHANCED ERROR ASSESSMENT")
        report.append("=" * 50)
        report.append(f"📅 Timestamp: {results['timestamp']}")
        report.append(f"📄 Files Scanned: {results['scan_summary']['total_files_scanned']}")
        report.append(f"🚨 Project Errors: {results['project_errors']['total']}")
        report.append(f"🗂️ Virtual Env Errors Filtered: {results['scan_summary']['venv_errors_filtered']}")
        report.append("")
        
        if results['project_errors']['total'] > 0:
            report.append("📊 ERROR BREAKDOWN:")
            for category, count in results['project_errors']['by_category'].items():
                report.append(f"  • {category}: {count}")
            report.append("")
            
            report.append("📋 SEVERITY BREAKDOWN:")
            for severity, count in results['project_errors']['by_severity'].items():
                report.append(f"  🔴 {severity.upper()}: {count}")
            report.append("")
            
            report.append("📁 PROJECT FILES WITH ERRORS:")
            for file_info in results['project_errors']['files']:
                report.append(f"\n📁 {file_info['filepath']} ({file_info['error_count']} errors):")
                for error in file_info['errors']:
                    report.append(f"   Line {error['line_number']}: {error['message']} ({error['error_type']})")
        else:
            report.append("🎉 No project errors found!")
        
        report.append(f"\n📊 ERROR TYPES: {', '.join(results['error_types_found']) if results['error_types_found'] else 'None'}")
        
        # Save detailed JSON report
        with open('integrated_error_report.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        report.append(f"\n💾 Detailed report saved: integrated_error_report.json")
        
        # VS Code integration recommendations
        if results['project_errors']['total'] > 0:
            report.append("\n🔧 NEXT STEPS:")
            report.append("1. Fix critical syntax errors first")
            report.append("2. Address expected expression errors (line continuation issues)")
            report.append("3. Fix unclosed parentheses")
            report.append("4. Run assessment again after fixes")
        
        return '\n'.join(report)

def main():
    """Main execution."""
    assessor = IntegratedErrorAssessment()
    report = assessor.generate_report()
    print(report)
    
    # Return error count for exit code
    results = assessor.scan_workspace()
    return results['project_errors']['total']

if __name__ == "__main__":
    error_count = main()
    exit(1 if error_count > 0 else 0)
