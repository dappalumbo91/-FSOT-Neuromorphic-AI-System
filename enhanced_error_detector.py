#!/usr/bin/env python3
"""
Enhanced Error Detector for FSOT Neuromorphic AI System
========================================================
Detects all types of errors including those that standard tools miss.
"""

import ast
import os
import json
from typing import List, Dict, Any
from datetime import datetime

class EnhancedErrorDetector:
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def scan_file(self, filepath: str) -> Dict[str, Any]:
        """Scan a single Python file for all types of errors."""
        file_errors = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Check for syntax errors
            try:
                ast.parse(content)
            except SyntaxError as e:
                file_errors.append({
                    'type': 'SyntaxError',
                    'message': str(e.msg),
                    'line': e.lineno,
                    'column': e.offset,
                    'severity': 'error'
                })
            
            # Check for invalid escape characters
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                # Look for invalid backslash sequences (specific error from VS Code)
                if '\\u5c' in repr(line):  # Check raw representation
                    file_errors.append({
                        'type': 'InvalidCharacter',
                        'message': 'Invalid character sequence in token',
                        'line': i,
                        'column': 1,
                        'severity': 'error'
                    })
                
                # Look for malformed docstrings
                if line.strip().startswith('"""') and line.strip().endswith('\\'):
                    file_errors.append({
                        'type': 'MalformedDocstring',
                        'message': 'Invalid character in docstring',
                        'line': i,
                        'column': len(line.rstrip()) - 1,
                        'severity': 'error'
                    })
                
                # Look for unclosed parentheses
                if 'print(' in line:
                    open_parens = line.count('(')
                    close_parens = line.count(')')
                    if open_parens > close_parens:
                        file_errors.append({
                            'type': 'UnclosedParen',
                            'message': '"(" was not closed',
                            'line': i,
                            'column': line.find('print(') + 6,
                            'severity': 'error'
                        })
                
                # Look for expected expression errors
                if line.strip().endswith('\\'):
                    file_errors.append({
                        'type': 'ExpectedExpression',
                        'message': 'Expected expression',
                        'line': i,
                        'column': len(line.rstrip()),
                        'severity': 'error'
                    })
        
        except Exception as e:
            file_errors.append({
                'type': 'FileError',
                'message': f'Error reading file: {str(e)}',
                'line': 1,
                'column': 1,
                'severity': 'error'
            })
        
        return {
            'filepath': filepath,
            'errors': file_errors,
            'error_count': len(file_errors)
        }
    
    def scan_workspace(self, workspace_path: str = '.') -> Dict[str, Any]:
        """Scan entire workspace for errors."""
        total_errors = 0
        total_warnings = 0
        file_results = []
        
        for root, dirs, files in os.walk(workspace_path):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    result = self.scan_file(filepath)
                    file_results.append(result)
                    total_errors += result['error_count']
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_files': len(file_results),
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'files': file_results,
            'summary': {
                'files_with_errors': len([f for f in file_results if f['error_count'] > 0]),
                'error_types': list(set([e['type'] for f in file_results for e in f['errors']]))
            }
        }

def main():
    """Run enhanced error detection."""
    print("🔍 ENHANCED ERROR DETECTION REPORT")
    print("=" * 50)
    
    detector = EnhancedErrorDetector()
    results = detector.scan_workspace('.')
    
    print(f"📅 Timestamp: {results['timestamp']}")
    print(f"📄 Files Scanned: {results['total_files']}")
    print(f"🚨 Total Errors: {results['total_errors']}")
    print(f"⚠️  Total Warnings: {results['total_warnings']}")
    print()
    
    if results['total_errors'] > 0:
        print("📋 DETAILED ERRORS:")
        for file_result in results['files']:
            if file_result['error_count'] > 0:
                print(f"\n📁 {file_result['filepath']}:")
                for error in file_result['errors']:
                    print(f"   Line {error['line']}: {error['message']} ({error['type']})")
    
    print(f"\n📊 ERROR TYPES FOUND: {', '.join(results['summary']['error_types']) if results['summary']['error_types'] else 'None'}")
    
    # Save detailed report
    with open('enhanced_error_report.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"\n💾 Detailed report saved: enhanced_error_report.json")
    
    return results['total_errors']

if __name__ == "__main__":
    error_count = main()
    exit(1 if error_count > 0 else 0)
