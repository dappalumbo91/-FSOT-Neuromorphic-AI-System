#!/usr/bin/env python3
"""
FSOT AI Auto-Fix Engine
======================

Companion to the self-analyzer that can automatically apply intelligent fixes
to code issues detected by the AI self-analysis system.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any

class FSotAutoFixer:
    """
    Intelligent auto-fix engine that applies AI-recommended code fixes
    """
    
    def __init__(self):
        self.fixes_applied = 0
        self.fixes_failed = 0
        
    def apply_fixes_from_report(self, json_report_path: str) -> Dict[str, Any]:
        """Apply fixes from a JSON analysis report"""
        
        with open(json_report_path, 'r') as f:
            report = json.load(f)
        
        print("🔧 FSOT AI Auto-Fix Engine Starting...")
        print("=" * 50)
        
        results = {
            'fixes_attempted': 0,
            'fixes_successful': 0,
            'fixes_failed': 0,
            'files_modified': [],
            'errors': []
        }
        
        project_root = Path(__file__).parent / "FSOT_Clean_System"
        
        for issue in report['issues']:
            if issue.get('fix_code') and issue['severity'] in ['critical', 'high']:
                try:
                    file_path = project_root / issue['file_path']
                    
                    if self._apply_fix(file_path, issue):
                        results['fixes_successful'] += 1
                        if str(file_path) not in results['files_modified']:
                            results['files_modified'].append(str(file_path))
                        print(f"✅ Fixed: {issue['title']} in {issue['file_path']}")
                    else:
                        results['fixes_failed'] += 1
                        print(f"❌ Failed: {issue['title']} in {issue['file_path']}")
                        
                    results['fixes_attempted'] += 1
                    
                except Exception as e:
                    results['fixes_failed'] += 1
                    results['errors'].append(f"{issue['file_path']}: {str(e)}")
                    print(f"❌ Error fixing {issue['file_path']}: {e}")
        
        print(f"\n📊 Auto-Fix Results:")
        print(f"   Attempted: {results['fixes_attempted']}")
        print(f"   Successful: {results['fixes_successful']}")
        print(f"   Failed: {results['fixes_failed']}")
        print(f"   Files Modified: {len(results['files_modified'])}")
        
        return results
    
    def _apply_fix(self, file_path: Path, issue: Dict[str, Any]) -> bool:
        """Apply a specific fix to a file"""
        try:
            if not file_path.exists():
                return False
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            line_num = issue['line_number'] - 1  # Convert to 0-based
            
            if 0 <= line_num < len(lines):
                # Apply the fix
                lines[line_num] = issue['fix_code']
                
                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                
                return True
                
        except Exception as e:
            print(f"Error applying fix: {e}")
            
        return False

def main():
    """Main auto-fix entry point"""
    fixer = FSotAutoFixer()
    
    json_report = "FSOT_Self_Analysis_Report.json"
    if Path(json_report).exists():
        results = fixer.apply_fixes_from_report(json_report)
        print(f"\n🎉 Auto-fix complete! Check the modified files.")
    else:
        print(f"❌ Report file {json_report} not found. Run the analyzer first!")

if __name__ == "__main__":
    main()
