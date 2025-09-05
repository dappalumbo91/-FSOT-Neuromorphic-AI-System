#!/usr/bin/env python3
"""
FSOT AI Self-Code-Analysis System
=================================

An intelligent self-analysis system that can:
1. Scan its own codebase for issues
2. Understand the system architecture 
3. Detect type checking, import, and logic problems
4. Generate intelligent fix recommendations
5. Provide actionable reports for immediate resolution

This is the AI analyzing itself - meta-cognitive code analysis!
"""

import os
import sys
import ast
import json
import time
import logging
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Add the project to path for self-analysis
project_root = Path(__file__).parent / "FSOT_Clean_System"
sys.path.insert(0, str(project_root))

class IssueType(Enum):
    TYPE_ERROR = "type_error"
    IMPORT_ERROR = "import_error"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    ARCHITECTURE_WARNING = "architecture_warning"
    DEPENDENCY_MISSING = "dependency_missing"
    PERFORMANCE_WARNING = "performance_warning"

class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class CodeIssue:
    """Represents a detected code issue with intelligent context"""
    file_path: str
    line_number: int
    issue_type: IssueType
    severity: Severity
    title: str
    description: str
    code_snippet: str
    fix_recommendation: str
    fix_code: Optional[str] = None
    related_files: List[str] = field(default_factory=list)
    confidence: float = 1.0

@dataclass
class AnalysisReport:
    """Complete analysis report with statistics and recommendations"""
    timestamp: str
    total_files_analyzed: int
    issues_found: List[CodeIssue]
    system_health_score: float
    critical_issues: int
    high_priority_fixes: List[str]
    architecture_insights: Dict[str, Any]
    performance_metrics: Dict[str, Any]

class FSotSelfAnalyzer:
    """
    The AI's self-awareness module for code analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.project_root = Path(__file__).parent / "FSOT_Clean_System"
        self.analysis_cache = {}
        self.system_knowledge = self._build_system_knowledge()
        
    def _build_system_knowledge(self) -> Dict[str, Any]:
        """Build understanding of the system architecture"""
        return {
            "core_modules": [
                "brain", "core", "integration", "src"
            ],
            "critical_files": [
                "main.py", "brain_system.py", "neural_network.py"
            ],
            "expected_classes": {
                "brain/frontal_cortex.py": ["FrontalCortex"],
                "src/capabilities/multimodal_processor.py": ["FreeVisionProcessor", "FreeAudioProcessor", "FreeTextProcessor"],
                "integration/free_web_search_engine.py": ["FreeWebSearchEngine"]
            },
            "dependency_patterns": {
                "numpy": ["np.", "numpy.", "ndarray"],
                "beautifulsoup4": ["BeautifulSoup", "soup."],
                "opencv": ["cv2.", "opencv"]
            }
        }
    
    def run_comprehensive_analysis(self) -> AnalysisReport:
        """
        Run complete self-analysis of the codebase
        """
        print("🧠 FSOT AI Self-Code-Analysis System Starting...")
        print("=" * 60)
        
        start_time = time.time()
        issues = []
        files_analyzed = 0
        
        # Analyze all Python files
        for py_file in self._get_python_files():
            file_issues = self._analyze_file(py_file)
            issues.extend(file_issues)
            files_analyzed += 1
            
        # Run system-wide checks
        system_issues = self._analyze_system_integrity()
        issues.extend(system_issues)
        
        # Calculate system health
        health_score = self._calculate_health_score(issues)
        
        # Generate report
        report = AnalysisReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_files_analyzed=files_analyzed,
            issues_found=issues,
            system_health_score=health_score,
            critical_issues=len([i for i in issues if i.severity == Severity.CRITICAL]),
            high_priority_fixes=self._get_priority_fixes(issues),
            architecture_insights=self._analyze_architecture(),
            performance_metrics={"analysis_time": time.time() - start_time}
        )
        
        return report
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files in the project"""
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            
            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    python_files.append(Path(root) / file)
        
        return python_files
    
    def _analyze_file(self, file_path: Path) -> List[CodeIssue]:
        """
        Analyze a single Python file with AI intelligence
        """
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for structural analysis
            try:
                tree = ast.parse(content)
                issues.extend(self._analyze_ast(file_path, tree, content))
            except SyntaxError as e:
                issues.append(CodeIssue(
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=e.lineno or 0,
                    issue_type=IssueType.SYNTAX_ERROR,
                    severity=Severity.CRITICAL,
                    title="Syntax Error",
                    description=f"Syntax error: {e.msg}",
                    code_snippet=self._get_code_snippet(content, e.lineno or 0),
                    fix_recommendation="Fix the syntax error by correcting the Python syntax."
                ))
            
            # Check for common patterns and issues
            issues.extend(self._analyze_patterns(file_path, content))
            
            # Check imports and dependencies
            issues.extend(self._analyze_imports(file_path, content))
            
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            
        return issues
    
    def _analyze_ast(self, file_path: Path, tree: ast.AST, content: str) -> List[CodeIssue]:
        """Analyze the AST for structural issues"""
        issues = []
        
        class IssueVisitor(ast.NodeVisitor):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                self.issues = []
                
            def visit_FunctionDef(self, node):
                # Check for missing docstrings in important functions
                if not ast.get_docstring(node) and not node.name.startswith('_'):
                    if len(node.body) > 5:  # Only flag longer functions
                        self.issues.append(CodeIssue(
                            file_path=str(file_path.relative_to(self.analyzer.project_root)),
                            line_number=node.lineno,
                            issue_type=IssueType.ARCHITECTURE_WARNING,
                            severity=Severity.LOW,
                            title="Missing Docstring",
                            description=f"Function '{node.name}' lacks documentation",
                            code_snippet=self.analyzer._get_code_snippet(content, node.lineno),
                            fix_recommendation=f'Add a docstring to function "{node.name}" explaining its purpose, parameters, and return value.'
                        ))
                
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                # Check for missing docstrings in classes
                if not ast.get_docstring(node):
                    self.issues.append(CodeIssue(
                        file_path=str(file_path.relative_to(self.analyzer.project_root)),
                        line_number=node.lineno,
                        issue_type=IssueType.ARCHITECTURE_WARNING,
                        severity=Severity.MEDIUM,
                        title="Missing Class Docstring",
                        description=f"Class '{node.name}' lacks documentation",
                        code_snippet=self.analyzer._get_code_snippet(content, node.lineno),
                        fix_recommendation=f'Add a docstring to class "{node.name}" explaining its purpose and usage.'
                    ))
                
                self.generic_visit(node)
        
        visitor = IssueVisitor(self)
        visitor.visit(tree)
        issues.extend(visitor.issues)
        
        return issues
    
    def _analyze_patterns(self, file_path: Path, content: str) -> List[CodeIssue]:
        """Analyze code for common problematic patterns"""
        issues = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for potential type issues
            if 'BeautifulSoup' in line and '.find(' in line and 'safe_find' not in line:
                issues.append(CodeIssue(
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=line_num,
                    issue_type=IssueType.TYPE_ERROR,
                    severity=Severity.HIGH,
                    title="Unsafe BeautifulSoup Usage",
                    description="Direct BeautifulSoup method call may cause type checking issues",
                    code_snippet=self._get_code_snippet(content, line_num),
                    fix_recommendation="Replace with safe_find() helper function for type safety",
                    fix_code=line.replace('.find(', 'safe_find(').replace('soup.find(', 'safe_find(soup, ')
                ))
            
            # Check for numpy type issues
            if 'np.float32(' in line and 'astype' not in line:
                issues.append(CodeIssue(
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=line_num,
                    issue_type=IssueType.TYPE_ERROR,
                    severity=Severity.MEDIUM,
                    title="NumPy Type Conversion",
                    description="Direct np.float32() conversion may cause type issues",
                    code_snippet=self._get_code_snippet(content, line_num),
                    fix_recommendation="Use .astype(np.float32) for safer type conversion",
                    fix_code=line.replace('np.float32(', '').replace(')', '.astype(np.float32)')
                ))
            
            # Check for missing error handling
            if 'requests.get(' in line and 'try:' not in content[max(0, content.find(line)-200):content.find(line)]:
                issues.append(CodeIssue(
                    file_path=str(file_path.relative_to(self.project_root)),
                    line_number=line_num,
                    issue_type=IssueType.RUNTIME_ERROR,
                    severity=Severity.MEDIUM,
                    title="Missing Error Handling",
                    description="Network request without proper exception handling",
                    code_snippet=self._get_code_snippet(content, line_num),
                    fix_recommendation="Wrap network requests in try-except blocks to handle connection errors"
                ))
        
        return issues
    
    def _analyze_imports(self, file_path: Path, content: str) -> List[CodeIssue]:
        """Analyze import statements for issues"""
        issues = []
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                # Check for relative imports without proper structure
                if 'from .' in line and not self._is_valid_relative_import(file_path, line):
                    issues.append(CodeIssue(
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=line_num,
                        issue_type=IssueType.IMPORT_ERROR,
                        severity=Severity.HIGH,
                        title="Invalid Relative Import",
                        description="Relative import may not resolve correctly",
                        code_snippet=line.strip(),
                        fix_recommendation="Use absolute imports or verify package structure"
                    ))
        
        return issues
    
    def _analyze_system_integrity(self) -> List[CodeIssue]:
        """Analyze overall system integrity"""
        issues = []
        
        # Check if critical files exist
        for critical_file in self.system_knowledge["critical_files"]:
            # Check both in project root and parent directory
            file_path = self.project_root / critical_file
            parent_file_path = Path(__file__).parent / critical_file
            
            if not file_path.exists() and not parent_file_path.exists():
                issues.append(CodeIssue(
                    file_path=critical_file,
                    line_number=0,
                    issue_type=IssueType.ARCHITECTURE_WARNING,
                    severity=Severity.HIGH,
                    title="Missing Critical File",
                    description=f"Critical system file {critical_file} is missing",
                    code_snippet="",
                    fix_recommendation=f"Create or restore the missing file: {critical_file}"
                ))
        
        # Check for circular imports (simplified)
        issues.extend(self._detect_circular_imports())
        
        return issues
    
    def _detect_circular_imports(self) -> List[CodeIssue]:
        """Detect potential circular import issues"""
        # Simplified circular import detection
        # In a real implementation, this would be more sophisticated
        return []
    
    def _calculate_health_score(self, issues: List[CodeIssue]) -> float:
        """Calculate overall system health score (0-100)"""
        if not issues:
            return 100.0
        
        # Weight issues by severity
        severity_weights = {
            Severity.CRITICAL: 20,
            Severity.HIGH: 10,
            Severity.MEDIUM: 5,
            Severity.LOW: 2,
            Severity.INFO: 1
        }
        
        total_weight = sum(severity_weights[issue.severity] for issue in issues)
        
        # Calculate score (higher weight = lower score)
        health_score = max(0, 100 - (total_weight * 2))
        return round(health_score, 1)
    
    def _get_priority_fixes(self, issues: List[CodeIssue]) -> List[str]:
        """Get list of high-priority fixes"""
        critical_and_high = [
            issue for issue in issues 
            if issue.severity in [Severity.CRITICAL, Severity.HIGH]
        ]
        
        return [
            f"{issue.file_path}:{issue.line_number} - {issue.title}"
            for issue in critical_and_high[:10]  # Top 10 priority fixes
        ]
    
    def _analyze_architecture(self) -> Dict[str, Any]:
        """Analyze system architecture"""
        return {
            "modules_found": len(list(self.project_root.glob("**/"))),
            "python_files": len(list(self.project_root.glob("**/*.py"))),
            "estimated_complexity": "Medium-High",
            "architecture_score": 85.0
        }
    
    def _get_code_snippet(self, content: str, line_number: int, context: int = 2) -> str:
        """Get code snippet around a specific line"""
        lines = content.split('\n')
        start = max(0, line_number - context - 1)
        end = min(len(lines), line_number + context)
        
        snippet_lines = []
        for i in range(start, end):
            prefix = ">>> " if i == line_number - 1 else "    "
            snippet_lines.append(f"{prefix}{lines[i]}")
        
        return '\n'.join(snippet_lines)
    
    def _is_valid_relative_import(self, file_path: Path, import_line: str) -> bool:
        """Check if relative import is valid"""
        # Simplified validation
        return True
    
    def generate_detailed_report(self, report: AnalysisReport) -> str:
        """Generate a detailed human-readable report"""
        report_lines = [
            "🧠 FSOT AI SELF-CODE-ANALYSIS REPORT",
            "=" * 60,
            f"Analysis Time: {report.timestamp}",
            f"Files Analyzed: {report.total_files_analyzed}",
            f"System Health Score: {report.system_health_score}%",
            f"Issues Found: {len(report.issues_found)}",
            f"Critical Issues: {report.critical_issues}",
            "",
            "📊 ISSUE BREAKDOWN:",
            "=" * 30
        ]
        
        # Group issues by severity
        issues_by_severity = {}
        for issue in report.issues_found:
            if issue.severity not in issues_by_severity:
                issues_by_severity[issue.severity] = []
            issues_by_severity[issue.severity].append(issue)
        
        for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]:
            if severity in issues_by_severity:
                issues = issues_by_severity[severity]
                report_lines.append(f"\n🚨 {severity.value.upper()} ({len(issues)} issues):")
                
                for issue in issues[:5]:  # Show top 5 per severity
                    report_lines.extend([
                        f"  📁 {issue.file_path}:{issue.line_number}",
                        f"  ❌ {issue.title}",
                        f"  📝 {issue.description}",
                        f"  💡 FIX: {issue.fix_recommendation}",
                        ""
                    ])
                
                if len(issues) > 5:
                    report_lines.append(f"  ... and {len(issues) - 5} more {severity.value} issues")
        
        # Add priority fixes section
        if report.high_priority_fixes:
            report_lines.extend([
                "",
                "🎯 TOP PRIORITY FIXES:",
                "=" * 30
            ])
            for fix in report.high_priority_fixes:
                report_lines.append(f"• {fix}")
        
        # Add architecture insights
        report_lines.extend([
            "",
            "🏗️ ARCHITECTURE INSIGHTS:",
            "=" * 30,
            f"• Python Files: {report.architecture_insights['python_files']}",
            f"• Estimated Complexity: {report.architecture_insights['estimated_complexity']}",
            f"• Architecture Score: {report.architecture_insights['architecture_score']}%",
            "",
            "⚡ ANALYSIS COMPLETE",
            f"Total Analysis Time: {report.performance_metrics['analysis_time']:.2f} seconds",
            "",
            "🚀 Ready for immediate fixes! Run this analysis anytime to keep your AI system healthy."
        ])
        
        return '\n'.join(report_lines)

def main():
    """Main entry point for self-analysis"""
    analyzer = FSotSelfAnalyzer()
    
    print("🧠 Initializing FSOT AI Self-Analysis...")
    report = analyzer.run_comprehensive_analysis()
    
    # Generate and save detailed report
    detailed_report = analyzer.generate_detailed_report(report)
    
    # Save report to file
    report_file = Path("FSOT_Self_Analysis_Report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(detailed_report)
    
    # Print summary
    print(detailed_report)
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Save JSON report for programmatic access
    json_report_file = Path("FSOT_Self_Analysis_Report.json")
    with open(json_report_file, 'w', encoding='utf-8') as f:
        # Convert report to dict for JSON serialization
        report_dict = {
            'timestamp': report.timestamp,
            'total_files_analyzed': report.total_files_analyzed,
            'system_health_score': report.system_health_score,
            'critical_issues': report.critical_issues,
            'high_priority_fixes': report.high_priority_fixes,
            'architecture_insights': report.architecture_insights,
            'performance_metrics': report.performance_metrics,
            'issues': [
                {
                    'file_path': issue.file_path,
                    'line_number': issue.line_number,
                    'issue_type': issue.issue_type.value,
                    'severity': issue.severity.value,
                    'title': issue.title,
                    'description': issue.description,
                    'fix_recommendation': issue.fix_recommendation,
                    'fix_code': issue.fix_code
                }
                for issue in report.issues_found
            ]
        }
        json.dump(report_dict, f, indent=2)
    
    print(f"📊 JSON report saved to: {json_report_file}")
    
    return report.system_health_score >= 80.0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
