#!/usr/bin/env python3
"""
GitHub Workflow Validator
=========================
Validates YAML syntax and structure for GitHub Actions workflows.
"""

import yaml
import json
from pathlib import Path
from datetime import datetime

def validate_workflow_yaml(file_path):
    """Validate GitHub workflow YAML file."""
    print(f"🔍 Validating workflow file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            workflow = yaml.safe_load(content)
        
        print("✅ YAML syntax is valid")
        
        # Check required GitHub workflow structure
        required_fields = ['name', 'on', 'jobs']
        for field in required_fields:
            if field in workflow:
                print(f"✅ Required field '{field}' present")
            else:
                print(f"❌ Missing required field '{field}'")
                return False
        
        # Validate jobs structure
        jobs = workflow.get('jobs', {})
        if isinstance(jobs, dict) and len(jobs) > 0:
            print(f"✅ Found {len(jobs)} job(s): {list(jobs.keys())}")
            
            for job_name, job_config in jobs.items():
                if 'runs-on' in job_config:
                    print(f"✅ Job '{job_name}' has runs-on: {job_config['runs-on']}")
                else:
                    print(f"⚠️ Job '{job_name}' missing runs-on")
                
                if 'steps' in job_config and isinstance(job_config['steps'], list):
                    step_count = len(job_config['steps'])
                    print(f"✅ Job '{job_name}' has {step_count} steps")
                else:
                    print(f"⚠️ Job '{job_name}' has no steps or invalid steps format")
        else:
            print("❌ No valid jobs found")
            return False
        
        return True
        
    except yaml.YAMLError as e:
        print(f"❌ YAML syntax error: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def check_workflow_best_practices(file_path):
    """Check GitHub workflow best practices."""
    print(f"\n🔧 Checking best practices for: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        workflow = yaml.safe_load(content)
    
    recommendations = []
    
    # Check for version pinning
    jobs = workflow.get('jobs', {})
    for job_name, job_config in jobs.items():
        steps = job_config.get('steps', [])
        for step in steps:
            if 'uses' in step:
                action = step['uses']
                if '@v' in action and not action.endswith(('@v4', '@v3')):
                    recommendations.append(f"Consider updating action '{action}' to latest version")
                elif '@' not in action:
                    recommendations.append(f"Action '{action}' should specify version")
    
    # Check for caching
    has_cache = any('cache' in str(step).lower() for job in jobs.values() 
                   for step in job.get('steps', []))
    if not has_cache:
        recommendations.append("Consider adding caching for dependencies to speed up builds")
    
    # Check for matrix strategy
    has_matrix = any('matrix' in job.get('strategy', {}) for job in jobs.values())
    if has_matrix:
        print("✅ Uses matrix strategy for multi-environment testing")
    else:
        recommendations.append("Consider using matrix strategy for testing multiple environments")
    
    if recommendations:
        print("💡 Recommendations:")
        for rec in recommendations:
            print(f"   • {rec}")
    else:
        print("✅ No recommendations - following best practices!")
    
    return recommendations

def generate_workflow_report():
    """Generate comprehensive workflow validation report."""
    print("📊 GitHub Workflow Validation Report")
    print("=" * 40)
    
    workflow_dir = Path(".github/workflows")
    if not workflow_dir.exists():
        print("❌ No .github/workflows directory found")
        return
    
    workflow_files = list(workflow_dir.glob("*.yml")) + list(workflow_dir.glob("*.yaml"))
    
    if not workflow_files:
        print("❌ No workflow files found")
        return
    
    print(f"Found {len(workflow_files)} workflow file(s)")
    
    results = {}
    for workflow_file in workflow_files:
        print(f"\n" + "="*50)
        is_valid = validate_workflow_yaml(workflow_file)
        recommendations = check_workflow_best_practices(workflow_file) if is_valid else []
        
        results[str(workflow_file)] = {
            "valid": is_valid,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        }
    
    # Summary
    valid_count = sum(1 for r in results.values() if r["valid"])
    total_count = len(results)
    
    print(f"\n📈 VALIDATION SUMMARY:")
    print(f"   Valid workflows: {valid_count}/{total_count}")
    print(f"   Success rate: {(valid_count/total_count)*100:.1f}%")
    
    if valid_count == total_count:
        print("🎉 All workflows are valid!")
    else:
        print("⚠️ Some workflows need attention")
    
    # Save report
    report_file = f"workflow_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Report saved to: {report_file}")
    
    return results

def main():
    """Main validation execution."""
    print("🔍 GITHUB WORKFLOW VALIDATOR")
    print("=" * 35)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = generate_workflow_report()
    
    print(f"\n🎯 Workflow validation complete!")
    
    return results

if __name__ == "__main__":
    main()
