#!/usr/bin/env python3
"""
Modular_AI_Project Cleanup Script
Removes clutter while preserving valuable capabilities that have been extracted
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Set
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ModularAICleanup:
    """Handles cleanup of the Modular_AI_Project folder"""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.items_to_delete: Set[str] = set()
        self.items_preserved: Set[str] = set()
        self.deleted_count = 0
        self.preserved_count = 0
        
        if not self.project_path.exists():
            raise FileNotFoundError(f"Project path does not exist: {project_path}")
    
    def define_cleanup_rules(self):
        """Define what should be deleted vs preserved"""
        
        # Files/folders to DELETE (clutter)
        self.delete_patterns = {
            # Development artifacts
            '.coverage', '.env', '.env.example', '.env.production.example',
            '.git', '.github', '.gradio', '.model_cache', '.pylintrc', 
            '.pytest_cache', '.security_key', '.venv', '.vscode',
            '__pycache__', 'htmlcov', 'cache', 'logs', 'temp',
            
            # Backup files
            'backups', 'backup_archive', 'backup_local_files', 'code_backups',
            
            # Screenshots and images
            'capture.png', 'demo_performance_chart.png', 'demo_region_20250903_175715.png',
            'demo_screenshot.png', 'demo_screenshot_20250903_175715.png',
            'cv_analysis_20250903_175715.png', 'neuromorphic_brain_architecture.png',
            'neuromorphic_brain_performance.png',
            
            # Test and debug files
            'debug_browser.py', 'debug_kb.py', 'debug_report_1756914457.json',
            'debug_report_20250903_222123.json', 'debug_report_20250903_222331.json',
            'debug_report_20250903_222458.json', 'debug_report_20250903_223005.json',
            'pylance_compatibility_summary.md', 'pylance_fix_report.py',
            'fix_file.py', 'fix_file_script.py', 'fix_json.py', 'final_fix.py',
            'comprehensive_fix.py', 'escape_fix.py', 'line_by_line_fix.py',
            'phase1_fixes.py', 'targeted_debug_fix.py',
            
            # Installation and setup
            'admin_installer.py', 'install_firefox.py', 'install_programs.py',
            'modular_program_installer.py', 'download_essentials.py',
            'setup_and_launch.bat', 'setup_openai_api.py', 'setup_portable_programs.py',
            'setup_production.py', 'setup_windows_integration.py', 'create_shortcut.py',
            
            # Deployment files
            'docker-compose.prod.yml', 'docker-compose.yml', 'Dockerfile',
            'Dockerfile.multi-stage', 'nginx.conf', 'deploy_production.sh',
            'production_server.py', 'PRODUCTION_README.md',
            
            # Training and model files
            'training', 'models', 'model_cache', 'optuna_results.json',
            'optuna_study.db', 'pflt_model_state_20250902_174734.pth',
            'model_state.pth', 'fsot_model.pth',
            
            # Duplicate/legacy files
            'hello_world.py', 'main.py', 'consolidated_entry_point.py',
            'consolidated_entry_point.py.backup', 'consolidated_entry_point_final.py',
            
            # Web integration (too complex)
            'grok_integration.py', 'human_level_grok_interactor.py',
            'human_level_web_interactor.py', 'inspect_grok_page.py',
            'simple_grok_inspector.py', 'chrome_profiles',
            
            # Specialized/complex systems
            'cortana_activated_system.py', 'complete_cortana_system.py',
            'ultimate_cortana_conversation.py', 'CHROME_INTEGRATION_README.md',
            'tesseract_download.html', 'universal_translation',
            'virtual_desktop_workspace',
            
            # Generated extensions and builds
            'generated_extensions', 'build', 'dist',
            
            # Evaluation and test results
            'evaluation_summary_20250902_165453.txt', 'evaluation_summary_20250902_195453.txt',
            'evaluation_summary_20250902_200559.txt', 'evaluation_summary_20250902_200740.txt',
            'evaluation_summary_20250903_125611.txt', 'evaluation_summary_20250903_125900.txt',
            'correlation_test_results.json', 'stability_test_report.json',
            'final_test_report.json', 'final_simple_test_report.json',
            'advanced_test_report.json', 'integration_test_results.json',
            
            # Programs and downloads
            'Programs', 'downloads', 'uploads', 'static', 'templates',
        }
        
        # Pattern-based deletions (files starting with these patterns)
        self.delete_file_patterns = {
            'screen_capture_', 'page_screenshot_', 'debug_report_',
            'evaluation_summary_', 'phase1_', 'final_', 'complete_',
            'comprehensive_', 'advanced_', 'enhanced_', 'ultimate_',
            'consolidated_', 'unified_evaluation_report_',
        }
        
        # File extensions to delete
        self.delete_extensions = {'.png', '.backup', '.log'}
        
        # Files to PRESERVE (valuable capabilities already extracted)
        self.preserve_patterns = {
            # Core documentation
            'README.md', 'LICENSE', 'requirements.txt',
            
            # Already extracted capabilities
            'adaptive_memory_manager.py',  # Extracted to utils/memory_manager.py
            'enhanced_multimodal_system.py',  # Will be adapted
            'brain_modules',  # Template for additional modules
            'api_config.json',  # API configuration template
            'windows_speech.py',  # Speech capabilities
            'web_search_engine.py',  # Web search capabilities
            'safety',  # Security framework
        }
    
    def should_delete(self, item_path: Path) -> bool:
        """Determine if an item should be deleted"""
        item_name = item_path.name
        
        # Check preserve patterns first
        for pattern in self.preserve_patterns:
            if pattern in str(item_path):
                return False
        
        # Check exact matches
        if item_name in self.delete_patterns:
            return True
        
        # Check file patterns
        for pattern in self.delete_file_patterns:
            if item_name.startswith(pattern):
                return True
        
        # Check extensions
        if item_path.suffix in self.delete_extensions:
            return True
        
        return False
    
    def scan_directory(self) -> tuple[List[Path], List[Path]]:
        """Scan directory and categorize items"""
        items_to_delete = []
        items_to_preserve = []
        
        for item in self.project_path.iterdir():
            if self.should_delete(item):
                items_to_delete.append(item)
            else:
                items_to_preserve.append(item)
        
        return items_to_delete, items_to_preserve
    
    def preview_cleanup(self):
        """Preview what will be deleted/preserved"""
        items_to_delete, items_to_preserve = self.scan_directory()
        
        print(f"\n📋 CLEANUP PREVIEW for {self.project_path}")
        print("=" * 60)
        
        print(f"\n🗑️  ITEMS TO DELETE ({len(items_to_delete)}):")
        for item in sorted(items_to_delete):
            item_type = "📁" if item.is_dir() else "📄"
            print(f"   {item_type} {item.name}")
        
        print(f"\n💾 ITEMS TO PRESERVE ({len(items_to_preserve)}):")
        for item in sorted(items_to_preserve):
            item_type = "📁" if item.is_dir() else "📄"
            print(f"   {item_type} {item.name}")
        
        # Calculate space savings estimate
        total_size = 0
        delete_size = 0
        
        for item in self.project_path.iterdir():
            try:
                if item.is_file():
                    size = item.stat().st_size
                    total_size += size
                    if item in items_to_delete:
                        delete_size += size
                elif item.is_dir():
                    # Estimate directory size
                    try:
                        dir_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                        total_size += dir_size
                        if item in items_to_delete:
                            delete_size += dir_size
                    except:
                        pass
            except:
                pass
        
        print(f"\n📊 ESTIMATED CLEANUP IMPACT:")
        print(f"   Total items: {len(items_to_delete) + len(items_to_preserve)}")
        print(f"   Items to delete: {len(items_to_delete)} ({len(items_to_delete)/(len(items_to_delete) + len(items_to_preserve))*100:.1f}%)")
        print(f"   Items to preserve: {len(items_to_preserve)}")
        
        if total_size > 0:
            print(f"   Estimated space savings: {delete_size/(1024*1024):.1f}MB ({delete_size/total_size*100:.1f}%)")
    
    def execute_cleanup(self, dry_run: bool = True):
        """Execute the cleanup"""
        items_to_delete, items_to_preserve = self.scan_directory()
        
        if dry_run:
            print(f"\n🔍 DRY RUN MODE - No files will be deleted")
            self.preview_cleanup()
            return
        
        print(f"\n🗑️  EXECUTING CLEANUP...")
        
        success_count = 0
        error_count = 0
        
        for item in items_to_delete:
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                    print(f"   📁 Deleted directory: {item.name}")
                else:
                    item.unlink()
                    print(f"   📄 Deleted file: {item.name}")
                success_count += 1
                
            except Exception as e:
                print(f"   ❌ Error deleting {item.name}: {e}")
                error_count += 1
        
        print(f"\n✅ CLEANUP COMPLETE!")
        print(f"   Successfully deleted: {success_count} items")
        print(f"   Errors: {error_count} items")
        print(f"   Preserved: {len(items_to_preserve)} items")
        
        if error_count == 0:
            print(f"\n🎉 Cleanup successful! The Modular_AI_Project folder is now clean.")
        else:
            print(f"\n⚠️  Cleanup completed with {error_count} errors. Please check manually.")

def main():
    """Main entry point"""
    print("🧹 MODULAR_AI_PROJECT CLEANUP TOOL")
    print("=" * 50)
    
    # Get project path
    project_path = Path(__file__).parent.parent / "Modular_AI_Project"
    
    if not project_path.exists():
        print(f"❌ Project path not found: {project_path}")
        return 1
    
    try:
        cleanup = ModularAICleanup(str(project_path))
        cleanup.define_cleanup_rules()
        
        # Show preview first
        cleanup.preview_cleanup()
        
        # Ask for confirmation
        print(f"\n⚠️  WARNING: This will permanently delete files!")
        print(f"   Project path: {project_path}")
        
        response = input(f"\nProceed with cleanup? (yes/no): ").lower().strip()
        
        if response in ['yes', 'y']:
            cleanup.execute_cleanup(dry_run=False)
        else:
            print("❌ Cleanup cancelled.")
            return 0
        
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
