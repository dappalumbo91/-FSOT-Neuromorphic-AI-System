#!/usr/bin/env python3
"""
FSOT Visual AI File Management System
====================================
Professional file organization and categorization system for visual AI outputs.
"""

import os
import shutil
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import hashlib

class FSotVisualFileManager:
    """Professional file management system for FSOT visual AI outputs."""
    
    def __init__(self, base_directory: Optional[str] = None):
        self.base_dir = Path(base_directory) if base_directory else Path.cwd()
        self.organized_dir = self.base_dir / "FSOT_Visual_Archive"
        
        # Define categorical structure
        self.categories = {
            "consciousness": {
                "path": "01_Consciousness_Simulations",
                "description": "Monte Carlo consciousness evolution and awareness visualizations",
                "keywords": ["consciousness", "monte_carlo", "awareness", "evolution"]
            },
            "neural": {
                "path": "02_Neural_Networks",
                "description": "Neural network architectures and processing visualizations",
                "keywords": ["neural", "network", "processing", "architecture", "layers"]
            },
            "fractal": {
                "path": "03_Fractal_Patterns",
                "description": "Fractal analysis and consciousness pattern recognition",
                "keywords": ["fractal", "pattern", "dimension", "complexity"]
            },
            "artistic": {
                "path": "04_AI_Artistic_Creation",
                "description": "AI-generated artwork and creative visualizations",
                "keywords": ["artistic", "creation", "art", "creative", "dreams"]
            },
            "search": {
                "path": "05_Search_Analysis",
                "description": "Google search results and image analysis",
                "keywords": ["google", "search", "analysis", "results"]
            },
            "reports": {
                "path": "06_System_Reports",
                "description": "System reports, logs, and analysis documents",
                "keywords": ["report", "log", "analysis", "status", "debug"]
            },
            "experiments": {
                "path": "07_Experiments",
                "description": "Experimental visualizations and test results",
                "keywords": ["experiment", "test", "trial", "demo"]
            },
            "documentation": {
                "path": "08_Documentation",
                "description": "System documentation and guides",
                "keywords": ["doc", "guide", "readme", "instruction"]
            }
        }
        
        self.file_index = {}
        self._initialize_structure()
    
    def _initialize_structure(self):
        """Initialize the organized directory structure."""
        print("🗂️ Initializing FSOT Visual File Management System...")
        
        # Create main archive directory
        self.organized_dir.mkdir(exist_ok=True)
        
        # Create categorical subdirectories
        for category, info in self.categories.items():
            category_path = self.organized_dir / info["path"]
            category_path.mkdir(exist_ok=True)
            
            # Create subdirectories for different time periods
            for subdir in ["Current", "Archive_2024", "Archive_2025"]:
                (category_path / subdir).mkdir(exist_ok=True)
            
            # Create metadata file for category
            metadata_file = category_path / "category_info.json"
            if not metadata_file.exists():
                with open(metadata_file, 'w') as f:
                    json.dump({
                        "category": category,
                        "description": info["description"],
                        "keywords": info["keywords"],
                        "created": datetime.now().isoformat(),
                        "file_count": 0
                    }, f, indent=2)
        
        print(f"✅ Archive structure created at: {self.organized_dir}")
    
    def analyze_and_categorize_files(self):
        """Analyze all files in the current directory and categorize them."""
        print("\n🔍 Analyzing files for categorization...")
        
        # Get all files in current directory
        all_files = [f for f in self.base_dir.iterdir() if f.is_file()]
        
        categorized_files = {category: [] for category in self.categories.keys()}
        uncategorized_files = []
        
        for file_path in all_files:
            category = self._determine_category(file_path)
            if category:
                categorized_files[category].append(str(file_path))
            else:
                uncategorized_files.append(str(file_path))
        
        # Print analysis results
        print(f"\n📊 File Analysis Results:")
        for category, files in categorized_files.items():
            if files:
                print(f"   📁 {self.categories[category]['path']}: {len(files)} files")
        
        if uncategorized_files:
            print(f"   ❓ Uncategorized: {len(uncategorized_files)} files")
        
        return categorized_files, uncategorized_files
    
    def _determine_category(self, file_path: Path) -> Optional[str]:
        """Determine the appropriate category for a file."""
        filename = file_path.name.lower()
        
        # Check against category keywords
        for category, info in self.categories.items():
            for keyword in info["keywords"]:
                if keyword in filename:
                    return category
        
        # Special rules for specific file types
        if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            # Image files - try to categorize by name patterns
            if any(term in filename for term in ['consciousness', 'monte_carlo']):
                return 'consciousness'
            elif any(term in filename for term in ['neural', 'network']):
                return 'neural'
            elif any(term in filename for term in ['fractal', 'pattern']):
                return 'fractal'
            elif any(term in filename for term in ['artistic', 'art', 'creation']):
                return 'artistic'
            elif any(term in filename for term in ['search', 'google']):
                return 'search'
        
        elif file_path.suffix.lower() in ['.json', '.txt', '.md']:
            # Report/documentation files
            if any(term in filename for term in ['report', 'log', 'status', 'debug']):
                return 'reports'
            elif any(term in filename for term in ['readme', 'doc', 'guide']):
                return 'documentation'
        
        elif file_path.suffix.lower() == '.py':
            # Python files - likely experiments or core system
            if any(term in filename for term in ['test', 'demo', 'experiment']):
                return 'experiments'
            elif any(term in filename for term in ['visual', 'ai', 'demo']):
                return 'experiments'
        
        return None
    
    def organize_files(self, move_files: bool = True) -> Dict[str, Any]:
        """Organize files into categorical structure."""
        print(f"\n🗂️ {'Moving' if move_files else 'Copying'} files to organized structure...")
        
        categorized_files, uncategorized_files = self.analyze_and_categorize_files()
        organization_report = {
            "timestamp": datetime.now().isoformat(),
            "operation": "move" if move_files else "copy",
            "results": {},
            "errors": []
        }
        
        for category, files in categorized_files.items():
            if not files:
                continue
            
            category_info = self.categories[category]
            target_dir = self.organized_dir / category_info["path"] / "Current"
            
            category_results = {
                "files_processed": 0,
                "files_moved": [],
                "errors": []
            }
            
            for file_path_str in files:
                try:
                    source_path = Path(file_path_str)
                    if not source_path.exists():
                        continue
                    
                    # Generate unique filename with timestamp if needed
                    target_filename = self._generate_unique_filename(target_dir, source_path.name)
                    target_path = target_dir / target_filename
                    
                    # Perform file operation
                    if move_files:
                        shutil.move(str(source_path), str(target_path))
                        print(f"   📦 Moved: {source_path.name} → {category_info['path']}/Current/")
                    else:
                        shutil.copy2(str(source_path), str(target_path))
                        print(f"   📋 Copied: {source_path.name} → {category_info['path']}/Current/")
                    
                    # Update file index
                    self._update_file_index(category, source_path, target_path)
                    
                    category_results["files_processed"] += 1
                    category_results["files_moved"].append({
                        "original": str(source_path),
                        "new_location": str(target_path),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing {source_path}: {e}"
                    print(f"   ❌ {error_msg}")
                    category_results["errors"].append(error_msg)
                    organization_report["errors"].append(error_msg)
            
            organization_report["results"][category] = category_results
        
        # Handle uncategorized files
        if uncategorized_files:
            uncategorized_dir = self.organized_dir / "00_Uncategorized"
            uncategorized_dir.mkdir(exist_ok=True)
            
            print(f"\n❓ Processing {len(uncategorized_files)} uncategorized files...")
            uncategorized_results = {"files_processed": 0, "files_moved": [], "errors": []}
            
            for file_path_str in uncategorized_files:
                try:
                    source_path = Path(file_path_str)
                    if not source_path.exists():
                        continue
                    
                    target_filename = self._generate_unique_filename(uncategorized_dir, source_path.name)
                    target_path = uncategorized_dir / target_filename
                    
                    if move_files:
                        shutil.move(str(source_path), str(target_path))
                    else:
                        shutil.copy2(str(source_path), str(target_path))
                    
                    uncategorized_results["files_processed"] += 1
                    uncategorized_results["files_moved"].append({
                        "original": str(source_path),
                        "new_location": str(target_path)
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing uncategorized {source_path}: {e}"
                    uncategorized_results["errors"].append(error_msg)
                    organization_report["errors"].append(error_msg)
            
            organization_report["results"]["uncategorized"] = uncategorized_results
        
        # Save organization report
        report_path = self.organized_dir / "organization_report.json"
        with open(report_path, 'w') as f:
            json.dump(organization_report, f, indent=2)
        
        # Update category metadata
        self._update_category_metadata()
        
        return organization_report
    
    def _generate_unique_filename(self, target_dir: Path, original_name: str) -> str:
        """Generate a unique filename if conflicts exist."""
        base_path = target_dir / original_name
        if not base_path.exists():
            return original_name
        
        # Add timestamp to make unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_parts = original_name.rsplit('.', 1)
        if len(name_parts) == 2:
            return f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
        else:
            return f"{original_name}_{timestamp}"
    
    def _update_file_index(self, category: str, source_path: Path, target_path: Path):
        """Update the file index for quick retrieval."""
        file_hash = self._calculate_file_hash(target_path)
        
        file_entry = {
            "category": category,
            "original_name": source_path.name,
            "current_location": str(target_path),
            "file_hash": file_hash,
            "file_size": target_path.stat().st_size,
            "created_date": datetime.fromtimestamp(target_path.stat().st_ctime).isoformat(),
            "indexed_date": datetime.now().isoformat(),
            "file_type": target_path.suffix.lower(),
            "description": self._generate_file_description(source_path.name, category)
        }
        
        self.file_index[target_path.name] = file_entry
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for integrity checking."""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return "unknown"
    
    def _generate_file_description(self, filename: str, category: str) -> str:
        """Generate a descriptive text for the file."""
        category_info = self.categories.get(category, {})
        base_description = category_info.get("description", "FSOT AI system file")
        
        # Add specific details based on filename
        if "monte_carlo" in filename.lower():
            return f"{base_description} - Monte Carlo consciousness simulation results"
        elif "fractal" in filename.lower():
            return f"{base_description} - Fractal pattern analysis visualization"
        elif "neural" in filename.lower():
            return f"{base_description} - Neural network processing visualization"
        elif "artistic" in filename.lower():
            return f"{base_description} - AI artistic creation output"
        elif "google" in filename.lower():
            return f"{base_description} - Google search analysis results"
        elif "report" in filename.lower():
            return f"{base_description} - System analysis report"
        else:
            return base_description
    
    def _update_category_metadata(self):
        """Update metadata files for each category."""
        for category, info in self.categories.items():
            category_path = self.organized_dir / info["path"]
            metadata_file = category_path / "category_info.json"
            
            # Count files in category
            current_dir = category_path / "Current"
            file_count = len([f for f in current_dir.iterdir() if f.is_file()])
            
            # Update metadata
            metadata = {
                "category": category,
                "description": info["description"],
                "keywords": info["keywords"],
                "created": datetime.now().isoformat(),
                "file_count": file_count,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def create_retrieval_index(self) -> str:
        """Create a comprehensive retrieval index."""
        print("\n📝 Creating file retrieval index...")
        
        index_file = self.organized_dir / "file_retrieval_index.json"
        
        # Save file index
        with open(index_file, 'w') as f:
            json.dump(self.file_index, f, indent=2)
        
        # Create human-readable index
        readable_index_file = self.organized_dir / "FILE_INDEX.md"
        
        with open(readable_index_file, 'w') as f:
            f.write("# FSOT Visual AI File Index\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for category, info in self.categories.items():
                category_files = [entry for entry in self.file_index.values() 
                                if entry['category'] == category]
                
                if category_files:
                    f.write(f"## {info['path']}\n")
                    f.write(f"{info['description']}\n\n")
                    
                    for file_entry in category_files:
                        f.write(f"### {file_entry['original_name']}\n")
                        f.write(f"- **Location**: `{file_entry['current_location']}`\n")
                        f.write(f"- **Description**: {file_entry['description']}\n")
                        f.write(f"- **Size**: {file_entry['file_size']:,} bytes\n")
                        f.write(f"- **Created**: {file_entry['created_date']}\n")
                        f.write(f"- **Hash**: `{file_entry['file_hash'][:16]}...`\n\n")
        
        print(f"✅ Retrieval index created: {readable_index_file}")
        return str(readable_index_file)
    
    def generate_organization_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive organization summary."""
        print("\n📊 Generating organization summary...")
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "archive_location": str(self.organized_dir),
            "categories": {},
            "total_files": len(self.file_index),
            "total_size": 0
        }
        
        for category, info in self.categories.items():
            category_path = self.organized_dir / info["path"] / "Current"
            if category_path.exists():
                files = [f for f in category_path.iterdir() if f.is_file()]
                total_size = sum(f.stat().st_size for f in files)
                
                summary["categories"][category] = {
                    "path": str(category_path),
                    "description": info["description"],
                    "file_count": len(files),
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2)
                }
                
                summary["total_size"] += total_size
        
        summary["total_size_mb"] = round(summary["total_size"] / (1024 * 1024), 2)
        
        # Save summary
        summary_file = self.organized_dir / "organization_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def main():
    """Main execution function."""
    print("🗂️ FSOT VISUAL AI FILE MANAGEMENT SYSTEM")
    print("=" * 50)
    
    # Initialize file manager
    manager = FSotVisualFileManager()
    
    # Analyze current files
    categorized, uncategorized = manager.analyze_and_categorize_files()
    
    # Ask user for confirmation
    total_files = sum(len(files) for files in categorized.values()) + len(uncategorized)
    print(f"\n📋 Ready to organize {total_files} files")
    print("   This will move files to the organized structure.")
    
    response = input("\n🤔 Proceed with file organization? (y/n): ").lower().strip()
    
    if response == 'y':
        # Organize files
        report = manager.organize_files(move_files=True)
        
        # Create retrieval index
        index_file = manager.create_retrieval_index()
        
        # Generate summary
        summary = manager.generate_organization_summary()
        
        print(f"\n🎉 FILE ORGANIZATION COMPLETE!")
        print(f"📁 Archive location: {manager.organized_dir}")
        print(f"📝 Index file: {index_file}")
        print(f"📊 Total files organized: {summary['total_files']}")
        print(f"💾 Total size: {summary['total_size_mb']} MB")
        
        print(f"\n✨ Your FSOT Visual AI files are now professionally organized!")
        
    else:
        print("\n❌ File organization cancelled.")

if __name__ == "__main__":
    main()
