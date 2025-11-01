"""
Fix Import Paths After Reorganization
=====================================

After moving files from root to src/core/ and src/enhancements/,
all import statements need to be updated.

This script:
1. Scans all Python files
2. Finds old imports (from root)
3. Updates to new imports (from src.*)
4. Creates backup before modifying
"""

import os
import re
from pathlib import Path
import shutil

# Mapping of old imports to new imports
IMPORT_MAPPING = {
    # Core modules
    'polynomial_structure_analyzer': 'src.core.polynomial_structure_analyzer',
    'safe_dispatcher': 'src.core.safe_dispatcher',
    'integrated_pipeline': 'src.core.integrated_pipeline',
    'pauli_utils': 'src.core.pauli_utils',
    
    # Enhancement modules
    'cdcl_probe': 'src.enhancements.cdcl_probe',
    'sequential_testing': 'src.enhancements.sequential_testing',
    'ml_classifier': 'src.enhancements.ml_classifier',
}


def fix_imports_in_file(filepath):
    """Fix imports in a single file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    changes = []
    
    # Pattern 1: from module import ...
    for old_module, new_module in IMPORT_MAPPING.items():
        # Match: from polynomial_structure_analyzer import ...
        pattern1 = rf'from {old_module} import'
        replacement1 = f'from {new_module} import'
        if pattern1 in content:
            content = content.replace(pattern1, replacement1)
            changes.append(f"  from {old_module} → from {new_module}")
        
        # Match: import polynomial_structure_analyzer
        pattern2 = rf'\nimport {old_module}\b'
        replacement2 = f'\nimport {new_module}'
        if re.search(pattern2, content):
            content = re.sub(pattern2, replacement2, content)
            changes.append(f"  import {old_module} → import {new_module}")
        
        # Match: import polynomial_structure_analyzer as alias
        pattern3 = rf'\nimport {old_module} as '
        replacement3 = f'\nimport {new_module} as '
        if re.search(pattern3, content):
            content = re.sub(pattern3, replacement3, content)
            changes.append(f"  import {old_module} as ... → import {new_module} as ...")
    
    if content != original_content:
        # Create backup
        backup_path = str(filepath) + '.backup'
        shutil.copy2(filepath, backup_path)
        
        # Write fixed content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True, changes
    
    return False, []


def fix_all_imports(base_path='.'):
    """Fix imports in all Python files"""
    base = Path(base_path)
    
    # Directories to scan
    scan_dirs = [
        base / 'tests',
        base / 'benchmarks',
        base / 'experiments',
        base / 'src' / 'core',
        base / 'src' / 'enhancements',
    ]
    
    fixed_files = []
    
    print("="*70)
    print("FIXING IMPORTS AFTER REORGANIZATION")
    print("="*70)
    print()
    
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        
        print(f"Scanning: {scan_dir}/")
        
        for py_file in scan_dir.glob('*.py'):
            # Skip __pycache__ and backup files
            if '__pycache__' in str(py_file) or py_file.suffix == '.backup':
                continue
            
            modified, changes = fix_imports_in_file(py_file)
            
            if modified:
                fixed_files.append((py_file, changes))
                print(f"  ✓ Fixed: {py_file.name}")
                for change in changes:
                    print(f"    {change}")
    
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Files fixed: {len(fixed_files)}")
    
    if fixed_files:
        print()
        print("Backups created with .backup extension")
        print("If everything works, you can delete the .backup files")
        print()
        print("To restore from backup:")
        print("  1. Delete the modified file")
        print("  2. Rename .backup file (remove .backup extension)")
    
    return fixed_files


def add_sys_path_to_files(base_path='.'):
    """
    Add sys.path modification to files that need it.
    This allows imports to work from any directory.
    """
    base = Path(base_path)
    
    sys_path_code = """
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
"""
    
    # Files that need sys.path modification (tests, benchmarks, experiments)
    scan_dirs = [
        (base / 'tests', 2),  # 2 levels up (tests/ -> Quantum_sat/)
        (base / 'benchmarks', 2),  # 2 levels up
        (base / 'experiments', 2),  # 2 levels up
    ]
    
    print()
    print("="*70)
    print("ADDING SYS.PATH MODIFICATIONS (for import resolution)")
    print("="*70)
    print()
    
    for scan_dir, levels_up in scan_dirs:
        if not scan_dir.exists():
            continue
        
        print(f"Processing: {scan_dir}/")
        
        for py_file in scan_dir.glob('*.py'):
            if '__pycache__' in str(py_file) or py_file.suffix == '.backup':
                continue
            
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if already has sys.path modification
            if 'sys.path.insert' in content or 'sys.path.append' in content:
                print(f"  ⊘ Skip (already has sys.path): {py_file.name}")
                continue
            
            # Add sys.path code at the beginning (after docstring if present)
            lines = content.split('\n')
            
            # Find insertion point (after module docstring)
            insert_idx = 0
            in_docstring = False
            for i, line in enumerate(lines):
                if i == 0 and (line.startswith('"""') or line.startswith("'''")):
                    in_docstring = True
                elif in_docstring and ('"""' in line or "'''" in line):
                    insert_idx = i + 1
                    break
                elif not line.strip().startswith('#') and line.strip():
                    insert_idx = i
                    break
            
            # Adjust path code for correct number of levels
            path_code = sys_path_code.replace('parent.parent', 'parent' * levels_up)
            
            # Insert sys.path code
            new_content = '\n'.join(lines[:insert_idx]) + '\n' + path_code + '\n' + '\n'.join(lines[insert_idx:])
            
            # Create backup
            backup_path = py_file.with_suffix('.py.backup2')
            shutil.copy2(py_file, backup_path)
            
            # Write modified content
            with open(py_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"  ✓ Added sys.path: {py_file.name}")


if __name__ == "__main__":
    print()
    
    # Step 1: Fix import statements
    fixed = fix_all_imports()
    
    # Step 2: Add sys.path modifications
    add_sys_path_to_files()
    
    print()
    print("="*70)
    print("ALL DONE!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Test the imports: python tests/test_adaptive_monte_carlo.py")
    print("  2. Test the benchmarks: python benchmarks/demo_production_system.py")
    print("  3. If everything works, delete .backup files")
    print()
