#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to check all imported packages in the codebase and verify they're in requirements.
"""

import importlib.util
import os
import re
import sys


def scan_file_for_imports(file_path):
    """Scan a Python file for import statements and return a list of package names."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Find import statements
    import_patterns = [
        r'^import\s+([a-zA-Z0-9_\.]+)',  # import x
        r'^from\s+([a-zA-Z0-9_\.]+)\s+import',  # from x import y
    ]

    imports = []

    for pattern in import_patterns:
        matches = re.findall(pattern, content, re.MULTILINE)
        imports.extend(matches)

    # Get the base package names (first component of the dotted path)
    base_packages = set()
    for imp in imports:
        base_pkg = imp.split('.')[0]
        if base_pkg and base_pkg not in ['__future__']:
            base_packages.add(base_pkg)

    return base_packages


def check_package_installable(package_name):
    """Check if a package is installable via pip."""
    # Exclude standard library modules
    if package_name in sys.builtin_module_names:
        return False

    # Check if it's a standard library module
    if importlib.util.find_spec(package_name) is not None:
        if not hasattr(importlib.util.find_spec(package_name), 'submodule_search_locations'):
            return False
        if importlib.util.find_spec(package_name).submodule_search_locations is None:
            return False
        for path in importlib.util.find_spec(package_name).submodule_search_locations:
            if 'site-packages' not in path and 'dist-packages' not in path:
                return False

    # Some common packages that are installable
    common_installable = {
        'torch', 'numpy', 'transformers', 'PyQt6', 'bitsandbytes', 'onnx',
        'onnxruntime', 'psutil', 'flask', 'requests', 'tqdm', 'matplotlib',
        'scipy', 'pandas', 'sklearn', 'skimage', 'PIL', 'cv2', 'tensorflow',
        'keras', 'h5py', 'json5', 'yaml', 'tomli', 'toml', 'configparser',
        'dotenv', 'pytest', 'nose'
    }

    # Some packages from the standard library that might be confused for installable packages
    stdlib_packages = {
        'os', 'sys', 're', 'math', 'random', 'time', 'datetime', 'collections',
        'json', 'csv', 'xml', 'html', 'urllib', 'socket', 'logging', 'threading',
        'multiprocessing', 'asyncio', 'pathlib', 'typing', 'enum', 'abc', 'io',
        'pickle', 'shelve', 'dbm', 'sqlite3', 'zlib', 'gzip', 'zipfile', 'tarfile',
        'hashlib', 'hmac', 'ssl', 'email', 'mimetypes', 'base64', 'bz2', 'lzma',
        'binascii', 'ctypes', 'gc', 'inspect', 'site', 'code', 'codeop', 'pdb',
        'profile', 'pstats', 'timeit', 'trace', 'tracemalloc', 'curses', 'readline',
        'rlcompleter', 'stat', 'fileinput', 'filecmp', 'tempfile', 'glob', 'fnmatch',
        'linecache', 'shlex', 'pwd', 'grp'
    }

    if package_name in common_installable:
        return True

    if package_name in stdlib_packages:
        return False

    # Try to check if the package is importable
    try:
        if importlib.util.find_spec(package_name) is None:
            # Not in standard library, might be installable
            return True
    except (ImportError, ValueError):
        # Not in standard library, might be installable
        return True

    # Guess conservatively
    return True


def main():
    """Scan all Python files in the repository for imports and check against requirements."""
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_dir)

    print(f"Scanning Python files in {repo_dir} for imports...")

    all_packages = set()

    # Scan all Python files
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                # Skip venv files
                if 'venv' in file_path:
                    continue
                try:
                    packages = scan_file_for_imports(file_path)
                    all_packages.update(packages)
                except Exception as e:
                    print(f"Error scanning {file_path}: {e}")

    print(f"Found {len(all_packages)} packages:")
    print(", ".join(sorted(all_packages)))

    # Check which are installable
    installable_packages = set()
    for package in all_packages:
        if check_package_installable(package):
            installable_packages.add(package)

    print(f"\nOf these, {len(installable_packages)} appear to be installable:")
    print(", ".join(sorted(installable_packages)))

    # Check requirements files
    req_file = os.path.join(repo_dir, 'requirements.txt')
    req_memory_file = os.path.join(repo_dir, 'requirements_memory.txt')

    required_packages = set()

    if os.path.exists(req_file):
        with open(req_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name from requirements line
                    package = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
                    required_packages.add(package)

    if os.path.exists(req_memory_file):
        with open(req_memory_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name from requirements line
                    package = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
                    required_packages.add(package)

    print(f"\nRequired packages from requirements files:")
    print(", ".join(sorted(required_packages)))

    # Check for missing requirements
    missing_requirements = installable_packages - required_packages

    if missing_requirements:
        print(f"\n❌ Missing requirements:")
        for package in sorted(missing_requirements):
            print(f"  - {package}")

        print("\nUpdated requirements.txt should include:")
        for package in sorted(required_packages.union(missing_requirements)):
            print(f"{package}")

        return 1
    else:
        print("\n✅ All installable packages are included in requirements files!")
        return 0


if __name__ == '__main__':
    sys.exit(main())
