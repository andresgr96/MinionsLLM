#!/bin/sh

set -e

cd "$(dirname "$0")"

# Extract packages from pyproject.toml and get unique top-level packages
python3 -c "
import toml
import os

# Read pyproject.toml
with open('../pyproject.toml', 'r') as f:
    data = toml.load(f)

# Extract packages from [tool.setuptools] section
packages = data.get('tool', {}).get('setuptools', {}).get('packages', [])

# Get unique top-level packages (before any dots)
top_level_packages = set()
for pkg in packages:
    top_level = pkg.split('.')[0]
    top_level_packages.add(top_level)

# Add examples and tests directories if they exist
additional_dirs = ['examples', 'tests']
for dir_name in additional_dirs:
    if os.path.exists(f'../{dir_name}'):
        top_level_packages.add(dir_name)

# Sort and print all packages
for pkg in sorted(top_level_packages):
    print(pkg)
"
