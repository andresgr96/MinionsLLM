#!/bin/sh

set -e

cd "$(dirname "$0")"

# Extract packages from pyproject.toml and get unique top-level packages
python3 -c "
import toml

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

# Sort and print all packages
for pkg in sorted(top_level_packages):
    print(pkg)
"
