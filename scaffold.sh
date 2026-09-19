#!/bin/bash

# Scaffold the Soft Potato 3.0 src/ directory structure
mkdir -p .github/workflows
mkdir -p src/softpotato/core
mkdir -p src/softpotato/kinetics
mkdir -p src/softpotato/geometry
mkdir -p src/softpotato/techniques
mkdir -p src/softpotato/simulate
mkdir -p tests

# Create root configuration and documentation files
touch .github/workflows/ci.yml
touch pyproject.toml
touch README.md
touch LICENSE

# Initialize core namespace and module files
touch src/softpotato/__init__.py
touch src/softpotato/core/__init__.py src/softpotato/core/species.py src/softpotato/core/reactions.py
touch src/softpotato/kinetics/__init__.py src/softpotato/kinetics/models.py
touch src/softpotato/geometry/__init__.py src/softpotato/geometry/grids.py src/softpotato/geometry/electrodes.py
touch src/softpotato/techniques/__init__.py src/softpotato/techniques/voltammetry.py src/softpotato/techniques/step.py
touch src/softpotato/simulate/__init__.py src/softpotato/simulate/solver.py src/softpotato/simulate/efd.py src/softpotato/simulate/ifd.py

# Initialize pytest suite files
touch tests/conftest.py
touch tests/test_core.py
touch tests/test_kinetics.py
touch tests/test_geometry_laplacians.py
touch tests/test_solvers.py

echo "Soft Potato 3.0 repository structure generated successfully."