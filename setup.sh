#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Creating 'softpotato' project architecture..."

# 1. Create directory hierarchy
mkdir -p softpotato/src/softpotato/{physics,mechanism,technique,grid,solver,core}
mkdir -p softpotato/tests/{unit,integration}

# 2. Root-level files
touch softpotato/pyproject.toml
touch softpotato/README.md

# 3. Core package entrypoints
touch softpotato/src/softpotato/__init__.py
touch softpotato/src/softpotato/simulator.py
touch softpotato/src/softpotato/results.py

# 4. Physics module (sp.physics)
touch softpotato/src/softpotato/physics/__init__.py
touch softpotato/src/softpotato/physics/species.py
touch softpotato/src/softpotato/physics/kinetics.py

# 5. Mechanism module (sp.mechanism)
touch softpotato/src/softpotato/mechanism/__init__.py
touch softpotato/src/softpotato/mechanism/steps.py
touch softpotato/src/softpotato/mechanism/builder.py

# 6. Experimental Technique module (sp.technique)
touch softpotato/src/softpotato/technique/__init__.py
touch softpotato/src/softpotato/technique/base.py
touch softpotato/src/softpotato/technique/chrono.py
touch softpotato/src/softpotato/technique/sweep.py

# 7. Spatial Grid module (sp.grid)
touch softpotato/src/softpotato/grid/__init__.py
touch softpotato/src/softpotato/grid/spatial.py
touch softpotato/src/softpotato/grid/mesh_generators.py

# 8. Numerical Solver module (sp.solver)
touch softpotato/src/softpotato/solver/__init__.py
touch softpotato/src/softpotato/solver/ivp.py

# 9. Core PDE Engine module (sp.core)
touch softpotato/src/softpotato/core/__init__.py
touch softpotato/src/softpotato/core/pde_builder.py
touch softpotato/src/softpotato/core/boundary_conditions.py

# 10. Test suite init files
touch softpotato/tests/__init__.py
touch softpotato/tests/unit/__init__.py
touch softpotato/tests/integration/__init__.py

echo "Done! The 'softpotato' package scaffolding has been generated."