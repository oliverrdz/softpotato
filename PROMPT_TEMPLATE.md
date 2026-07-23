I am working on the softpotato project. Here is our current project memory file (GEMINI.md):

<paste the contents of GEMINI.txt here>

Please implement the active task from our checklist:
1. File: `src/softpotato/mesh/uniform_1d.py`
2. Class: `Uniform1DMesh` inheriting from `BaseMesh` in `src/softpotato/core/abcs.py`.
3. Functionality: 
   - Accept domain bounds (`x_min`, `x_max`) and total node count (`n_points`).
   - Implement `get_nodes()` returning a 1D NumPy array of spatial coordinates.
   - Implement `num_nodes()` returning the node count.
   - Provide helper properties for grid spacing `dx` and domain length `L`.
4. Include strict type hints (`typing`), Pydantic validation for inputs, and clear docstrings.
5. Provide a matching test suite in `tests/test_mesh.py` using `pytest`.
