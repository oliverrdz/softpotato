Installation
============

Requirements
------------

Soft Potato requires **Python 3.10** or higher. We recommend using a virtual environment (such as ``venv`` or ``conda``) to manage dependencies:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate


Stable Release (PyPI)
---------------------

The recommended way to install the latest stable version of Soft Potato is via `PyPI <https://pypi.org/project/softpotato/>`_ using ``pip``:

.. code-block:: bash

   pip install softpotato

To upgrade an existing installation to the latest stable release:

.. code-block:: bash

   pip install --upgrade softpotato

Verify your installation by printing the installed version:

.. code-block:: bash

   python -c "import softpotato as sp; print(sp.__version__)"


Bleeding Edge (GitHub)
----------------------

To access the latest features, active development, and unreleased bug fixes, you can install the bleeding-edge version directly from the `GitHub repository <https://github.com/oliverrdz/softpotato>`_:

.. code-block:: bash

   pip install git+https://github.com/oliverrdz/softpotato.git

To update your bleeding-edge installation to the latest commit on ``main``:

.. code-block:: bash

   pip install --upgrade git+https://github.com/oliverrdz/softpotato.git

You can also target a specific branch or tag if desired:

.. code-block:: bash

   pip install git+https://github.com/oliverrdz/softpotato.git@main


From Source (Development)
-------------------------

If you plan to contribute to Soft Potato or modify the source code, clone the repository and install it in editable mode with development and documentation dependencies:

.. code-block:: bash

   git clone https://github.com/oliverrdz/softpotato.git
   cd softpotato
   pip install -e ".[dev,docs]"

Available optional dependency extras defined in ``pyproject.toml``:

* ``[test]``: Dependencies for running tests (``pytest``, ``pytest-cov``, ``nbmake``).
* ``[docs]``: Dependencies for building the documentation (``sphinx``, ``sphinx-rtd-theme``, ``nbsphinx``, etc.).
* ``[dev]``: All development tools including tests, linters (``ruff``, ``mypy``), and documentation builders.

