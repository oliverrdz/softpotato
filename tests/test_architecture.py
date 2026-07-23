import pytest

from softpotato import BaseMesh, BaseSolver


def test_abcs_cannot_be_instantiated_directly():
    """Verify that abstract contracts block direct instantiation."""
    with pytest.raises(TypeError):
        _ = BaseMesh()

    with pytest.raises(TypeError):
        _ = BaseSolver()
