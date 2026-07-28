import numpy as np
import pandas as pd
import pytest

from softpotato.results import SimulationResult


@pytest.fixture
def sample_result():
    t = np.linspace(0, 1, 10)
    E = np.linspace(0.0, -0.5, 10)
    i = np.sin(t) * 1e-6
    x = np.linspace(0, 1e-4, 50)
    concentrations = {
        "ox": np.outer(np.ones(50), np.ones(10)),
        "red": np.outer(np.zeros(50), np.ones(10)),
    }
    metadata = {"technique": "LSV", "scan_rate": 0.1}
    return SimulationResult(
        t=t, E=E, i=i, x=x, concentrations=concentrations, metadata=metadata
    )


def test_simulation_result_initialization(sample_result):
    assert len(sample_result.t) == 10
    assert len(sample_result.E) == 10
    assert len(sample_result.i) == 10
    assert len(sample_result.x) == 50
    assert "ox" in sample_result.concentrations
    assert "red" in sample_result.concentrations


def test_to_dataframe(sample_result):
    df = sample_result.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["time", "potential", "current"]
    assert len(df) == 10


def test_to_csv(sample_result, tmp_path):
    file_path = tmp_path / "output.csv"
    sample_result.to_csv(file_path)
    assert file_path.exists()
    df = pd.read_csv(file_path)
    assert len(df) == 10


def test_to_json(sample_result, tmp_path):
    file_path = tmp_path / "metadata.json"
    sample_result.to_json(file_path)
    assert file_path.exists()


def test_to_npz(sample_result, tmp_path):
    file_path = tmp_path / "output.npz"
    sample_result.to_npz(file_path)
    assert file_path.exists()

    data = np.load(file_path)
    assert "t" in data
    assert "E" in data
    assert "i" in data
    assert "x" in data
    assert "c_ox" in data
    assert "c_red" in data


def test_plotting_helpers(sample_result):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    res_ax = sample_result.plot_cv(ax=ax)
    assert res_ax is not None
    plt.close(fig)

    fig, ax = plt.subplots()
    res_ax_prof = sample_result.plot_profiles(time_index=-1, ax=ax)
    assert res_ax_prof is not None
    plt.close(fig)
