from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, PositiveFloat

from softpotato.core.abcs import BaseTechnique


class LinearSweepVoltammetry(PydanticBaseModel, BaseTechnique):
    r"""
    Linear Sweep Voltammetry (LSV) technique waveform generator.

    Generates a continuous, unidirectional linear potential sweep from $E_{start}$
    to $E_{end}$ at a constant scan rate $v$.
    """

    E_start: float = Field(..., description="Starting potential (V)")
    E_end: float = Field(..., description="End potential (V)")
    scan_rate: PositiveFloat = Field(
        ..., description="Scan rate v in Volts per second (V/s)"
    )

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        E_start: float,
        E_end: float,
        scan_rate: float,
    ) -> None:
        super().__init__(
            E_start=E_start,
            E_end=E_end,
            scan_rate=scan_rate,
        )

    @property
    def t_total(self) -> float:
        """Total duration of the potential sweep in seconds."""
        return abs(self.E_end - self.E_start) / self.scan_rate

    @property
    def t_span(self) -> tuple[float, float]:
        """Simulation time interval $(0, t_{total})$ in seconds."""
        return (0.0, self.t_total)

    def __call__(self, t: float) -> float:
        """
        Evaluate potential $E(t)$ at time $t$.
        """
        if t <= 0.0:
            return float(self.E_start)

        if t >= self.t_total:
            return float(self.E_end)

        direction = 1.0 if self.E_end > self.E_start else -1.0
        return self.E_start + direction * self.scan_rate * t
