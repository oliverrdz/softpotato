from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, PositiveFloat, PositiveInt

from softpotato.core.abcs import BaseTechnique


class CyclicVoltammetry(PydanticBaseModel, BaseTechnique):
    r"""
    Cyclic Voltammetry (CV) technique waveform generator.

    Generates forward and reverse linear potential sweeps at constant scan rate $v$
    for one or more complete cycles ($n_{cycles} \ge 1$).
    """

    E_start: float = Field(..., description="Starting potential (V)")
    E_vertex1: float = Field(..., description="First vertex reversal potential (V)")
    E_vertex2: float | None = Field(
        default=None,
        description="Second vertex potential (V). Defaults to E_start if None.",
    )
    scan_rate: PositiveFloat = Field(
        ..., description="Scan rate v in Volts per second (V/s)"
    )
    n_cycles: PositiveInt = Field(
        default=1, description="Number of complete potential sweep cycles"
    )

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        E_start: float,
        E_vertex1: float,
        scan_rate: float,
        E_vertex2: float | None = None,
        n_cycles: int = 1,
    ) -> None:
        v2 = E_start if E_vertex2 is None else E_vertex2
        super().__init__(
            E_start=E_start,
            E_vertex1=E_vertex1,
            E_vertex2=v2,
            scan_rate=scan_rate,
            n_cycles=n_cycles,
        )

    @property
    def t_leg1_first(self) -> float:
        """Duration of the very first forward leg ($E_{start} \to E_{vertex1}$)."""
        return abs(self.E_vertex1 - self.E_start) / self.scan_rate

    @property
    def t_leg2(self) -> float:
        """Duration of the reverse leg ($E_{vertex1} \to E_{vertex2}$)."""
        return abs(self.E_vertex2 - self.E_vertex1) / self.scan_rate

    @property
    def cycle_duration_subsequent(self) -> float:
        """Duration of a full subsequent cycle ($E_{vertex2} \to E_{vertex1} \to E_{vertex2}$)."""
        return 2.0 * self.t_leg2

    @property
    def t_total(self) -> float:
        """Total duration across all $n_{cycles}$ cycles."""
        t_cycle_1 = self.t_leg1_first + self.t_leg2
        if self.n_cycles == 1:
            return t_cycle_1
        return t_cycle_1 + (self.n_cycles - 1) * self.cycle_duration_subsequent

    @property
    def t_span(self) -> tuple[float, float]:
        """Simulation time interval $(0, t_{total})$ in seconds."""
        return (0.0, self.t_total)

    def __call__(self, t: float) -> float:
        """
        Evaluate potential $E(t)$ at time $t$.
        """
        if t <= 0.0:
            return self.E_start

        if t >= self.t_total:
            return float(self.E_vertex2)

        # First cycle evaluation
        t_cycle_1 = self.t_leg1_first + self.t_leg2
        if t <= t_cycle_1:
            if t <= self.t_leg1_first:
                direction = 1.0 if self.E_vertex1 > self.E_start else -1.0
                return self.E_start + direction * self.scan_rate * t
            else:
                t_rev = t - self.t_leg1_first
                direction = 1.0 if self.E_vertex2 > self.E_vertex1 else -1.0
                return self.E_vertex1 + direction * self.scan_rate * t_rev

        # Subsequent cycles (k >= 1)
        t_rem = t - t_cycle_1
        t_sub_cycle = self.cycle_duration_subsequent
        t_in_cycle = t_rem % t_sub_cycle

        if t_in_cycle <= self.t_leg2:
            direction = 1.0 if self.E_vertex1 > self.E_vertex2 else -1.0
            return self.E_vertex2 + direction * self.scan_rate * t_in_cycle
        else:
            t_rev = t_in_cycle - self.t_leg2
            direction = 1.0 if self.E_vertex2 > self.E_vertex1 else -1.0
            return self.E_vertex1 + direction * self.scan_rate * t_rev
