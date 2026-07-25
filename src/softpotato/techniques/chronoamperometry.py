from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, NonNegativeFloat, PositiveFloat

from softpotato.core.abcs import BaseTechnique


class Chronoamperometry(PydanticBaseModel, BaseTechnique):
    r"""
    Chronoamperometry (CA) technique waveform generator.

    Generates single- or double-potential step signals $E(t)$ held for fixed durations.
    """

    E_init: float = Field(default=0.0, description="Initial resting potential (V)")
    E_step1: float = Field(..., description="First stepped potential (V)")
    t_step1: PositiveFloat = Field(
        ..., description="Duration of first potential step in seconds (s)"
    )
    E_step2: float | None = Field(
        default=None, description="Second stepped potential (V) for double-step CA"
    )
    t_step2: PositiveFloat | None = Field(
        default=None, description="Duration of second potential step in seconds (s)"
    )
    t_init: NonNegativeFloat = Field(
        default=0.0, description="Initial hold time at E_init prior to step (s)"
    )

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        E_step1: float,
        t_step1: float,
        E_init: float = 0.0,
        E_step2: float | None = None,
        t_step2: float | None = None,
        t_init: float = 0.0,
    ) -> None:
        # Default t_step2 to t_step1 if E_step2 is specified without duration
        if E_step2 is not None and t_step2 is None:
            t_step2 = t_step1

        super().__init__(
            E_init=E_init,
            E_step1=E_step1,
            t_step1=t_step1,
            E_step2=E_step2,
            t_step2=t_step2,
            t_init=t_init,
        )

    @property
    def t_total(self) -> float:
        """Total duration of the experiment across all steps in seconds."""
        total = self.t_init + self.t_step1
        if self.E_step2 is not None and self.t_step2 is not None:
            total += self.t_step2
        return total

    @property
    def t_span(self) -> tuple[float, float]:
        """Simulation time interval $(0, t_{total})$ in seconds."""
        return (0.0, self.t_total)

    def __call__(self, t: float) -> float:
        """
        Evaluate potential $E(t)$ at time $t$.
        """
        if t < self.t_init:
            return float(self.E_init)

        t_after_init = t - self.t_init

        if t_after_init < self.t_step1:
            return float(self.E_step1)

        if self.E_step2 is not None and self.t_step2 is not None:
            t_after_step1 = t_after_init - self.t_step1
            if t_after_step1 <= self.t_step2:
                return float(self.E_step2)
            return float(self.E_step2)

        return float(self.E_step1)
