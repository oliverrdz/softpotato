import math

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, field_validator

from softpotato.core.abcs import BaseTechnique


class CustomWaveform(PydanticBaseModel, BaseTechnique):
    r"""
    Sequential composite technique waveform generator.

    Combines an arbitrary sequence of `BaseTechnique` instances (e.g., CA, LSV, CV)
    to form a multi-segment potential signal $E(t)$ evaluated sequentially over time.
    """

    techniques: list[BaseTechnique] = Field(
        ...,
        description="Ordered list of BaseTechnique instances to execute sequentially.",
    )

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, techniques: list[BaseTechnique]) -> None:
        super().__init__(techniques=techniques)

    @field_validator("techniques")
    @classmethod
    def validate_techniques_not_empty(
        cls, v: list[BaseTechnique]
    ) -> list[BaseTechnique]:
        if not v:
            raise ValueError(
                "CustomWaveform requires at least one technique in the list."
            )
        return v

    @property
    def t_total(self) -> float:
        """Total cumulative duration of all chained techniques in seconds."""
        return float(sum(tech.t_total for tech in self.techniques))

    @property
    def t_span(self) -> tuple[float, float]:
        """Simulation time interval $(0, t_{total})$ in seconds."""
        return (0.0, self.t_total)

    def __call__(self, t: float) -> float:
        """
        Evaluate composite potential $E(t)$ at global time $t$.
        """
        if t <= 0.0:
            return float(self.techniques[0](0.0))

        if t >= self.t_total or math.isclose(t, self.t_total, abs_tol=1e-9):
            last_tech = self.techniques[-1]
            return float(last_tech(last_tech.t_total))

        t_accumulated = 0.0
        for tech in self.techniques:
            t_next = t_accumulated + tech.t_total
            if t <= t_next or math.isclose(t, t_next, abs_tol=1e-9):
                t_local = t - t_accumulated
                return float(tech(t_local))
            t_accumulated = t_next

        last_tech = self.techniques[-1]
        return float(last_tech(last_tech.t_total))
