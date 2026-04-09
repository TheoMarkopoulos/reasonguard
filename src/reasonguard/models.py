from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    reasoning_trace: str
    final_answer: str
    model: str = "unknown"


class ReasonGuardResult(BaseModel):
    reliable: bool
    tier: int
    hvr: float
    hedge_count: int
    verify_count: int
    verbalized_confidence: float | None = None
    selfdoubt_score: float | None = None
    decision: str
    mi_estimate: float | None = None


class ScoreResponse(ReasonGuardResult):
    pass


class HealthResponse(BaseModel):
    status: str = "ok"


class CalibrateRequest(BaseModel):
    model: str
    traces: list[dict] = Field(
        ...,
        description="List of {reasoning_trace, final_answer} dicts (target: 90 traces)",
    )


class CalibrateResponse(BaseModel):
    model: str
    mu_hvr: float
    sigma_hvr: float
    mu_verb: float
    sigma_verb: float
    n_traces: int
