"""FastAPI app entry point for ReasonGuard."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .calibration import calibrate
from .config import settings
from .metrics import REQUESTS_TOTAL
from .models import CalibrateRequest, CalibrateResponse, HealthResponse, ScoreRequest, ScoreResponse
from .proxy import extract_content_from_response, forward_chat_completion
from .scoring.pipeline import score_trace
from .trace_parser import split_trace_and_answer

app = FastAPI(
    title="ReasonGuard",
    description="Score the reliability of reasoning LLM output in a single pass.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
async def health():
    REQUESTS_TOTAL.labels(endpoint="/health").inc()
    return HealthResponse()


@app.get("/metrics")
async def metrics():
    REQUESTS_TOTAL.labels(endpoint="/metrics").inc()
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    REQUESTS_TOTAL.labels(endpoint="/score").inc()
    result = score_trace(
        reasoning_trace=req.reasoning_trace,
        final_answer=req.final_answer,
        model=req.model,
    )
    return ScoreResponse(**result.model_dump())


@app.post("/calibrate", response_model=CalibrateResponse)
async def calibrate_endpoint(req: CalibrateRequest):
    REQUESTS_TOTAL.labels(endpoint="/calibrate").inc()
    return calibrate(model=req.model, traces=req.traces)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    REQUESTS_TOTAL.labels(endpoint="/v1/chat/completions").inc()
    body = await request.json()

    # Forward to upstream LLM
    try:
        upstream_response = await forward_chat_completion(body)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upstream LLM error: {exc}")

    # Extract content and score it
    content = extract_content_from_response(upstream_response)
    if content:
        trace, answer = split_trace_and_answer(content)
        model_name = upstream_response.get("model", body.get("model", "unknown"))
        result = score_trace(
            reasoning_trace=trace or content,
            final_answer=answer,
            model=model_name,
        )
        upstream_response["reasonguard"] = result.model_dump()
    else:
        upstream_response["reasonguard"] = None

    return JSONResponse(content=upstream_response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
