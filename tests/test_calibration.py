import math

from fastapi.testclient import TestClient

from src.reasonguard.calibration import calibrate, _mean, _std
from src.reasonguard.main import app
from src.reasonguard.scoring.pipeline import get_calibration, _calibration_store

client = TestClient(app)


class TestMeanStd:
    def test_mean_basic(self):
        assert _mean([1.0, 2.0, 3.0]) == 2.0

    def test_mean_single(self):
        assert _mean([5.0]) == 5.0

    def test_mean_empty(self):
        assert _mean([]) == 0.0

    def test_std_basic(self):
        # [1, 2, 3] -> μ=2, variance=2/3, σ=sqrt(2/3)
        result = _std([1.0, 2.0, 3.0])
        assert math.isclose(result, math.sqrt(2.0 / 3.0), rel_tol=1e-6)

    def test_std_single_returns_default(self):
        # With only one value, std returns 1.0 (safe default)
        assert _std([42.0]) == 1.0

    def test_std_empty_returns_default(self):
        assert _std([]) == 1.0

    def test_std_identical_values(self):
        # All same -> variance=0 -> returns epsilon
        result = _std([5.0, 5.0, 5.0])
        assert result == 1e-8


class TestCalibrate:
    def test_calibrate_basic(self):
        traces = [
            {"reasoning_trace": "Maybe A. Let me check.", "final_answer": "A. Confidence: 80%"},
            {"reasoning_trace": "Perhaps B.", "final_answer": "B. Confidence: 60%"},
            {"reasoning_trace": "The answer is C.", "final_answer": "C. Confidence: 95%"},
        ]
        result = calibrate("test-cal-model", traces)
        assert result.model == "test-cal-model"
        assert result.n_traces == 3
        assert result.mu_hvr >= 0
        assert result.sigma_hvr > 0
        assert 0 < result.mu_verb < 1
        assert result.sigma_verb > 0

    def test_calibrate_stores_in_pipeline(self):
        traces = [
            {"reasoning_trace": "Maybe A.", "final_answer": "A. Confidence: 70%"},
            {"reasoning_trace": "Perhaps B.", "final_answer": "B. Confidence: 90%"},
        ]
        calibrate("pipeline-check-model", traces)
        cal = get_calibration("pipeline-check-model")
        assert "mu_hvr" in cal
        assert "sigma_hvr" in cal
        assert "mu_verb" in cal
        assert "sigma_verb" in cal

    def test_calibrate_no_verbalized_confidence(self):
        traces = [
            {"reasoning_trace": "Maybe A.", "final_answer": "Answer is A."},
            {"reasoning_trace": "Perhaps B.", "final_answer": "Answer is B."},
        ]
        result = calibrate("no-verb-model", traces)
        # Falls back to defaults when no verbalized confidence found
        assert result.mu_verb == 0.8
        assert result.sigma_verb == 0.15

    def test_calibrate_updates_scoring(self):
        # Calibrate with known values, then check get_calibration returns them
        traces = [
            {"reasoning_trace": "Let me check. Let me verify.", "final_answer": "X. Confidence: 90%"},
            {"reasoning_trace": "Let me check. Let me verify.", "final_answer": "X. Confidence: 90%"},
            {"reasoning_trace": "Maybe possibly perhaps.", "final_answer": "Y. Confidence: 40%"},
        ]
        result = calibrate("update-model", traces)
        cal = get_calibration("update-model")
        assert math.isclose(cal["mu_hvr"], result.mu_hvr, rel_tol=1e-6)
        assert math.isclose(cal["sigma_hvr"], result.sigma_hvr, rel_tol=1e-6)


class TestCalibrateEndpoint:
    def test_calibrate_endpoint(self):
        resp = client.post("/calibrate", json={
            "model": "api-test-model",
            "traces": [
                {"reasoning_trace": "Maybe A.", "final_answer": "A. Confidence: 80%"},
                {"reasoning_trace": "Perhaps B.", "final_answer": "B. Confidence: 60%"},
                {"reasoning_trace": "Clearly C.", "final_answer": "C. Confidence: 99%"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "api-test-model"
        assert data["n_traces"] == 3
        assert "mu_hvr" in data
        assert "sigma_hvr" in data
        assert "mu_verb" in data
        assert "sigma_verb" in data

    def test_calibrate_endpoint_empty_traces(self):
        resp = client.post("/calibrate", json={
            "model": "empty-model",
            "traces": [],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_traces"] == 0
        # Empty HVR list -> mean=0, std=1.0 (default)
        assert data["mu_hvr"] == 0.0
        assert data["sigma_hvr"] == 1.0

    def test_calibrate_then_score_uses_calibration(self):
        # Calibrate
        client.post("/calibrate", json={
            "model": "calibrated-model",
            "traces": [
                {"reasoning_trace": "Maybe.", "final_answer": "A. Confidence: 50%"},
                {"reasoning_trace": "Perhaps.", "final_answer": "B. Confidence: 50%"},
                {"reasoning_trace": "Possibly.", "final_answer": "C. Confidence: 50%"},
            ],
        })
        # Score with that model — should use calibrated stats
        resp = client.post("/score", json={
            "reasoning_trace": "The answer is clearly D. Let me verify.",
            "final_answer": "D. Confidence: 95%",
            "model": "calibrated-model",
        })
        data = resp.json()
        assert data["decision"] == "ACCEPT"
        assert data["selfdoubt_score"] is not None

    def test_calibrate_missing_model(self):
        resp = client.post("/calibrate", json={
            "traces": [{"reasoning_trace": "x", "final_answer": "y"}],
        })
        assert resp.status_code == 422
