"""API integration tests for ReasonGuard endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.reasonguard.main import app

client = TestClient(app)


class TestHealth:
    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestScore:
    def test_score_clean_trace(self):
        resp = client.post("/score", json={
            "reasoning_trace": "The answer is clearly B. Let me verify by substituting back.",
            "final_answer": "The answer is B. Confidence: 95%",
            "model": "test-model",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["hvr"] == 0.0
        assert data["hedge_count"] == 0
        assert data["decision"] == "ACCEPT"
        assert data["tier"] == 1
        assert data["reliable"] is True
        assert data["verbalized_confidence"] == 0.95

    def test_score_uncertain_trace(self):
        resp = client.post("/score", json={
            "reasoning_trace": "Maybe the answer is A. Perhaps B. Possibly C. Probably A.",
            "final_answer": "The answer is A. Confidence: 30%",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["hvr"] > 1.0
        assert data["hedge_count"] >= 4
        assert data["tier"] == 2

    def test_score_no_confidence(self):
        resp = client.post("/score", json={
            "reasoning_trace": "Clearly the answer is B.",
            "final_answer": "The answer is B.",
        })
        data = resp.json()
        assert data["verbalized_confidence"] is None
        # Without verbalized confidence, selfdoubt_score is None
        assert data["selfdoubt_score"] is None

    def test_score_missing_fields(self):
        resp = client.post("/score", json={"reasoning_trace": "test"})
        assert resp.status_code == 422  # validation error

    def test_score_default_model(self):
        resp = client.post("/score", json={
            "reasoning_trace": "test",
            "final_answer": "answer",
        })
        assert resp.status_code == 200

    def test_score_mi_estimate_is_null(self):
        resp = client.post("/score", json={
            "reasoning_trace": "test trace",
            "final_answer": "answer Confidence: 50%",
        })
        data = resp.json()
        assert data["mi_estimate"] is None


class TestChatCompletions:
    UPSTREAM_RESPONSE = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "qwen3-4b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "<think>Let me check this carefully. Maybe it's A.</think>The answer is B. Confidence: 90%",
            },
            "finish_reason": "stop",
        }],
    }

    @patch("src.reasonguard.main.forward_chat_completion", new_callable=AsyncMock)
    def test_chat_completions_augments_response(self, mock_forward):
        mock_forward.return_value = dict(self.UPSTREAM_RESPONSE)
        resp = client.post("/v1/chat/completions", json={
            "model": "qwen3-4b",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
        })
        assert resp.status_code == 200
        data = resp.json()

        # Original response preserved
        assert data["id"] == "chatcmpl-123"
        assert data["choices"][0]["message"]["content"].startswith("<think>")

        # ReasonGuard metadata added
        rg = data["reasonguard"]
        assert rg is not None
        assert "hvr" in rg
        assert "decision" in rg
        assert "hedge_count" in rg
        assert "verify_count" in rg
        assert rg["verbalized_confidence"] == 0.90

    @patch("src.reasonguard.main.forward_chat_completion", new_callable=AsyncMock)
    def test_chat_completions_clean_trace_tier1(self, mock_forward):
        mock_forward.return_value = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "model": "test",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<think>Let me verify this. I can confirm the answer.</think>The answer is 4. Confidence: 99%",
                },
                "finish_reason": "stop",
            }],
        }
        resp = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "2+2?"}],
        })
        rg = resp.json()["reasonguard"]
        assert rg["hvr"] == 0.0
        assert rg["tier"] == 1
        assert rg["decision"] == "ACCEPT"

    @patch("src.reasonguard.main.forward_chat_completion", new_callable=AsyncMock)
    def test_chat_completions_empty_content(self, mock_forward):
        mock_forward.return_value = {
            "id": "chatcmpl-789",
            "object": "chat.completion",
            "model": "test",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }],
        }
        resp = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
        })
        data = resp.json()
        assert data["reasonguard"] is None

    @patch("src.reasonguard.main.forward_chat_completion", new_callable=AsyncMock)
    def test_chat_completions_upstream_error(self, mock_forward):
        mock_forward.side_effect = Exception("connection refused")
        resp = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 502

    @patch("src.reasonguard.main.forward_chat_completion", new_callable=AsyncMock)
    def test_chat_completions_no_choices(self, mock_forward):
        mock_forward.return_value = {
            "id": "chatcmpl-000",
            "object": "chat.completion",
            "model": "test",
            "choices": [],
        }
        resp = client.post("/v1/chat/completions", json={
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
        })
        data = resp.json()
        assert data["reasonguard"] is None
