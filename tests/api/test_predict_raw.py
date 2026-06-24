"""Tests for /api/v1/predict/raw and /api/v1/predict/batch endpoints."""

import pytest


@pytest.fixture
def raw_predict_payload():
    """Minimal valid raw prediction payload — 6 quarters of RPPI + macro."""
    rppi = [
        {"year": 2019, "quarter": 1, "rppi_index": 140.0},
        {"year": 2019, "quarter": 2, "rppi_index": 141.5},
        {"year": 2019, "quarter": 3, "rppi_index": 142.0},
        {"year": 2019, "quarter": 4, "rppi_index": 143.0},
        {"year": 2020, "quarter": 1, "rppi_index": 144.5},
        {"year": 2020, "quarter": 2, "rppi_index": 145.0},
    ]
    macro = [
        {"year": 2019, "quarter": 1, "cash_rate": 1.5, "cpi": 115.0, "unemployment_rate": 5.0},
        {"year": 2019, "quarter": 2, "cash_rate": 1.25, "cpi": 115.5, "unemployment_rate": 5.1},
        {"year": 2019, "quarter": 3, "cash_rate": 1.0, "cpi": 116.0, "unemployment_rate": 5.2},
        {"year": 2019, "quarter": 4, "cash_rate": 0.75, "cpi": 116.5, "unemployment_rate": 5.3},
        {"year": 2020, "quarter": 1, "cash_rate": 0.5, "cpi": 117.0, "unemployment_rate": 5.5},
        {"year": 2020, "quarter": 2, "cash_rate": 0.25, "cpi": 117.5, "unemployment_rate": 7.0},
    ]
    return {"city": "Sydney", "rppi_history": rppi, "macro_history": macro}


@pytest.fixture
def batch_predict_payload(valid_predict_payload):
    mel_payload = {**valid_predict_payload, "city": "Melbourne"}
    return {"requests": [valid_predict_payload, mel_payload]}


class TestPredictRawEndpoint:
    def test_returns_200(self, client, raw_predict_payload):
        resp = client.post("/api/v1/predict/raw", json=raw_predict_payload)
        assert resp.status_code == 200, resp.json()

    def test_response_has_prediction(self, client, raw_predict_payload):
        body = client.post("/api/v1/predict/raw", json=raw_predict_payload).json()
        assert "predicted_qoq_pct_change" in body
        assert isinstance(body["predicted_qoq_pct_change"], float)

    def test_direction_valid(self, client, raw_predict_payload):
        body = client.post("/api/v1/predict/raw", json=raw_predict_payload).json()
        assert body["direction"] in {"up", "down", "flat"}

    def test_city_reflected(self, client, raw_predict_payload):
        body = client.post("/api/v1/predict/raw", json=raw_predict_payload).json()
        assert body["city"] == "Sydney"

    def test_prediction_period_is_next_quarter(self, client, raw_predict_payload):
        body = client.post("/api/v1/predict/raw", json=raw_predict_payload).json()
        # Last period in history is 2020Q2 → prediction should be for 2020Q3
        assert body["year"] == 2020
        assert body["quarter"] == 3

    def test_confidence_interval_ordered(self, client, raw_predict_payload):
        body = client.post("/api/v1/predict/raw", json=raw_predict_payload).json()
        ci = body["confidence_interval"]
        assert ci["lower"] <= body["predicted_qoq_pct_change"] + 1e-6
        assert ci["upper"] >= body["predicted_qoq_pct_change"] - 1e-6

    def test_mismatched_history_lengths_returns_422(self, client, raw_predict_payload):
        payload = {**raw_predict_payload}
        payload["macro_history"] = payload["macro_history"][:4]
        resp = client.post("/api/v1/predict/raw", json=payload)
        assert resp.status_code == 422

    def test_period_mismatch_returns_422(self, client, raw_predict_payload):
        payload = dict(raw_predict_payload)
        macro = list(payload["macro_history"])
        macro[0] = {**macro[0], "quarter": 3}  # mismatched period
        payload["macro_history"] = macro
        resp = client.post("/api/v1/predict/raw", json=payload)
        assert resp.status_code == 422

    def test_too_few_observations_returns_422(self, client, raw_predict_payload):
        # Min is 6; sending 5 should fail validation
        payload = {
            **raw_predict_payload,
            "rppi_history": raw_predict_payload["rppi_history"][:5],
            "macro_history": raw_predict_payload["macro_history"][:5],
        }
        resp = client.post("/api/v1/predict/raw", json=payload)
        assert resp.status_code == 422

    def test_invalid_city_returns_422(self, client, raw_predict_payload):
        payload = {**raw_predict_payload, "city": "Atlantis"}
        resp = client.post("/api/v1/predict/raw", json=payload)
        assert resp.status_code == 422

    def test_all_cities_accepted(self, client, raw_predict_payload):
        cities = ["Sydney", "Melbourne", "Brisbane", "Adelaide",
                  "Perth", "Hobart", "Darwin", "Canberra"]
        for city in cities:
            payload = {**raw_predict_payload, "city": city}
            resp = client.post("/api/v1/predict/raw", json=payload)
            assert resp.status_code == 200, f"Failed for {city}: {resp.json()}"


class TestBatchPredictEndpoint:
    def test_returns_200(self, client, batch_predict_payload):
        resp = client.post("/api/v1/predict/batch", json=batch_predict_payload)
        assert resp.status_code == 200, resp.json()

    def test_response_count_matches_input(self, client, batch_predict_payload):
        body = client.post("/api/v1/predict/batch", json=batch_predict_payload).json()
        assert len(body["predictions"]) == len(batch_predict_payload["requests"])

    def test_cities_preserved_in_order(self, client, batch_predict_payload):
        body = client.post("/api/v1/predict/batch", json=batch_predict_payload).json()
        assert body["predictions"][0]["city"] == "Sydney"
        assert body["predictions"][1]["city"] == "Melbourne"

    def test_all_directions_valid(self, client, batch_predict_payload):
        body = client.post("/api/v1/predict/batch", json=batch_predict_payload).json()
        for pred in body["predictions"]:
            assert pred["direction"] in {"up", "down", "flat"}

    def test_empty_list_returns_422(self, client):
        resp = client.post("/api/v1/predict/batch", json={"requests": []})
        assert resp.status_code == 422
