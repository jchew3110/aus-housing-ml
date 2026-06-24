"""Tests for FastAPI endpoints."""



class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_model_loaded_true(self, client):
        body = client.get("/health").json()
        assert body["model_loaded"] is True

    def test_status_healthy(self, client):
        body = client.get("/health").json()
        assert body["status"] == "healthy"

    def test_timestamp_present(self, client):
        body = client.get("/health").json()
        assert "timestamp" in body
        assert len(body["timestamp"]) > 0


class TestModelInfoEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/model-info")
        assert resp.status_code == 200

    def test_contains_metrics(self, client):
        body = client.get("/model-info").json()
        assert "metrics" in body
        assert "test" in body["metrics"]

    def test_contains_cities(self, client):
        body = client.get("/model-info").json()
        assert "cities_supported" in body
        assert "Sydney" in body["cities_supported"]

    def test_feature_count_positive(self, client):
        body = client.get("/model-info").json()
        assert body["feature_count"] > 0


class TestPredictEndpoint:
    def test_valid_request_returns_200(self, client, valid_predict_payload):
        resp = client.post("/api/v1/predict", json=valid_predict_payload)
        assert resp.status_code == 200

    def test_response_has_prediction(self, client, valid_predict_payload):
        body = client.post("/api/v1/predict", json=valid_predict_payload).json()
        assert "predicted_qoq_pct_change" in body
        assert isinstance(body["predicted_qoq_pct_change"], float)

    def test_direction_is_valid_enum(self, client, valid_predict_payload):
        body = client.post("/api/v1/predict", json=valid_predict_payload).json()
        assert body["direction"] in {"up", "down", "flat"}

    def test_confidence_interval_ordered(self, client, valid_predict_payload):
        body = client.post("/api/v1/predict", json=valid_predict_payload).json()
        ci = body["confidence_interval"]
        assert ci["lower"] <= body["predicted_qoq_pct_change"] + 1e-6
        assert ci["upper"] >= body["predicted_qoq_pct_change"] - 1e-6

    def test_city_reflected_in_response(self, client, valid_predict_payload):
        body = client.post("/api/v1/predict", json=valid_predict_payload).json()
        assert body["city"] == valid_predict_payload["city"]

    def test_invalid_city_returns_422(self, client, valid_predict_payload):
        payload = {**valid_predict_payload, "city": "NotACity"}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    def test_negative_rppi_returns_422(self, client, valid_predict_payload):
        payload = {**valid_predict_payload, "rppi_current": -10.0}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    def test_cash_rate_above_30_returns_422(self, client, valid_predict_payload):
        payload = {**valid_predict_payload, "cash_rate": 35.0}
        resp = client.post("/api/v1/predict", json=payload)
        assert resp.status_code == 422

    def test_all_cities_accepted(self, client, valid_predict_payload):
        cities = ["Sydney", "Melbourne", "Brisbane", "Adelaide", "Perth", "Hobart", "Darwin", "Canberra"]  # noqa: E501
        for city in cities:
            payload = {**valid_predict_payload, "city": city}
            resp = client.post("/api/v1/predict", json=payload)
            assert resp.status_code == 200, f"Failed for city: {city}"

    def test_model_name_in_response(self, client, valid_predict_payload):
        body = client.post("/api/v1/predict", json=valid_predict_payload).json()
        assert "model_name" in body
        assert body["model_name"] == "ridge"
