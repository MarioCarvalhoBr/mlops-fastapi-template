class TestHealthEndpoint:
    def test_health_check_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "version": "0.1.0"}

    def test_health_check_has_correct_content_type(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]


class TestPredictEndpoint:
    def test_predict_endpoint(self, client):
        response = client.post("/predict", params={"input": "good movie"})
        assert response.status_code == 200
        assert "prediction" in response.json()

    def test_predict_positive(self, client):
        response = client.post("/predict", params={"input": "This is a great movie!"})
        assert response.status_code == 200
        assert response.json()["prediction"] == "positive"

    def test_predict_negative(self, client):
        response = client.post("/predict", params={"input": "This is bad"})
        assert response.status_code == 200
        assert response.json()["prediction"] == "negative"


class TestAPIDocumentation:
    def test_openapi_schema_accessible(self, client):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema

    def test_swagger_ui_accessible(self, client):
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestErrorHandling:
    def test_nonexistent_endpoint_returns_404(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_invalid_method_on_health_endpoint(self, client):
        response = client.post("/health")
        assert response.status_code == 405  # Method Not Allowed
