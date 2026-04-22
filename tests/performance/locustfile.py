from locust import HttpUser, between, task


class MLAPIUser(HttpUser):
    """
    Locust user class for testing the ML API.
    Simulates a user making requests to the API endpoints.
    """

    # Wait between 1 and 3 seconds between requests
    wait_time = between(1, 3)

    @task(10)
    def test_predict(self):
        """
        Test the predict endpoint.
        This task has weight 10, making it the most frequently called.
        """
        payload = {"input": "The movie was good"}
        with self.client.post("/predict", params=payload, catch_response=True) as response:
            if response.status_code == 200:
                response_data = response.json()
                if "prediction" in response_data:
                    response.success()
                else:
                    response.failure(f"Missing prediction in response: {response_data}")
            else:
                response.failure(f"HTTP {response.status_code}")

    def on_start(self):
        """
        Called when a user starts testing.
        Used for setup tasks like authentication.
        """
        # Verify the API is reachable
        response = self.client.get("/health")
        if response.status_code != 200:
            print(f"Warning: API health check failed with status {response.status_code}")
