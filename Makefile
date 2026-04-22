.PHONY: test test-unit test-integration format lint type-check load-test clean

test:
	poetry run pytest tests/ -v

test-unit:
	poetry run pytest tests/unit/ -v

test-integration:
	poetry run pytest tests/integration/ -v

format:
	poetry run black .
	poetry run isort .

lint:
	poetry run flake8 .

type-check:
	poetry run mypy .

load-test:
	poetry run locust -f tests/performance/locustfile.py --host=http://localhost:8000

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +