# 🚀 MLOps FastAPI Template

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0%2B-009688)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository provides a production-grade template for serving Machine Learning models using **FastAPI**. The project focuses on rigorous Software Engineering patterns applied to MLOps, including clean architecture, automated testing, and performance monitoring.

## 📋 Table of Contents
- [🚀 MLOps FastAPI Template](#-mlops-fastapi-template)
  - [📋 Table of Contents](#-table-of-contents)
  - [🌟 Overview](#-overview)
  - [🛠 Technologies Used](#-technologies-used)
  - [📂 Project Structure](#-project-structure)
  - [⚙️ Installation and Configuration](#️-installation-and-configuration)
    - [Prerequisites](#prerequisites)
    - [Step by Step](#step-by-step)
  - [🚀 Usage and Execution](#-usage-and-execution)
  - [🧪 Code Quality and Tests](#-code-quality-and-tests)
    - [Execute Automated Tests](#execute-automated-tests)
    - [Linting and Formatting](#linting-and-formatting)
  - [📈 Load Testing](#-load-testing)
  - [📄 License](#-license)
  - [👥 Authors](#-authors)
  - [🔗 Useful Links](#-useful-links)
  - [Credits and References](#credits-and-references)

---

## 🌟 Overview

Unlike isolated research scripts, this project implements a robust **Service Layer** that isolates the inference logic from the web framework. The architecture follows the `src/` layout, ensuring the source code is testable, modular, and ready for horizontal scalability in cloud environments or containers.

## 🛠 Technologies Used

The project integrates the most modern tools of the Python ecosystem for MLOps:

* **[FastAPI](https://fastapi.tiangolo.com/):** High-performance web framework for building APIs.
* **[Pydantic V2](https://docs.pydantic.dev/):** Data validation and configuration management via environment variables.
* **[Poetry](https://python-poetry.org/):** Dependency management and deterministic virtual environments.
* **[Pytest](https://docs.pytest.org/):** Unit and integration testing suite.
* **[Locust](https://locust.io/):** Load testing to validate latency and RPS (Requests Per Second) under pressure.
* **[MyPy](http://mypy-lang.org/):** Static type checking to prevent runtime errors.
* **[Black](https://github.com/psf/black) & [isort](https://pycqa.github.io/isort/):** Automatic code formatting and import organization.

---

## 📂 Project Structure

The organization follows the principle of separation of concerns:

```text
mlops-fastapi-template/
├── src/                        # Application source code
│   ├── api/                    # FastAPI Routes and Endpoints
│   ├── config/                 # Configuration management (Settings/Pydantic)
│   ├── core/                   # Core components (Logger, etc.)
│   ├── models/                 # ML Models abstraction (Wrappers)
│   ├── services/               # Business and inference logic
│   ├── utils/                  # Helper functions
│   └── main.py                 # Application entry point
├── tests/                      # Automated test suite
│   ├── unit/                   # Isolated function tests
│   ├── integration/            # Endpoints and system integration tests
│   └── performance/            # Load testing scripts (Locust)
├── configs/                    # YAML configuration files
├── Makefile                    # Automation of common tasks
├── pyproject.toml              # Project dependencies and metadata
└── setup_env.sh                # Environment bootstrapping script
```

---

## ⚙️ Installation and Configuration

### Prerequisites
* Python 3.12+
* Poetry installed (`pip install poetry`)

### Step by Step

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/MarioCarvalhoBr/mlops-fastapi-template.git
    cd mlops-fastapi-template
    ```

2.  **Environment Setup:**
    The setup script will create the virtual environment and install dependencies:
    ```bash
    chmod +x setup_env.sh
    ./setup_env.sh
    ```

3.  **Environment Variables:**
    Make sure to configure your `.env` file (created automatically from `.env.example` by the setup script).

---

## 🚀 Usage and Execution

To start the development server with *hot-reload*:

```bash
poetry run uvicorn src.main:app --reload
```

The API will be available at `http://localhost:8000`. Access the interactive documentation at:
* **Swagger UI:** `http://localhost:8000/docs`
* **ReDoc:** `http://localhost:8000/redoc`

---

## 🧪 Code Quality and Tests

The project uses a `Makefile` to standardize development commands.

### Execute Automated Tests
```bash
make test        # Runs all tests (Unit + Integration)
make test-unit   # Runs only unit tests
```

### Linting and Formatting
```bash
make format      # Applies Black and isort
make lint        # Checks compliance with flake8
make type-check  # Runs MyPy for static type verification
```

---

## 📈 Load Testing

To validate how the API behaves with multiple simultaneous users:

1.  Ensure the API is running in one terminal.
2.  In another terminal, execute:
    ```bash
    make load-test
    ```
3.  Access `http://localhost:8089` to configure the number of users and observe real-time latency graphs.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors
- **Mário de Araújo Carvalho** - *Contributor and Developer* - [GitHub](https://github.com/MarioCarvalhoBr)

## 🔗 Useful Links

- **Documentation**: [Docs](https://github.com/AdaptaBrasil/mlops-fastapi-template/tree/main/docs)
- **Issues**: [Bug Tracker](https://github.com/AdaptaBrasil/mlops-fastapi-template/issues)
- **Changelog**: [Version History](https://github.com/AdaptaBrasil/mlops-fastapi-template/blob/main/CHANGELOG.md)
- **Code of Conduct**: [Code of Conduct](https://github.com/AdaptaBrasil/mlops-fastapi-template/blob/main/CODE_OF_CONDUCT.md)

## Credits and References

> "This project was structured and developed based on the educational Software Engineering for MLOps series from PyImageSearch. The base architecture, service isolation, and API construction best practices were implemented following the tutorial [FastAPI for MLOps: Python Project Structure and API Best Practices](https://pyimagesearch.com/2026/04/13/fastapi-for-mlops-python-project-structure-and-api-best-practices/). In turn, the entire validation pipeline — including unit tests, integration tests with Pytest, code quality tools, and dynamic load testing with Locust — was guided by the [Pytest Tutorial: MLOps Testing, Fixtures, and Locust Load Testing](https://pyimagesearch.com/2026/04/20/pytest-tutorial-mlops-testing-fixtures-and-locust-load-testing/). Combining the practices taught in these two articles enabled the creation of a robust *template* that reflects the highest industry standards for deploying Machine Learning models."

**Developed by [Mário de Araújo Carvalho](https://github.com/MarioCarvalhoBr)**
