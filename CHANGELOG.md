# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-22

### Added
- **Domínio Comercial**: O projeto foi rebatizado como "Retail-AI", focado em soluções para E-commerce.
- **Hugging Face Integration**: Implementação real do modelo `nlptown/bert-base-multilingual-uncased-sentiment` para análise de sentimento de avaliações.
- **Hardware Acceleration**: Suporte automático para detecção e uso de GPU (CUDA) ou CPU via PyTorch.
- **Offline Support**: Configuração para execução em modo offline via variáveis de ambiente.
- **Frontend Dashboard**: Criação da pasta `frontend/` com interface reativa em Vue.js 3 e Tailwind CSS.
- **Segurança e Validação**: Implementação de Pydantic Schemas rigorosos para validação de entrada (Security by Design) e limites de buffer de texto.
- **CORS Middleware**: Habilitação de Cross-Origin Resource Sharing no FastAPI para permitir integração frontend/backend.
- **Pipeline de CI**: Automação de testes e qualidade via GitHub Actions (`ci.yml`).

## [0.1.1] - 2026-04-22
### Added
- Link to `CHANGELOG.md` in the README.
### Changed
- Refined template documentation and structure.

## [0.1.1] - 2026-04-22

### Added
- Link to `CHANGELOG.md` in the README.

### Changed
- Refined template documentation and structure.

## [0.1.0] - 2026-04-22

### Added
- Initial project structure for FastAPI-based MLOps serving.
- `src/api` with initial endpoints (health check and inference routes).
- `src/services` featuring the `InferenceService` to decouple business logic from the routing layer.
- `src/models` with an abstraction layer and a `DummyModel` for initial testing.
- `src/core` setup including generic structured logging in `logger.py`.
- `src/config` implementation utilizing Pydantic V2 and `config.yaml` for environment variables and configuration management.
- Comprehensive test suite in `tests/`:
  - Unit tests for services and models.
  - Integration tests for API endpoints.
  - Performance and load tests using Locust (`locustfile.py`).
- Development tooling and scripts:
  - `Makefile` for automated tasks (`test`, `lint`, `format`, `type-check`, `load-test`).
  - `setup_env.sh` for fast virtual environment bootstrapping.
  - Formatting and linting established with Black, isort, and MyPy.
- Dependency management using Poetry (`pyproject.toml`).
- Base `README.md` with full project context and instructions.
- `CODE_OF_CONDUCT.md` and `LICENSE` (MIT).

[0.1.1]: https://github.com/MarioCarvalhoBr/mlops-fastapi-template/releases/tag/v0.1.1
[0.1.0]: https://github.com/MarioCarvalhoBr/mlops-fastapi-template/releases/tag/v0.1.0
