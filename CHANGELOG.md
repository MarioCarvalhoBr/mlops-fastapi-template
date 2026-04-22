# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-04-22

### Added
- **Comprehensive Testing Suite**: Achieved full unit and integration testing coverage for the `sentiment`, `vision`, `retrieval`, and `multimodal` modules, utilizing `unittest.mock` for fast, offline, and deterministic CI/CD execution.
- **AI Assistant Guidelines**: Added `.github/copilot-instructions.md` establishing strict architectural, clean code, and SOLID principles for AI-assisted development.

### Fixed
- **Static Typing (MyPy)**: Resolved Python 3.12 syntax mismatches (pattern matching compatibility) and enforced strict `Optional` typing across all services and routes.
- **Cyclomatic Complexity (Flake8)**: Refactored `load_multimodal_catalog` using the Extract Method pattern to adhere to the strict complexity threshold (McCabe C901), fully respecting the Single Responsibility Principle.
- **Service Layer Contracts**: Corrected exception propagation mapping (`ValueError` vs `RuntimeError`) between the Model, Service, and API layers to ensure predictable HTTP status codes.

## [0.5.0] - 2026-04-22

### Added
- **Multimodal Service**: Introduced true zero-shot cross-modal retrieval mapping Text and Images into a joint vector space using the CLIP architecture (`openai/clip-vit-base-patch32`).
- **File-based Catalog Engine**: Migrated from hardcoded dictionaries to a physical data structure standard (`data/catalog/<product_id>/`). Supports `info.json` and `n` images per product directory.
- **Multimodal Endpoint**: Added `/multimodal-search` accepting concurrent `multipart/form-data` payloads (Text Query + Image File) and applying late fusion embedding strategies.
- **Frontend Expansion**: Added the new "Multimodal Search" tab, retaining all previous capabilities without degradation.


## [0.4.0] - 2026-04-22

### Added
- **Retrieval Service (Semantic Search)**: Integrated the `sentence-transformers/all-MiniLM-L6-v2` model natively via Hugging Face Transformers pipeline for generating dense vector embeddings.
- **In-Memory Vector Database**: Implemented an in-memory product catalog with cosine similarity matching using PyTorch (`torch.nn.functional.cosine_similarity`).
- **Semantic Search Endpoint**: Added the `/semantic-search` endpoint mapping user intent to product features, bypassing strict lexical search limitations.
- **Frontend Dashboard Update**: Introduced the "Semantic Search" tab in the Vue.js interface to test intent-based queries directly against the live catalog.

## [0.3.0] - 2026-04-22

### Added
- **Vision Service (Object Detection)**: Integrated the `facebook/detr-resnet-50` model via Hugging Face pipeline for identifying products in images.
- **Image Upload Endpoint**: Added the `/detect-objects` endpoint in FastAPI supporting `multipart/form-data` for image processing.
- **Dynamic Bounding Boxes (Frontend)**: Enhanced the Vue.js dashboard with a new "Visual Detection" tab that renders bounding boxes, labels, and confidence scores directly over images using HTML5 `<canvas>`.
- **Dependency Management**: Added `pillow`, `python-multipart`, `timm`, and strictly pinned `torchvision>=0.17.0` to resolve PyTorch ABI compatibility issues with Python 3.12.
- **English Standardization**: Refactored the entire codebase, including UI, logs, and docstrings, strictly to English, adhering to global software engineering standards.

## [0.2.0] - 2026-04-22

### Added
- **Commercial Domain**: Project rebranded as "Retail-AI", focusing on E-commerce solutions.
- **Hugging Face Integration**: Real implementation of the `nlptown/bert-base-multilingual-uncased-sentiment` model for product review sentiment analysis.
- **Hardware Acceleration**: Automatic support for GPU (CUDA) or CPU detection via PyTorch.
- **Offline Support**: Configuration for offline execution via environment variables (`hf_hub_offline`).
- **Frontend Dashboard**: Created the `frontend/` directory with a reactive UI using Vue.js 3 and Tailwind CSS.
- **Security and Validation**: Implemented strict Pydantic Schemas for input validation (Security by Design) and text buffer limits.
- **CORS Middleware**: Enabled Cross-Origin Resource Sharing in FastAPI to allow frontend/backend integration.
- **CI Pipeline**: Automated testing and code quality checks via GitHub Actions (`ci.yml`).

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

[0.6.0]: https://github.com/MarioCarvalhoBr/mlops-fastapi-template/releases/tag/v0.6.0
[0.5.0]: https://github.com/MarioCarvalhoBr/mlops-fastapi-template/releases/tag/v0.5.0
[0.4.0]: https://github.com/MarioCarvalhoBr/mlops-fastapi-template/releases/tag/v0.4.0
[0.3.0]: https://github.com/MarioCarvalhoBr/mlops-fastapi-template/releases/tag/v0.3.0
[0.2.0]: https://github.com/MarioCarvalhoBr/mlops-fastapi-template/releases/tag/v0.2.0
[0.1.1]: https://github.com/MarioCarvalhoBr/mlops-fastapi-template/releases/tag/v0.1.1
[0.1.0]: https://github.com/MarioCarvalhoBr/mlops-fastapi-template/releases/tag/v0.1.0