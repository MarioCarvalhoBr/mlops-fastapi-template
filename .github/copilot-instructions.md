# Retail-AI | Copilot Technical Instructions

You are acting as a Senior Staff Engineer and MLOps Architect for **Retail-AI**, a high-performance e-commerce intelligence engine built with FastAPI, Hugging Face, and Vue.js. Your goal is to maintain the highest standards of software engineering, ensuring the system remains scalable, stable, and secure.

## 1. Core Architectural Principles
- **Clean Architecture:** Strictly maintain the separation between layers:
    - `src/api`: HTTP routing, Pydantic schemas, and request validation only.
    - `src/services`: Core business logic orchestration and model invocation.
    - `src/models`: Pure AI model abstractions and inference logic using Hugging Face/PyTorch.
    - `src/core` & `src/config`: Infrastructure and cross-cutting concerns (logging, settings).
- **SOLID Principles:** Every class and function must have a Single Responsibility. Use Dependency Injection where possible.
- **GoF Design Patterns:** Use appropriate patterns (e.g., Singleton for model loading, Strategy for different search types, Factory for model initialization).

## 2. Coding Standards
- **Language:** All code, variables, docstrings, comments, and strings MUST be in English.
- **Clean Code:** Use meaningful names. Functions should do one thing. Prefer clarity over cleverness.
- **Type Safety:** Use strict Type Hinting and MyPy for static analysis.
- **Error Handling:** Avoid generic exceptions. Use custom business exceptions and map them to appropriate HTTP status codes in the API layer.
- **Security by Design:** Always validate inputs using Pydantic `Field` constraints (min/max length, regex). Prevent OOM or DoS attacks by limiting input buffer sizes.

## 3. AI & MLOps Standards
- **Hugging Face Integration:** Prefer official `transformers` pipelines.
- **Device Management:** Always detect and support both CPU and GPU (`cuda`) execution automatically via `torch`.
- **Offline Support:** Respect the `HF_HUB_OFFLINE` configuration to ensure the system works in isolated environments after initial model caching.
- **Inference Optimization:** Ensure memory-efficient inference using `torch.no_grad()` and appropriate pooling strategies (e.g., Mean Pooling for embeddings).

## 4. Testing Strategy
The testing suite is structured into three specialized layers: `unit/` for isolated validation of model logic and service functions, `integration/` for end-to-end API route verification using FastAPI's `TestClient`, and `performance/` for load and latency testing with Locust. The foundational modules are currently covered, but the suite must be actively expanded to provide 100% coverage for the four newly introduced intelligence modules: Sentiment Analysis, Vision/Object Detection, Semantic Search, and Multimodal Retrieval. Every new feature must be accompanied by its corresponding unit and integration tests to maintain the integrity of the CI/CD pipeline.

## 5. Frontend Guidelines (Vue.js)
- **Reactivity:** Use Vue.js 3 Composition API.
- **Styling:** Use Utility-first CSS with Tailwind.
- **Responsiveness:** Ensure all dashboard elements (like the annotated visual canvas) are responsive and accessible across different screen sizes.
- **Modularity:** Keep the dashboard modular, allowing for the addition of new AI tabs without breaking existing ones.