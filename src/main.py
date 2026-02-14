"""
src/main.py — Application Entry Point

Responsibility:
    This is the FastAPI application factory. It initializes the FastAPI instance,
    registers API route handlers from src/api/routes.py, and configures
    application-wide settings (CORS, middleware, lifespan events).

    On startup, it loads environment variables via python-dotenv and initializes
    shared resources (e.g., the LanceDB connection from src/database/vector_store.py).

Architecture Context:
    Replaces the AWS API Gateway + Lambda invocation layer from the original
    serverless architecture. Acts as the single HTTP entry point for all
    REST endpoints defined in openapi.yml.

Related Files:
    - src/api/routes.py      → Route definitions mounted here
    - src/database/vector_store.py → Initialized during app startup
    - .env                   → API keys loaded at startup
"""
