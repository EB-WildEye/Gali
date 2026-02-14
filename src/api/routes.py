"""
src/api/routes.py — API Route Definitions

Responsibility:
    Defines all REST API endpoints as specified in openapi.yml:
      - POST /chat              → Create a new chat session
      - GET  /chat/{chatId}     → Validate / retrieve a chat session
      - POST /chat/{chatId}/message → Send a message and receive an agent response

    Each route handler is a thin controller: it validates the incoming request,
    delegates business logic to src/core/rag_engine.py, and formats the response.
    No business logic or database calls should live in this file.

Architecture Context:
    Maps 1:1 to the API Gateway routes from the original serverless architecture.
    Follows the Controller pattern — routes are the "glue" between HTTP and core logic.

Related Files:
    - src/main.py             → Routes are registered here
    - src/core/rag_engine.py  → Business logic invoked by route handlers
    - openapi.yml             → Canonical API contract
"""
