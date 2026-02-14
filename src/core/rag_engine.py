"""
src/core/rag_engine.py — RAG Engine (Core Business Logic)

Responsibility:
    Implements the Retrieval Augmented Generation pipeline:
      1. Accepts the user's message and conversation history.
      2. Generates a vector embedding of the user query (via Google Gemini embeddings).
      3. Delegates vector similarity search to src/database/vector_store.py.
      4. Constructs an augmented prompt: retrieved documents + chat history + user query.
      5. Sends the augmented prompt to Google Gemini LLM for response generation.
      6. Returns the generated agent response.

    This file owns prompt engineering, context window management, and LLM interaction.
    It does NOT handle HTTP concerns or direct database access.

Architecture Context:
    Replaces the Chat Lambda's core logic from the original serverless architecture.
    Mirrors the RAG flow described in TECHNICAL.md Section 6 (steps 1-8), with
    Google Gemini replacing Amazon Bedrock for both embeddings and LLM inference.

Related Files:
    - src/api/routes.py            → Invokes this engine from route handlers
    - src/database/vector_store.py → Called for vector similarity search
    - TECHNICAL.md §6              → RAG flow specification
"""
