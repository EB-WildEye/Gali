"""
Gali - RAG Engine (Production-Grade)
======================================
Orchestrates the full Retrieval-Augmented Generation pipeline with
clinical-grade safety, intelligent memory management, and advanced
retrieval logic.

Architecture (OOP)
------------------
``ClinicalSafetyScanner``
    Fuzzy-logic weighted keyword engine that computes an urgency
    score (0.0–1.0) from the user's input.  Combines keyword base
    weights with intensifier multipliers and multi-match boosting.
    Classifies into Red/Orange/Green flags.

``MemoryManager``
    Token-aware conversation memory.  Tracks token counts via
    Gemini's counting API.  When a session exceeds ``TOKEN_LIMIT``,
    auto-summarizes the conversation, scrubs PII (keeping only
    patient Name and Location), and replaces MongoDB history
    with the sanitized summary.

``MongoSessionManager``
    Async session store backed by ``motor``.  Full session isolation
    by ``session_id``.  Supports history retrieval, message append,
    bulk replace (for summarization), and session lifecycle.

``RAGManager``
    Stateless pipeline orchestrator.  Flow:
        1. New session → inject greeting
        2. Safety scan → urgency score + flag
        3. RED flag → immediate redirect (skip LLM)
        4. Memory check → auto-summarize if over token limit
        5. Fetch history → last N messages
        6. Refine query → strip fillers, inject history context
        7. Retrieve → vector search (over-fetch 2×)
        8. Re-rank → distance + keyword boost + source diversity
        9. Build prompt → system + RAG template with safety data
        10. Call Gemini → async LLM invocation
        11. Append disclaimer → mandatory on every response
        12. Save → persist to MongoDB
        13. Return answer

AWS Lambda Readiness
--------------------
- MongoDB client: **module-level singleton** — survives warm invocations.
- LanceDB connection: singleton (via ``GaliVectorStore``).
- ``RAGManager``: no request-scoped state — safe for concurrent use.
- All DB I/O: async via motor.

Usage:
    from gali.src.core.rag_engine import RAGManager
    rag = RAGManager(vector_store, embedder)
    answer = await rag.generate_response("session_123", "מה המינון?")
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import motor.motor_asyncio

from gali.config.prompt_templates import DISCLAIMER, EMERGENCY_KEYWORDS, GREEN_FLAG_PREFIX, INTENSIFIERS, MODERATE_KEYWORDS, NEW_SESSION_GREETING, NO_CONTEXT_RESPONSE_HE, ORANGE_FLAG_RESPONSE, RAG_PROMPT_TEMPLATE, RED_FLAG_RESPONSE, SUMMARIZATION_PROMPT, SYSTEM_PROMPT, URGENT_KEYWORDS
from gali.config.settings import settings
from gali.src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Type aliases ───────────────────────────────────────────────────────
ChatMessage = dict[str, str]
SearchResult = dict[str, str | int | float | list[float]]
UrgencyAssessment = dict[str, float | str | list[str]]

# ── Hebrew filler words for query refinement ──────────────────────────
_HEBREW_FILLERS: set[str] = {"מה", "איך", "האם", "של", "את", "על", "עם", "אני", "לי", "זה", "הוא", "היא", "אנחנו", "יש", "אין", "צריך", "רוצה", "בבקשה", "תודה", "שלום", "אפשר", "לדעת", "להגיד", "לספר", "בקשר", "לגבי"}

# ── PII regex patterns ────────────────────────────────────────────────
_RE_ISRAELI_ID = re.compile(r"\b\d{9}\b")
_RE_PHONE = re.compile(r"(?:\+972|0)[\-\s]?\d{1,2}[\-\s]?\d{3}[\-\s]?\d{4}")
_RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_RE_DOB = re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b")
_RE_WHITESPACE = re.compile(r"\s+")


# ══════════════════════════════════════════════════════════════════════
#  EMBEDDER PROTOCOL
# ══════════════════════════════════════════════════════════════════════


@runtime_checkable
class Embedder(Protocol):
    """Anything that can produce embedding vectors from text."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# ══════════════════════════════════════════════════════════════════════
#  MONGODB SINGLETON CLIENT
# ══════════════════════════════════════════════════════════════════════

_mongo_client: motor.motor_asyncio.AsyncIOMotorClient | None = None


def _get_mongo_client() -> motor.motor_asyncio.AsyncIOMotorClient:
    """Return (or create) the module-level async MongoDB client."""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGO_URI.get_secret_value())
        logger.info("MongoDB async client created (singleton).")
    return _mongo_client


# ══════════════════════════════════════════════════════════════════════
#  CLINICAL SAFETY SCANNER
# ══════════════════════════════════════════════════════════════════════


class ClinicalSafetyScanner:
    """
    Fuzzy-logic urgency scoring engine for clinical triage.

    Algorithm
    ---------
    1. Scan user text against three keyword tiers (emergency,
       urgent, moderate), each with calibrated base weights.
    2. Detect intensifier words that multiply nearby keyword weights.
    3. Compute final score: ``max(matched_weights) + multi_match_boost``,
       where ``multi_match_boost = 0.05 × (additional_matches)``.
    4. Apply intensifier multiplier to the highest-weight keyword.
    5. Clamp to [0.0, 1.0] and classify:
       - **RED** (≥ 0.70): Medical emergency — redirect immediately.
       - **ORANGE** (≥ 0.40): Needs follow-up — warn then assist.
       - **GREEN** (< 0.40): Normal query — answer from protocols.
    """

    __slots__ = ("_all_keywords",)

    def __init__(self) -> None:
        self._all_keywords: dict[str, float] = {}
        self._all_keywords.update(EMERGENCY_KEYWORDS)
        self._all_keywords.update(URGENT_KEYWORDS)
        self._all_keywords.update(MODERATE_KEYWORDS)


    def scan(self, text: str) -> UrgencyAssessment:
        """
        Scan text and return a full urgency assessment.

        Returns
        -------
        UrgencyAssessment
            ``{"score": float, "flag": str, "matched_keywords": list[str], "recommendation": str}``
        """
        text_lower = text.lower()

        matched: list[tuple[str, float]] = []
        for keyword, weight in self._all_keywords.items():
            if keyword.lower() in text_lower:
                matched.append((keyword, weight))

        if not matched:
            return {"score": 0.0, "flag": "GREEN", "matched_keywords": [], "recommendation": "No clinical urgency detected."}

        matched.sort(key=lambda x: x[1], reverse=True)
        max_weight = matched[0][1]

        intensifier_multiplier = self._detect_intensifier(text_lower)

        boosted_max = min(max_weight * intensifier_multiplier, 1.0)
        multi_boost = 0.05 * (len(matched) - 1)

        final_score = min(boosted_max + multi_boost, 1.0)
        flag = self._classify(final_score)
        matched_keywords = [kw for kw, _ in matched]

        recommendation = self._get_recommendation(flag)

        logger.info("[SAFETY] Score=%.2f, Flag=%s, Keywords=%s, Intensifier=%.1fx", final_score, flag, matched_keywords, intensifier_multiplier)
        return {"score": final_score, "flag": flag, "matched_keywords": matched_keywords, "recommendation": recommendation}


    @staticmethod
    def _detect_intensifier(text_lower: str) -> float:
        """Return the highest intensifier multiplier found in text, or 1.0."""
        max_mult = 1.0
        for word, multiplier in INTENSIFIERS.items():
            if word.lower() in text_lower:
                max_mult = max(max_mult, multiplier)
        return max_mult


    @staticmethod
    def _classify(score: float) -> str:
        """Classify urgency score into flag colour."""
        if score >= 0.70:
            return "RED"
        if score >= 0.40:
            return "ORANGE"
        return "GREEN"


    @staticmethod
    def _get_recommendation(flag: str) -> str:
        """Return a log-friendly recommendation string per flag."""
        if flag == "RED":
            return "EMERGENCY — redirect to ER/MDA immediately."
        if flag == "ORANGE":
            return "FOLLOW-UP — advise medical contact, then provide info."
        return "NORMAL — provide protocol information."


# ══════════════════════════════════════════════════════════════════════
#  MEMORY MANAGER
# ══════════════════════════════════════════════════════════════════════


class MemoryManager:
    """
    Token-aware memory manager with auto-summarization and PII scrubbing.

    Uses Gemini's ``count_tokens`` API to track conversation size.
    When a session exceeds ``TOKEN_LIMIT``, the manager:
        1. Concatenates all messages into a conversation transcript.
        2. Calls Gemini to produce a concise clinical summary.
        3. Scrubs PII from the summary (keeps Name + Location only).
        4. Replaces MongoDB history with the sanitized summary.
    """

    __slots__ = ("_genai_client", "_llm")

    def __init__(self) -> None:
        from google import genai

        self._genai_client = genai.Client(api_key=settings.GOOGLE_API_KEY.get_secret_value())
        self._llm = self._init_summarizer()


    @staticmethod
    def _init_summarizer() -> object:
        """Create a low-temperature LLM for deterministic summarization."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=settings.LLM_MODEL, temperature=0.1, google_api_key=settings.GOOGLE_API_KEY.get_secret_value())


    def count_tokens(self, text: str) -> int:
        """
        Count tokens using Gemini's counting API.

        Falls back to a heuristic estimate (1 token ≈ 3 chars for
        Hebrew) if the API call fails.
        """
        try:
            response = self._genai_client.models.count_tokens(model=settings.LLM_MODEL, contents=text)
            return response.total_tokens
        except Exception:
            logger.warning("[MEMORY] Token counting API failed — using heuristic estimate.")
            return len(text) // 3


    def count_messages_tokens(self, messages: list[ChatMessage]) -> int:
        """Count total tokens across all messages in a session."""
        full_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return self.count_tokens(full_text)


    async def check_and_summarize(self, session_id: str, messages: list[ChatMessage], session_manager: object) -> list[ChatMessage]:
        """
        Check if messages exceed token limit; summarize if needed.

        Parameters
        ----------
        session_id
            The session to potentially compress.
        messages
            Current message list from MongoDB.
        session_manager
            ``MongoSessionManager`` instance for history replacement.

        Returns
        -------
        list[ChatMessage]
            Either the original messages (if under limit) or a
            single-message list containing the sanitized summary.
        """
        if not messages:
            return messages

        token_count = self.count_messages_tokens(messages)
        logger.info("[MEMORY] Session '%s': %d tokens (limit=%d)", session_id, token_count, settings.TOKEN_LIMIT)

        if token_count < settings.TOKEN_LIMIT:
            return messages

        logger.warning("[MEMORY] Token limit exceeded — summarizing session '%s'.", session_id)

        summary = await self._summarize(messages)
        clean_summary = self._scrub_pii(summary)

        summary_message: ChatMessage = {"role": "assistant", "content": f"[📋 SESSION SUMMARY]\n{clean_summary}"}
        await session_manager.replace_history(session_id, [summary_message])  # type: ignore[union-attr]

        logger.info("[MEMORY] Session '%s' compressed: %d msgs → 1 summary (%d tokens → ~%d tokens).", session_id, len(messages), token_count, self.count_tokens(clean_summary))
        return [summary_message]


    async def _summarize(self, messages: list[ChatMessage]) -> str:
        """Call Gemini to produce a concise conversation summary."""
        conversation_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        prompt = SUMMARIZATION_PROMPT.format(conversation=conversation_text)

        try:
            from langchain_core.messages import HumanMessage

            response = await self._llm.ainvoke([HumanMessage(content=prompt)])  # type: ignore[union-attr]
            return response.content if hasattr(response, "content") else str(response)
        except Exception:
            logger.exception("[MEMORY] Summarization LLM call failed — using truncation fallback.")
            last_few = messages[-4:]
            return "\n".join(f"{m['role']}: {m['content']}" for m in last_few)


    @staticmethod
    def _scrub_pii(text: str) -> str:
        """
        Remove PII from text, preserving only Name and Location.

        Scrubs:
            - Israeli ID numbers (9 digits)
            - Phone numbers (Israeli format)
            - Email addresses
            - Date-like patterns (potential DOBs)
        """
        text = _RE_ISRAELI_ID.sub("[מזהה מוסתר]", text)
        text = _RE_PHONE.sub("[טלפון מוסתר]", text)
        text = _RE_EMAIL.sub("[אימייל מוסתר]", text)
        text = _RE_DOB.sub("[תאריך מוסתר]", text)
        return text


# ══════════════════════════════════════════════════════════════════════
#  MONGO SESSION MANAGER
# ══════════════════════════════════════════════════════════════════════


class MongoSessionManager:
    """
    Async chat-history store backed by MongoDB via ``motor``.

    Session isolation is enforced — every query filters by
    ``session_id``, so one user can never access another's history.

    Collection schema (``sessions``)::

        {
            "session_id": str,
            "messages": [{"role": str, "content": str}, ...],
            "created_at": datetime,
            "updated_at": datetime
        }
    """

    __slots__ = ("_collection",)

    def __init__(self, collection_name: str = "sessions") -> None:
        client = _get_mongo_client()
        db = client[settings.MONGO_DB_NAME]
        self._collection = db[collection_name]


    async def get_history(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        """Retrieve the last *limit* messages for a session."""
        limit = limit or settings.SESSION_HISTORY_LIMIT
        doc = await self._collection.find_one({"session_id": session_id}, {"messages": {"$slice": -limit}})
        if doc is None:
            return []
        return doc.get("messages", [])


    async def get_full_history(self, session_id: str) -> list[ChatMessage]:
        """Retrieve ALL messages for a session (used by MemoryManager)."""
        doc = await self._collection.find_one({"session_id": session_id}, {"messages": 1})
        if doc is None:
            return []
        return doc.get("messages", [])


    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message (upsert on first write)."""
        now = datetime.now(timezone.utc)
        message: ChatMessage = {"role": role, "content": content}
        await self._collection.update_one({"session_id": session_id}, {"$push": {"messages": message}, "$set": {"updated_at": now}, "$setOnInsert": {"created_at": now}}, upsert=True)


    async def replace_history(self, session_id: str, messages: list[ChatMessage]) -> None:
        """Replace the entire message history (used after summarization)."""
        now = datetime.now(timezone.utc)
        await self._collection.update_one({"session_id": session_id}, {"$set": {"messages": messages, "updated_at": now}})
        logger.info("[SESSION] History replaced for session '%s' (%d messages).", session_id, len(messages))


    async def is_new_session(self, session_id: str) -> bool:
        """Return True if this session has no messages yet."""
        doc = await self._collection.find_one({"session_id": session_id}, {"_id": 1})
        return doc is None


    async def clear_session(self, session_id: str) -> bool:
        """Delete a session entirely.  Returns True if removed."""
        result = await self._collection.delete_one({"session_id": session_id})
        return result.deleted_count > 0


    async def list_sessions(self) -> list[dict[str, str | datetime]]:
        """Return all session IDs with timestamps, newest first."""
        cursor = self._collection.find({}, {"session_id": 1, "created_at": 1, "updated_at": 1, "_id": 0}).sort("updated_at", -1)
        return await cursor.to_list(length=100)


# ══════════════════════════════════════════════════════════════════════
#  RAG MANAGER
# ══════════════════════════════════════════════════════════════════════


class RAGManager:
    """
    Orchestrates the full RAG pipeline: triage → memory → retrieve → generate.

    Parameters
    ----------
    vector_store
        An initialised ``GaliVectorStore`` for vector search.
    embedder
        An ``Embedder``-compatible object for query embedding.
    session_manager
        Optional custom ``MongoSessionManager``.
    memory_manager
        Optional custom ``MemoryManager``.
    safety_scanner
        Optional custom ``ClinicalSafetyScanner``.
    """

    __slots__ = ("_store", "_embedder", "_session", "_memory", "_scanner", "_llm")

    def __init__(self, vector_store: object, embedder: Embedder, session_manager: MongoSessionManager | None = None, memory_manager: MemoryManager | None = None, safety_scanner: ClinicalSafetyScanner | None = None) -> None:
        self._store = vector_store
        self._embedder = embedder
        self._session = session_manager or MongoSessionManager()
        self._memory = memory_manager or MemoryManager()
        self._scanner = safety_scanner or ClinicalSafetyScanner()
        self._llm = self._init_llm()


    @staticmethod
    def _init_llm() -> object:
        """Initialise the Gemini LLM via LangChain."""
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL, temperature=settings.LLM_TEMPERATURE, google_api_key=settings.GOOGLE_API_KEY.get_secret_value())
        logger.info("LLM initialised: %s (temperature=%.1f)", settings.LLM_MODEL, settings.LLM_TEMPERATURE)
        return llm


    async def generate_response(self, session_id: str, user_query: str) -> str:
        """
        Full RAG pipeline with clinical safety and smart memory.

        Steps:
            1.  New session check → inject greeting.
            2.  Clinical safety scan → urgency assessment.
            3.  RED flag → immediate redirect (no LLM call).
            4.  Memory management → auto-summarize if over token limit.
            5.  Fetch windowed history from MongoDB.
            6.  Refine query for vector search.
            7.  Retrieve context (over-fetch 2×).
            8.  Re-rank + filter for quality.
            9.  Build augmented prompt with safety data.
            10. Call Gemini (async).
            11. Append mandatory disclaimer.
            12. Handle ORANGE flag prefix.
            13. Save messages to MongoDB.
            14. Return answer.
        """
        t_start = time.perf_counter()

        # ── 1. New session check ──────────────────────────────────────
        is_new = await self._session.is_new_session(session_id)
        if is_new:
            logger.info("[RAG] New session: %s", session_id)
            await self._session.add_message(session_id, "assistant", NEW_SESSION_GREETING)

        # ── 2. Clinical safety scan ───────────────────────────────────
        assessment = self._scanner.scan(user_query)
        urgency_score: float = assessment["score"]  # type: ignore[assignment]
        flag: str = assessment["flag"]  # type: ignore[assignment]
        matched_keywords: list[str] = assessment["matched_keywords"]  # type: ignore[assignment]

        # ── 3. RED flag → immediate redirect ──────────────────────────
        if flag == "RED":
            logger.warning("[RAG] 🚨 RED FLAG — redirecting to emergency services.")
            await self._session.add_message(session_id, "user", user_query)
            await self._session.add_message(session_id, "assistant", RED_FLAG_RESPONSE)
            return RED_FLAG_RESPONSE

        # ── 4. Memory management (token-based summarization) ──────────
        t_memory = time.perf_counter()
        full_history = await self._session.get_full_history(session_id)
        full_history = await self._memory.check_and_summarize(session_id, full_history, self._session)
        memory_ms = (time.perf_counter() - t_memory) * 1000
        logger.info("[RAG] Memory check completed in %.1fms", memory_ms)

        # ── 5. Fetch windowed history ─────────────────────────────────
        t_history = time.perf_counter()
        history = await self._session.get_history(session_id)
        history_ms = (time.perf_counter() - t_history) * 1000
        logger.info("[RAG] History fetched: %d message(s) in %.1fms", len(history), history_ms)

        # ── 6. Refine query ───────────────────────────────────────────
        refined_query = self._refine_query(user_query, history)
        logger.info("[RAG] Query refined: '%s' → '%s'", user_query[:50], refined_query[:50])

        # ── 7. Retrieve context (over-fetch 2×) ──────────────────────
        t_search = time.perf_counter()
        raw_results: list[SearchResult] = self._store.search(query_text=refined_query, limit=settings.SEARCH_RESULTS_LIMIT * 2)  # type: ignore[attr-defined]
        search_ms_raw = (time.perf_counter() - t_search) * 1000

        # ── 8. Re-rank + filter ───────────────────────────────────────
        ranked_results = self._rerank_results(raw_results, user_query)
        search_ms = (time.perf_counter() - t_search) * 1000
        logger.info("[RAG] Search: %d raw → %d ranked in %.1fms", len(raw_results), len(ranked_results), search_ms)

        # ── 9. Build augmented prompt ─────────────────────────────────
        context_str = self._format_context(ranked_results)
        history_str = self._format_history(history)

        if not ranked_results:
            logger.warning("[RAG] No relevant context after re-ranking.")
            response = NO_CONTEXT_RESPONSE_HE
            if flag == "ORANGE":
                response = ORANGE_FLAG_RESPONSE + "\n\n" + response
            await self._session.add_message(session_id, "user", user_query)
            await self._session.add_message(session_id, "assistant", response)
            return response

        prompt = RAG_PROMPT_TEMPLATE.format(context=context_str, history=history_str, question=user_query, urgency_score=urgency_score, flag=flag, matched_keywords=", ".join(matched_keywords) if matched_keywords else "None")

        # ── 10. Call Gemini (async) ───────────────────────────────────
        t_llm = time.perf_counter()
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
            response_obj = await self._llm.ainvoke(messages)  # type: ignore[union-attr]
            answer = response_obj.content if hasattr(response_obj, "content") else str(response_obj)
        except Exception:
            logger.exception("[RAG] LLM call failed.")
            answer = "אירעה שגיאה בעת עיבוד הבקשה. אנא נסי שוב."

        llm_ms = (time.perf_counter() - t_llm) * 1000
        logger.info("[RAG] LLM response: %.1fms (%d chars)", llm_ms, len(answer))

        # ── 11. Append disclaimer if missing ──────────────────────────
        if DISCLAIMER not in answer:
            answer = answer.rstrip() + "\n\n" + DISCLAIMER

        # ── 12. ORANGE flag prefix ────────────────────────────────────
        if flag == "ORANGE":
            answer = ORANGE_FLAG_RESPONSE + "\n\n" + answer
        elif flag == "GREEN":
            if not answer.startswith(GREEN_FLAG_PREFIX):
                answer = GREEN_FLAG_PREFIX + answer

        # ── 13. Save to MongoDB ───────────────────────────────────────
        await self._session.add_message(session_id, "user", user_query)
        await self._session.add_message(session_id, "assistant", answer)

        # ── 14. Return ────────────────────────────────────────────────
        total_ms = (time.perf_counter() - t_start) * 1000
        logger.info("[RAG] Pipeline total: %.1fms (memory=%.1f, history=%.1f, search=%.1f, llm=%.1f)", total_ms, memory_ms, history_ms, search_ms, llm_ms)

        return answer

    # ══════════════════════════════════════════════════════════════════
    #  QUERY REFINEMENT
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _refine_query(query: str, history: list[ChatMessage]) -> str:
        """
        Optimise the user query for vector search accuracy.

        Steps:
            1. Normalise whitespace.
            2. Strip Hebrew filler/stop words.
            3. If the query is a follow-up (< 4 meaningful tokens),
               inject the last user question for context expansion.
            4. Safety fallback: return original if stripping leaves < 2 tokens.
        """
        normalised = _RE_WHITESPACE.sub(" ", query).strip()
        tokens = normalised.split()
        filtered = [t for t in tokens if t not in _HEBREW_FILLERS and len(t) > 1]

        if len(filtered) < 2:
            return normalised

        if len(filtered) < 4 and history:
            last_user_msgs = [m["content"] for m in history if m["role"] == "user"]
            if last_user_msgs:
                prev_tokens = last_user_msgs[-1].split()
                prev_filtered = [t for t in prev_tokens if t not in _HEBREW_FILLERS and len(t) > 1]
                filtered = prev_filtered + filtered

        return " ".join(filtered)

    # ══════════════════════════════════════════════════════════════════
    #  RE-RANKING & FILTERING
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _rerank_results(results: list[SearchResult], original_query: str) -> list[SearchResult]:
        """
        Re-rank and filter vector search results for quality.

        Strategy:
            1. Distance filter — discard above ``RELEVANCE_THRESHOLD``.
            2. Keyword boost — query-token overlap adds score bonus.
            3. Source diversity — penalty for repeated source files.
            4. Top-K trim — return at most ``SEARCH_RESULTS_LIMIT``.
        """
        if not results:
            return []

        threshold = settings.RELEVANCE_THRESHOLD
        query_tokens = set(original_query.lower().split())

        scored: list[tuple[float, SearchResult]] = []
        seen_sources: dict[str, int] = {}

        for result in results:
            distance = float(result.get("_distance", 999.0))
            if distance > threshold:
                continue

            text = str(result.get("text", "")).lower()
            text_tokens = set(text.split())
            overlap = len(query_tokens & text_tokens)
            keyword_boost = overlap * 0.05

            source = str(result.get("source_file", "unknown"))
            source_count = seen_sources.get(source, 0)
            diversity_penalty = source_count * 0.10

            final_score = distance - keyword_boost + diversity_penalty
            scored.append((final_score, result))
            seen_sources[source] = source_count + 1

        scored.sort(key=lambda x: x[0])

        limit = settings.SEARCH_RESULTS_LIMIT
        top_results = [result for _, result in scored[:limit]]
        logger.debug("[RERANK] %d/%d passed threshold (%.2f), returning top %d.", len(scored), len(results), threshold, len(top_results))
        return top_results

    # ══════════════════════════════════════════════════════════════════
    #  PROMPT FORMATTING
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        """Format vector results into a numbered context block with source citations."""
        if not results:
            return "(No relevant protocols found.)"

        blocks: list[str] = []
        for i, result in enumerate(results, 1):
            source = result.get("source_file", "unknown")
            distance = result.get("_distance", "N/A")
            text = result.get("text", "")
            blocks.append(f"[{i}] Source: {source} (relevance: {distance})\n{text}")

        return "\n\n".join(blocks)


    @staticmethod
    def _format_history(messages: list[ChatMessage]) -> str:
        """Format chat history into a readable conversation block."""
        if not messages:
            return "(No previous conversation.)"

        lines: list[str] = []
        for msg in messages:
            role_label = "User" if msg["role"] == "user" else "Gali"
            lines.append(f"{role_label}: {msg['content']}")

        return "\n".join(lines)
