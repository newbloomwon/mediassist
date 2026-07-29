"""
security.py — Input validation, session-based authentication, and rate limiting.

Drop this file into the root of your mediassist repo.
"""
import re
import time
from collections import defaultdict, deque
from fastapi import HTTPException, Request

# ── Input limits ──────────────────────────────────────────────────────────────

MAX_MESSAGE_LENGTH = 2000

# ── Prompt-injection detection ────────────────────────────────────────────────
#
# These patterns catch the most common prompt-injection and jailbreak attempts.
# They are not exhaustive — treat them as a first-line defence, not a complete
# solution.  Pair them with a hardened system prompt and tool-level authorisation
# checks (see agent.py).

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|system)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above|system)\s+instructions",
    r"forget\s+(all\s+)?(previous|prior|above|system)\s+instructions",
    r"override\s+(the\s+)?(system|previous|your)\s+(prompt|instructions)",
    r"new\s+system\s+prompt",
    r"<\s*/?system\s*>",           # XML-style system-tag injection
    r"\[system\]",                  # bracket-style injection
    r"\[INST\]",                    # Llama-style instruction token
    r"jailbreak",
    r"\bdan\s+mode\b",              # "Do Anything Now"
    r"do\s+anything\s+now",
    r"pretend\s+(you\s+are|to\s+be)\s+(?!mediassist)",
    r"you\s+are\s+now\s+(?!mediassist)",
    r"act\s+as\s+(a\s+)?(different|new|another|unrestricted|evil|hacker)",
    r"developer\s+mode",
    r"sudo\s+mode",
    r"god\s+mode",
    r"enable\s+(unrestricted|unlimited|admin|root|debug)\s+mode",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def validate_message(message: str) -> str:
    """
    Validates a user chat message.

    - Rejects empty messages
    - Enforces MAX_MESSAGE_LENGTH
    - Blocks common prompt-injection patterns

    Returns the stripped message on success, raises HTTPException on failure.
    """
    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Message too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed.",
        )

    for pattern in _COMPILED_PATTERNS:
        if pattern.search(message):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Your message could not be processed. "
                    "Please rephrase your question and try again."
                ),
            )

    return message.strip()


# ── Session-based authentication ──────────────────────────────────────────────

def get_session_patient_id(request: Request):
    """Returns the authenticated patient ID from the signed session cookie, or None."""
    return request.session.get("patient_id")


def require_auth(request: Request) -> int:
    """
    Returns the authenticated patient ID.
    Raises HTTP 401 if the session is missing or expired.
    """
    patient_id = get_session_patient_id(request)
    if patient_id is None:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Please log in first.",
        )
    return int(patient_id)


# ── Rate limiting ─────────────────────────────────────────────────────────────
#
# Simple in-memory sliding-window rate limiter.
# Limits each patient session to RATE_LIMIT_REQUESTS chat messages per
# RATE_LIMIT_WINDOW seconds.  Resets automatically as the window slides.
#
# Note: because this is in-memory, limits reset if the server restarts.
# For a production system, use Redis or a similar shared store.

RATE_LIMIT_REQUESTS = 20   # max messages per window
RATE_LIMIT_WINDOW   = 60   # window size in seconds

_request_log: dict = defaultdict(deque)  # patient_id → deque of timestamps


def check_rate_limit(patient_id: int) -> None:
    """
    Enforces the per-patient rate limit.
    Raises HTTP 429 if the patient has exceeded RATE_LIMIT_REQUESTS in the
    last RATE_LIMIT_WINDOW seconds.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    timestamps = _request_log[patient_id]

    # Drop timestamps that have fallen outside the window
    while timestamps and timestamps[0] < window_start:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Too many messages. You can send up to {RATE_LIMIT_REQUESTS} "
                f"messages per minute. Please wait a moment and try again."
            ),
        )

    timestamps.append(now)
