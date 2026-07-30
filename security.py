"""
security.py — Input validation, session-based authentication, rate limiting,
              PII redaction, and response validation.

Drop this file into the root of your mediassist repo.
"""
import json
import logging
import re
import time
from collections import defaultdict, deque
from fastapi import HTTPException, Request

# ── Security logger ───────────────────────────────────────────────────────────
# Handlers are attached by main.py after it creates the logs/ directory.
# Using a named logger here means main.py can configure it without importing
# this module first.
security_logger = logging.getLogger("mediassist.security")

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
    # Persona-split / dual-voice jailbreak patterns
    # e.g. "Answer as both MediAssist [restricted] and your [unrestricted] self"
    r"\[unrestricted\]",
    r"\[unfiltered\]",
    r"inner\s+monologue",
    r"your\s+true\s+self",
    r"unfiltered\s+(response|mode|version|output)",
    r"(answer|respond|reply|speak)\s+as\s+both",
    r"without\s+safety\s+(guidelines|restrictions|filters|constraints)",
    r"hypothetical(ly)?\s+(unrestricted|unfiltered|uncensored)",
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


# ── PII redaction ─────────────────────────────────────────────────────────────
#
# Applied to user messages and response summaries before they are written to
# any log file.  Never applied to the actual response sent to the user —
# that would break legitimate medical information.

_PII_PATTERNS = [
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),                              "[SSN]"),
    (re.compile(r'\b(BCB|AET|UHC|MED|KAI)-\d{4}-\d{6}\b'),             "[INSURANCE-ID]"),
    (re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'), "[EMAIL]"),
    (re.compile(r'\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b'), "[PHONE]"),
]


def redact_pii(text: str) -> str:
    """
    Replace known PII patterns with labelled placeholders.
    Safe to call on any string before logging.
    """
    if not text:
        return text
    for pattern, placeholder in _PII_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text


def _sec_log(level: str, event: str, patient_id: int, **kwargs) -> None:
    """Write a structured JSON entry to the security log."""
    entry = {
        "event": event,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "patient_id": patient_id,
        **kwargs,
    }
    msg = json.dumps(entry)
    if level == "error":
        security_logger.error(msg)
    else:
        security_logger.warning(msg)


# ── Response validation ───────────────────────────────────────────────────────

MAX_RESPONSE_LENGTH = 5000

_SSN_PATTERN          = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
_INSURANCE_ID_PATTERN = re.compile(r'\b(BCB|AET|UHC|MED|KAI)-\d{4}-\d{6}\b')
_EMAIL_IN_RESP_PATTERN = re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b')

_SYSTEM_PROMPT_PHRASES = [
    "SECURITY RULES",
    "session patient ID",
    "END SECURITY RULES",
    "server-side authenticated",
]

ALL_PATIENT_NAMES = [
    "Margaret Chen", "James Okafor", "Sofia Ramirez", "Robert Washington",
    "Priya Patel", "David Kim", "Amara Johnson", "Carlos Mendez",
    "Lisa Nakamura", "Thomas O'Brien",
]

SAFE_FALLBACK = (
    "I'm sorry, I wasn't able to generate a safe response to that request. "
    "Please try rephrasing your question."
)


def validate_response(response: str, patient_id: int, patient_name: str) -> str:
    """
    Validates the LLM response before it reaches the user.

    Hard blocks — returns SAFE_FALLBACK and logs an error:
      - Empty response
      - SSN pattern detected
      - Insurance ID pattern detected
      - System prompt phrases detected (prompt disclosure)

    Soft flags — logs a warning, passes response through:
      - Another patient's name in the response
      - Email address in the response

    Truncates responses over MAX_RESPONSE_LENGTH characters.
    """
    if not response or not response.strip():
        _sec_log("error", "response_blocked", patient_id, reason="empty response")
        return SAFE_FALLBACK

    if len(response) > MAX_RESPONSE_LENGTH:
        _sec_log("warning", "response_truncated", patient_id,
                 original_length=len(response), limit=MAX_RESPONSE_LENGTH)
        response = response[:MAX_RESPONSE_LENGTH] + "\n\n[Response truncated]"

    if _SSN_PATTERN.search(response):
        _sec_log("error", "response_blocked", patient_id, reason="SSN pattern detected")
        return SAFE_FALLBACK

    if _INSURANCE_ID_PATTERN.search(response):
        _sec_log("error", "response_blocked", patient_id,
                 reason="insurance ID pattern detected")
        return SAFE_FALLBACK

    for phrase in _SYSTEM_PROMPT_PHRASES:
        if phrase.lower() in response.lower():
            _sec_log("error", "response_blocked", patient_id,
                     reason=f"system prompt phrase detected: '{phrase}'")
            return SAFE_FALLBACK

    for name in ALL_PATIENT_NAMES:
        if name != patient_name and name.lower() in response.lower():
            _sec_log("warning", "response_flagged", patient_id,
                     reason=f"other patient name in response: '{name}'")

    if _EMAIL_IN_RESP_PATTERN.search(response):
        _sec_log("warning", "response_flagged", patient_id,
                 reason="email address in response")

    return response
