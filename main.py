import json
import logging
import os
import time
import uuid

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from typing import Optional

import database
from agent import run_agent
from security import validate_message, require_auth, check_rate_limit, redact_pii

app = FastAPI(title="MediAssist")

# ── Session middleware ─────────────────────────────────────────────────────────
# Sessions are signed with SECRET_KEY so clients cannot tamper with the patient_id
# stored in the cookie.  Set this to a long random string in your .env file.
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY environment variable is not set. "
        "Add SECRET_KEY=<random-string> to your .env file and restart."
    )

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    session_cookie="mediassist_session",
    same_site="strict",
    # Set https_only=True in production (Railway uses HTTPS by default).
    # Keep False for local development over http://localhost.
    https_only=os.environ.get("HTTPS_ONLY", "false").lower() == "true",
)

# ── Logging setup ─────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)

_json_fmt = logging.Formatter("%(message)s")

# access.log — one structured JSON line per chat request, no PHI
_access_handler = logging.FileHandler("logs/access.log")
_access_handler.setFormatter(_json_fmt)
access_logger = logging.getLogger("mediassist.access")
access_logger.addHandler(_access_handler)
access_logger.setLevel(logging.INFO)

# security.log — security events from security.py (blocks, flags, rate limits)
_security_handler = logging.FileHandler("logs/security.log")
_security_handler.setFormatter(_json_fmt)
_sec_logger = logging.getLogger("mediassist.security")
_sec_logger.addHandler(_security_handler)
_sec_logger.setLevel(logging.INFO)


# ── Request models ────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    patient_id: int


class ChatRequest(BaseModel):
    # patient_id is intentionally NOT accepted from the client anymore.
    # It is always read from the server-side session.
    message: str
    conversation_history: list = []


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    database.init_db()


# ── Static / index ────────────────────────────────────────────────────────────

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")


# ── Auth endpoints ────────────────────────────────────────────────────────────

@app.post("/login")
def login(req: LoginRequest, request: Request):
    """
    Authenticate a patient and store their ID in a signed session cookie.
    The client never needs to send patient_id again after this.
    """
    patient = database.get_patient_summary(req.patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    request.session["patient_id"] = req.patient_id
    return {"status": "ok", "patient_id": req.patient_id}


@app.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"status": "ok"}


# ── Patient endpoint ──────────────────────────────────────────────────────────

@app.get("/patient/{patient_id}")
def get_patient(patient_id: int, request: Request):
    """
    Return patient summary. Only allows fetching the currently authenticated patient.
    """
    session_patient_id = require_auth(request)
    if patient_id != session_patient_id:
        raise HTTPException(status_code=403, detail="Access denied.")
    summary = database.get_patient_summary(patient_id)
    if not summary:
        return JSONResponse(status_code=404, content={"error": "Patient not found"})
    return summary


# ── Chat endpoint ─────────────────────────────────────────────────────────────

@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    """
    Run the AI agent for the authenticated patient.

    Security improvements vs. original:
    - patient_id comes from the session, not the request body (prevents IDOR)
    - user message is validated for length and prompt-injection patterns
    """
    # Authenticate — raises 401 if no valid session
    patient_id = require_auth(request)

    # Rate limit — raises 429 if the patient is sending too fast
    check_rate_limit(patient_id)

    # Validate and sanitise the incoming message
    validated_message = validate_message(req.message)

    request_id = uuid.uuid4().hex[:8]
    start_time = time.time()

    response_text, tool_calls = run_agent(
        patient_id,
        validated_message,
        req.conversation_history
    )

    duration_ms = int((time.time() - start_time) * 1000)

    # Structured access log — PII redacted, tool outputs stripped
    log_entry = {
        "event": "chat_request",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "request_id": request_id,
        "patient_id": patient_id,
        "message_length": len(validated_message),
        "message_preview": redact_pii(validated_message[:80]),
        "tools_used": [tc["tool_name"] for tc in tool_calls],
        "tool_count": len(tool_calls),
        "response_length": len(response_text),
        "response_preview": redact_pii(response_text[:80]),
        "duration_ms": duration_ms,
        "status": "ok",
    }
    access_logger.info(json.dumps(log_entry))

    return {
        "response": response_text,
        "tool_calls": [tc["tool_name"] for tc in tool_calls],
        "request_id": request_id,
    }


# ── Reset endpoint ────────────────────────────────────────────────────────────

@app.post("/reset")
def reset_memory(request: Request):
    """Clear session memory for the authenticated patient."""
    patient_id = require_auth(request)
    database.clear_memory(patient_id)
    return {"status": "ok", "message": f"Session memory cleared for patient {patient_id}"}


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True, host="0.0.0.0", port=8000)
