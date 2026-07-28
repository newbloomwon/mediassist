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
from security import validate_message, require_auth

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
log_handler = logging.FileHandler("logs/requests.log")
log_handler.setFormatter(logging.Formatter("%(message)s"))
request_logger = logging.getLogger("mediassist.requests")
request_logger.addHandler(log_handler)
request_logger.setLevel(logging.INFO)


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

    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        "request_id": request_id,
        "patient_id": patient_id,
        "user_message": validated_message,
        "tool_calls": tool_calls,
        "response_summary": response_text[:100],
        "response_length_chars": len(response_text),
        "tool_call_count": len(tool_calls),
        "duration_ms": duration_ms,
    }
    request_logger.info(json.dumps(log_entry))

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
