
"""FastAPI backend — BGV System Approach 3"""
import json, os, uuid, httpx
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bgv_graph import run_initial_verification, resume_with_feedback, submit_decision

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

app = FastAPI(title="BGV System v3.0 — Production Architecture")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ─── Request / Response models ────────────────────────────────────────────────

class VerifyRequest(BaseModel):
    aadhaar: str
    hr_name: str
    hr_dob: str
    hr_address: str

class RerunRequest(BaseModel):
    """
    Approach 3 change: no session_id needed — thread_id IS the session.
    The graph state is stored in the checkpointer keyed by thread_id.
    FastAPI no longer maintains a sessions{} dict.
    """
    thread_id: str                  # returned by /verify, used to resume graph
    rerun_tool1: bool = False
    rerun_tool2: bool = False
    rerun_tool3: bool = False
    feedback: Optional[str]    = None
    hr_name: Optional[str]     = None
    hr_dob: Optional[str]      = None
    hr_address: Optional[str]  = None

class DecisionRequest(BaseModel):
    thread_id: str
    decision: str          # "accepted" or "rejected"
    remarks: Optional[str] = None   # optional HR remarks

class SuggestRequest(BaseModel):
    feedback: str
    openai_key: Optional[str] = None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("templates/index.html") as f:
        return f.read()


@app.post("/verify")
async def verify_candidate(req: VerifyRequest):
    """
    First run.
    - Generates a thread_id (UUID) which the checkpointer uses as the key
    - Graph fires all 3 tools in PARALLEL, then PAUSES at human_feedback_node
    - Returns partial report + thread_id for HR to use in /rerun
    """
    thread_id = str(uuid.uuid4())
    report = await run_initial_verification(
        aadhaar=req.aadhaar,
        hr_name=req.hr_name,
        hr_dob=req.hr_dob,
        hr_address=req.hr_address,
        thread_id=thread_id,
    )
    # thread_id goes to the client — used to resume the paused graph
    return JSONResponse({"thread_id": thread_id, "report": report})


@app.post("/rerun")
async def rerun_tools(req: RerunRequest):
    """
    Re-run path — RESUMES the paused graph, does not re-invoke from scratch.

    Approach 2: FastAPI held BGVState in sessions{}, re-called graph.ainvoke().
    Approach 3: FastAPI calls resume_with_feedback() which does:
                  1. graph.aupdate_state() — injects HumanFeedback into paused graph
                  2. graph.ainvoke(None, thread_id) — resumes from interrupt point
                The checkpointer restores the full prior state automatically.
    """
    if not any([req.rerun_tool1, req.rerun_tool2, req.rerun_tool3]):
        raise HTTPException(400, "Select at least one tool to re-run.")

    try:
        report = await resume_with_feedback(
            thread_id=req.thread_id,
            feedback_text=req.feedback,
            run_tool1=req.rerun_tool1,
            run_tool2=req.rerun_tool2,
            run_tool3=req.rerun_tool3,
            hr_name=req.hr_name,
            hr_dob=req.hr_dob,
            hr_address=req.hr_address,
        )
    except Exception as e:
        raise HTTPException(404, f"Thread not found or expired: {e}")

    return JSONResponse({"thread_id": req.thread_id, "report": report})


@app.post("/decide")
async def decide_candidate(req: DecisionRequest):
    """
    HR's final Accept / Reject decision.

    This is the clean exit point for the graph — it resumes the paused
    graph, writes the hr_decision into BGVState, logs it in the audit
    trail, and runs generate_report to produce the final closed report.

    The graph then reaches __end__ and is fully complete.

    Without this endpoint, a graph that needed no re-run would stay
    suspended at human_feedback_node indefinitely — the decision button
    gives it a proper terminal state.
    """
    if req.decision not in ("accepted", "rejected"):
        raise HTTPException(400, "decision must be 'accepted' or 'rejected'")
    try:
        report = await submit_decision(
            thread_id=req.thread_id,
            decision=req.decision,
            remarks=req.remarks,
        )
    except Exception as e:
        raise HTTPException(404, f"Thread not found or already closed: {e}")
    return JSONResponse({"thread_id": req.thread_id, "report": report})


@app.post("/suggest-rerun")
async def suggest_rerun(req: SuggestRequest):
    """LLM parses HR free-text → returns which tool flags to set True."""
    key = req.openai_key or OPENAI_API_KEY
    if not key:
        raise HTTPException(400, "API key not configured.")

    system_prompt = """You are an assistant for an HR Background Verification System.
There are 3 verification tools:
- Tool 1: Personal Identity Verification (name, DOB, address verified against Aadhaar DB)
- Tool 2: Criminal Background Check (criminal records, Interpol, sex offender registry)
- Tool 3: Financial & Fraud Check (credit score, sanctions, bankruptcy, PEP)

Given HR feedback, decide which tools need to be re-run.
Respond ONLY with valid JSON:
{"tool1": true/false, "tool2": true/false, "tool3": true/false, "reasoning": "one sentence"}"""

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.us.inc/usf/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "ultrasafe/usf-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f'HR feedback: "{req.feedback}"'},
                ],
                "max_tokens": 200, "temperature": 0,
            }
        )

    if resp.status_code != 200:
        raise HTTPException(502, f"AI API error: {resp.text}")

    raw = resp.json()["choices"][0]["message"]["content"].strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        raise HTTPException(500, f"Could not parse AI response: {raw}")


@app.get("/demo-candidates")
async def demo_candidates():
    from candidate_db import DEMO_CANDIDATES
    return {"candidates": DEMO_CANDIDATES}


@app.get("/session/{thread_id}")
async def get_session(thread_id: str):
    """
    Retrieve current report for a thread.
    In Approach 3 we could query the checkpointer directly,
    but returning the last report from /verify or /rerun is simpler
    for the prototype — the client stores it anyway.
    """
    return {"message": "Use thread_id from /verify response to call /rerun"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)