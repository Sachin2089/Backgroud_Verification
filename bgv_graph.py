
"""
BGV System — Approach 3: Production Architecture (fixed)
=========================================================

Root cause of previous InvalidUpdateError:
  Three parallel subgraphs each received the full BGVState via Send() and
  each returned updates to shared fields (aadhaar, hr_name, etc.).
  LangGraph sees 3 concurrent writes to the same key → InvalidUpdateError.

Fix:
  Each subgraph gets its OWN isolated TypedDict with ONLY the fields it
  writes (tool1/2/3, audit_trail, errors + internal retry fields).
  Read-only data (aadhaar, hr_name, etc.) is passed in the Send payload
  but SubState only declares the writable channels — so no conflict.

Graph topology:
  __start__
      │
  dispatch_node        ← thin pass-through (registered node)
      │
  route_tools()        ← condition fn, returns list[Send] (NOT a node)
   /    |    \
  T1   T2   T3         ← each subgraph: execute → retry → fail
   \    |    /
     join              ← conditional: first-pass→pause, re-run→report
      │           │
human_feedback  generate_report
(interrupt_before)
      │
  dispatch_node        ← selective re-run
   /    |    \
  ...parallel...
      │
    join → generate_report → END
"""

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from typing import TypedDict, Optional, Annotated
import operator

from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from candidate_db import lookup_by_aadhaar


TOOL_TIMEOUT_SECONDS = 10
MAX_RETRIES          = 2


# ─── State definitions ────────────────────────────────────────────────────────

class ToolState(TypedDict):
    status: str
    output: Optional[dict]
    timestamp: Optional[str]
    execution_time: Optional[float]
    data_hash: Optional[str]
    version: int
    cached: bool
    retry_count: int
    error_message: Optional[str]


class HumanFeedback(TypedDict):
    feedback_text: Optional[str]
    run_tool1: bool
    run_tool2: bool
    run_tool3: bool
    hr_name: Optional[str]
    hr_dob: Optional[str]
    hr_address: Optional[str]
    submitted: bool


class BGVState(TypedDict):
    aadhaar: str
    hr_name: str
    hr_dob: str
    hr_address: str
    tool1: ToolState
    tool2: ToolState
    tool3: ToolState
    run_tool1: bool
    run_tool2: bool
    run_tool3: bool
    human_feedback: HumanFeedback
    hr_decision: Optional[str]   
    report: Optional[dict]
    # operator.add merges lists from parallel branches safely
    audit_trail: Annotated[list[dict], operator.add]
    errors:      Annotated[list[str],  operator.add]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _make_tool_state(output, exec_time, prev_version=0, retry_count=0) -> ToolState:
    return ToolState(
        status="completed", output=output, timestamp=_now(),
        execution_time=exec_time, data_hash=_hash(output),
        version=prev_version + 1, cached=False,
        retry_count=retry_count, error_message=None,
    )

def _failed_tool_state(error, prev_version=0, retry_count=0, timed_out=False) -> ToolState:
    return ToolState(
        status="timeout" if timed_out else "failed",
        output={"error": error}, timestamp=_now(),
        execution_time=0, data_hash=None,
        version=prev_version + 1, cached=False,
        retry_count=retry_count, error_message=error,
    )

def _addr_match(a, b):
    def tokens(s):
        return set(re.sub(r"[,.\-]", " ", s.lower()).split())
    t1, t2 = tokens(a), tokens(b)
    if not t1 or not t2: return False
    if a.lower() in b.lower() or b.lower() in a.lower(): return True
    return len(t1 & t2) / max(len(t1), len(t2)) >= 0.6

def _name_match(a, b):
    ap, bp = a.lower().split(), b.lower().split()
    if not ap or not bp or ap[0] != bp[0]: return False
    return len(set(ap) & set(bp)) / max(len(set(ap)), len(set(bp))) >= 0.8


# ─── Tool input payload (read-only data passed via Send) ─────────────────────
# This is NOT a LangGraph state — it's just a plain dict passed as the
# Send argument. Subgraphs receive it as their initial state input.

class ToolInput(TypedDict):
    aadhaar: str
    hr_name: str
    hr_dob: str
    hr_address: str
    prev_version: int   # current tool version (for incrementing)


# ─── Raw execution functions ──────────────────────────────────────────────────

async def _execute_identity(inp: ToolInput) -> dict:
    start = time.time()
    await asyncio.sleep(1.2)

    record     = lookup_by_aadhaar(inp["aadhaar"])
    db_name    = record["name"]
    db_dob     = record["dob"]
    db_address = record["address"]
    db_expiry  = record.get("id_expiry", "")

    name_ok    = _name_match(inp["hr_name"], db_name)
    dob_ok     = inp["hr_dob"] == db_dob
    address_ok = _addr_match(inp["hr_address"], db_address)

    name_match_status = (
        ("Confirmed" if inp["hr_name"].lower().strip() == db_name.lower().strip() else "Partial")
        if name_ok else "Mismatch"
    )

    prev_addrs  = record.get("prev_addresses", [])
    addr_hist   = address_ok or any(_addr_match(inp["hr_address"], p) for p in prev_addrs)
    addr_status = "Current" if address_ok else ("Previous Address" if addr_hist else "Not Found")

    try:
        id_expired = datetime.strptime(db_expiry, "%Y-%m-%d") < datetime.now()
    except Exception:
        id_expired = False

    mismatches = []
    if not name_ok:    mismatches.append(f"Name mismatch: HR entered '{inp['hr_name']}', DB shows '{db_name}'")
    if not dob_ok:     mismatches.append(f"DOB mismatch: HR entered '{inp['hr_dob']}', DB shows '{db_dob}'")
    if not address_ok: mismatches.append(f"Address mismatch: HR entered '{inp['hr_address']}', DB shows '{db_address}'")
    if id_expired:     mismatches.append(f"Aadhaar expired on {db_expiry}")

    risk_score = (4 if not name_ok else 0) + (3 if not dob_ok else 0) + \
                 (2 if not address_ok else 0) + (3 if id_expired else 0)

    return {
        "output": {
            "db_name": db_name, "db_dob": db_dob, "db_address": db_address,
            "db_prev_addresses": prev_addrs,
            "hr_name": inp["hr_name"], "hr_dob": inp["hr_dob"], "hr_address": inp["hr_address"],
            "name_match_status": name_match_status,
            "name_verified": name_ok, "dob_verified": dob_ok, "address_verified": address_ok,
            "address_history_status": addr_status,
            "address_history_checked": [db_address] + prev_addrs,
            "id_type": record.get("id_type", "Aadhaar"), "id_number": inp["aadhaar"],
            "id_expiry": db_expiry, "id_expired": id_expired,
            "mismatches": mismatches,
            "risk_score": min(10, risk_score), "flagged": len(mismatches) > 0,
            "confidence": max(30, 100 - len(mismatches) * 20),
            "_generated": record.get("_generated", False),
        },
        "exec_time": round(time.time() - start, 3),
    }


async def _execute_criminal(inp: ToolInput) -> dict:
    start = time.time()
    await asyncio.sleep(1.5)

    record = lookup_by_aadhaar(inp["aadhaar"])
    crimes = record.get("criminal_records", [])
    risk_score = (4 if crimes else 0) + (4 if record.get("interpol") else 0) + \
                 (5 if record.get("sex_offender") else 0)

    def enrich(r):
        s = r.get("status", "")
        r["record_type"] = ("Conviction" if s=="Convicted" else
                            "Arrest/Pending" if s=="Under Trial" else "Arrest")
        return r

    enriched    = [enrich(dict(c)) for c in crimes]
    has_pending = any(c.get("status") == "Under Trial" for c in crimes)
    overall     = "Pending" if has_pending else ("Records Found" if crimes else "Clear")

    return {
        "output": {
            "overall_status": overall, "records_count": len(crimes), "records": enriched,
            "sex_offender_registry": record.get("sex_offender", False),
            "interpol_check": record.get("interpol", False),
            "jurisdictions_checked": ["Federal", "State", "County", "Interpol"],
            "risk_score": min(10, risk_score),
            "flagged": bool(crimes or record.get("interpol") or record.get("sex_offender")),
            "_generated": record.get("_generated", False),
        },
        "exec_time": round(time.time() - start, 3),
    }


async def _execute_financial(inp: ToolInput) -> dict:
    start = time.time()
    await asyncio.sleep(1.0)

    record = lookup_by_aadhaar(inp["aadhaar"])
    fi     = record.get("fraud_indicators", "None")
    risk_score = (
        (5 if fi=="High" else 2 if fi=="Low" else 0) +
        (4 if record.get("sanctions") else 0) + (2 if record.get("bankruptcy") else 0) +
        (3 if record.get("pep") else 0) + (2 if record.get("adverse_media") else 0) +
        (1 if record.get("litigation") else 0)
    )

    credit = record.get("credit_score", 650)
    cs = "Excellent" if credit>=750 else "Good" if credit>=700 else "Fair" if credit>=600 else "Poor"

    flags = []
    if record.get("sanctions"):     flags.append("OFAC/UN/EU Sanctions Hit")
    if record.get("pep"):           flags.append("Politically Exposed Person")
    if record.get("adverse_media"): flags.append("Adverse Media Found")
    if record.get("bankruptcy"):    flags.append("Bankruptcy Record")
    if record.get("litigation"):    flags.append("Active Litigation")

    return {
        "output": {
            "credit_score": credit, "credit_status": cs, "fraud_indicators": fi,
            "sanctions_check": record.get("sanctions", False),
            "bankruptcy": record.get("bankruptcy", False), "pep_status": record.get("pep", False),
            "adverse_media": record.get("adverse_media", False),
            "litigation": record.get("litigation", False),
            "flags": flags, "risk_score": min(10, risk_score),
            "compliance_status": "Non-Compliant" if risk_score >= 5 else "Compliant",
            "flagged": len(flags) > 0, "_generated": record.get("_generated", False),
        },
        "exec_time": round(time.time() - start, 3),
    }


# ─── Subgraph factory ─────────────────────────────────────────────────────────
#
# Each subgraph has its OWN isolated state with ONLY the channels it writes:
#   - tool_result  (ToolState output)
#   - audit_trail  (list append)
#   - errors       (list append)
#   - _retry_count / _last_error  (internal only)
#
# The ToolInput (aadhaar, hr_name, etc.) is passed via Send payload and
# stored in tool_input — it's NOT a shared channel so no concurrent conflict.
#
# The parent graph reads the subgraph's output and merges it back into
# BGVState under the correct tool1/2/3 key in the join node.

def _build_tool_subgraph(execute_fn, tool_label: str):
    """
    Builds a subgraph for one tool.
    Subgraph state is fully isolated — no shared channels with other subgraphs.
    """

    class SubState(TypedDict):
        # Read-only input (received via Send payload)
        tool_input: ToolInput
        # Written by this subgraph only
        tool_result: Optional[ToolState]
        audit_trail: Annotated[list[dict], operator.add]
        errors:      Annotated[list[str],  operator.add]
        # Internal retry control
        _retry_count: int
        _last_error: str

    async def execute_node(state: SubState) -> dict:
        inp         = state["tool_input"]
        retry_count = state.get("_retry_count", 0)
        prev_ver    = inp.get("prev_version", 0)
        try:
            result = await asyncio.wait_for(execute_fn(inp), timeout=TOOL_TIMEOUT_SECONDS)
            ts = _make_tool_state(result["output"], result["exec_time"],
                                  prev_ver, retry_count=retry_count)
            audit = {
                "tool": tool_label, "timestamp": ts["timestamp"],
                "version": ts["version"], "hash": ts["data_hash"],
                "execution_time": ts["execution_time"],
                "action": f"executed (retry #{retry_count})" if retry_count else "executed",
            }
            return {"tool_result": ts, "audit_trail": [audit], "_last_error": ""}
        except asyncio.TimeoutError:
            return {"_last_error": f"TIMEOUT after {TOOL_TIMEOUT_SECONDS}s"}
        except Exception as e:
            return {"_last_error": str(e)}

    def should_retry(state: SubState) -> str:
        err = state.get("_last_error", "")
        if not err: return "done"
        if "TIMEOUT" in err or state.get("_retry_count", 0) >= MAX_RETRIES: return "fail"
        return "retry"

    async def retry_node(state: SubState) -> dict:
        count = state.get("_retry_count", 0) + 1
        await asyncio.sleep(0.3 * count)   # exponential back-off
        return {"_retry_count": count}

    async def fail_node(state: SubState) -> dict:
        inp     = state["tool_input"]
        err     = state.get("_last_error", "Unknown error")
        retries = state.get("_retry_count", 0)
        ts = _failed_tool_state(err, prev_version=inp.get("prev_version", 0),
                                retry_count=retries, timed_out="TIMEOUT" in err)
        audit = {
            "tool": tool_label, "timestamp": ts["timestamp"],
            "version": ts["version"], "hash": None, "execution_time": 0,
            "action": f"FAILED after {retries} retries — {err}",
        }
        return {
            "tool_result": ts,
            "audit_trail": [audit],
            "errors": [f"{tool_label}: {err}"],
        }

    sub = StateGraph(SubState)
    sub.add_node("execute", execute_node)
    sub.add_node("retry",   retry_node)
    sub.add_node("fail",    fail_node)
    sub.set_entry_point("execute")
    sub.add_conditional_edges("execute", should_retry,
                              {"done": END, "retry": "retry", "fail": "fail"})
    sub.add_edge("retry", "execute")
    sub.add_edge("fail",  END)
    return sub.compile()


# ─── Compiled subgraphs ───────────────────────────────────────────────────────

_identity_sg  = _build_tool_subgraph(_execute_identity, "Personal Identity")
_criminal_sg  = _build_tool_subgraph(_execute_criminal,  "Criminal Background")
_financial_sg = _build_tool_subgraph(_execute_financial, "Financial & Fraud")


# ─── Parent graph nodes ───────────────────────────────────────────────────────

async def dispatch_node(state: BGVState) -> dict:
    """
    Thin registered node — returns empty dict (no state change).
    Immediately followed by route_tools() condition which fans out in parallel.

    WHY a separate node?
    LangGraph nodes must return dict. The Send-returning function must be
    the condition function of add_conditional_edges, not the node itself.
    Mixing them causes InvalidUpdateError.
    """
    return {}


def route_tools(state: BGVState) -> list[Send]:
    """
    Condition function (NOT a node) — returns list[Send].
    LangGraph fires all Send targets IN PARALLEL.
    Each Send carries a ToolInput payload (read-only data for the subgraph).
    """
    def make_input(prev_ts: ToolState) -> dict:
        return {
            "tool_input": {
                "aadhaar":      state["aadhaar"],
                "hr_name":      state["hr_name"],
                "hr_dob":       state["hr_dob"],
                "hr_address":   state["hr_address"],
                "prev_version": prev_ts.get("version", 0),
            },
            "tool_result": None,
            "audit_trail": [],
            "errors":      [],
            "_retry_count": 0,
            "_last_error":  "",
        }

    targets = []
    if state["run_tool1"]: targets.append(Send("tool1_node", make_input(state["tool1"])))
    if state["run_tool2"]: targets.append(Send("tool2_node", make_input(state["tool2"])))
    if state["run_tool3"]: targets.append(Send("tool3_node", make_input(state["tool3"])))
    if not targets:        targets.append(Send("join", state))
    return targets


# Wrapper nodes that run subgraphs and write results into BGVState keys

async def tool1_node(sub_state: dict) -> dict:
    """Runs identity subgraph, maps tool_result → BGVState.tool1"""
    result = await _identity_sg.ainvoke(sub_state)
    return {
        "tool1": result["tool_result"],
        "audit_trail": result.get("audit_trail", []),
        "errors":      result.get("errors", []),
    }

async def tool2_node(sub_state: dict) -> dict:
    """Runs criminal subgraph, maps tool_result → BGVState.tool2"""
    result = await _criminal_sg.ainvoke(sub_state)
    return {
        "tool2": result["tool_result"],
        "audit_trail": result.get("audit_trail", []),
        "errors":      result.get("errors", []),
    }

async def tool3_node(sub_state: dict) -> dict:
    """Runs financial subgraph, maps tool_result → BGVState.tool3"""
    result = await _financial_sg.ainvoke(sub_state)
    return {
        "tool3": result["tool_result"],
        "audit_trail": result.get("audit_trail", []),
        "errors":      result.get("errors", []),
    }


async def join(state: BGVState) -> dict:
    """
    After parallel branches converge:
    - Marks skipped tools as cached=True
    - Applies corrected HR values from human_feedback
    - Logs HR feedback text to audit trail
    """
    # Mark skipped tools cached. HR values are already updated by human_feedback_node.
    updates = {}
    for key, flag in [("tool1","run_tool1"),("tool2","run_tool2"),("tool3","run_tool3")]:
        if not state[flag] and state[key]["status"] == "completed":
            updates[key] = {**state[key], "cached": True}
    return updates


def route_after_join(state: BGVState) -> str:
    """
    First pass  (submitted=False) → pause at human_feedback_node
    Re-run pass (submitted=True)  → generate_report
    """
    return "generate_report" if state.get("human_feedback", {}).get("submitted") else "human_feedback_node"


async def human_feedback_node(state: BGVState) -> dict:
    """
    PAUSE POINT — graph halts here via interrupt_before.
    On resume, sets run_tool flags from HR's selections.
    """
    fb = state.get("human_feedback", {})
    if not fb.get("submitted"):
        return {}

    updates = {
        "run_tool1": fb.get("run_tool1", False),
        "run_tool2": fb.get("run_tool2", False),
        "run_tool3": fb.get("run_tool3", False),
    }


    if fb.get("hr_name"):    updates["hr_name"]    = fb["hr_name"]
    if fb.get("hr_dob"):     updates["hr_dob"]     = fb["hr_dob"]
    if fb.get("hr_address"): updates["hr_address"] = fb["hr_address"]


    return updates


async def generate_report(state: BGVState) -> dict:
    t1, t2, t3 = state["tool1"], state["tool2"], state["tool3"]
    r1 = (t1["output"] or {}).get("risk_score", 0)
    r2 = (t2["output"] or {}).get("risk_score", 0)
    r3 = (t3["output"] or {}).get("risk_score", 0)
    overall_risk   = round((r1+r2+r3)/3, 1)
    any_flagged    = any((t["output"] or {}).get("flagged", False) for t in [t1,t2,t3])
    overall_status = "FLAGGED" if any_flagged else "CLEAR"
    risk_level     = "High" if overall_risk>=6 else "Medium" if overall_risk>=3 else "Low"
    completed      = sum(1 for t in [t1,t2,t3] if t["status"]=="completed")
    db_name        = (t1["output"] or {}).get("db_name", state["hr_name"])
    now            = _now()

    report = {
        "generated_at": now, "last_updated": now,
        "aadhaar": state["aadhaar"],
        "candidate_name": db_name, "hr_entered_name": state["hr_name"],
        "hr_decision": state.get("hr_decision"),   # "accepted" | "rejected" | None
        "executive_summary": {
            "overall_status": overall_status, "risk_level": risk_level,
            "overall_risk_score": overall_risk,
            "sections_completed": completed, "sections_total": 3,
            "any_flagged": any_flagged,
        },
        "personal_identity": {
            "status": t1["status"], "cached": t1.get("cached", False),
            "version": t1.get("version", 0), "data": t1["output"],
            "retry_count": t1.get("retry_count", 0), "error_message": t1.get("error_message"),
        },
        "criminal_background": {
            "status": t2["status"], "cached": t2.get("cached", False),
            "version": t2.get("version", 0), "data": t2["output"],
            "retry_count": t2.get("retry_count", 0), "error_message": t2.get("error_message"),
        },
        "financial_fraud": {
            "status": t3["status"], "cached": t3.get("cached", False),
            "version": t3.get("version", 0), "data": t3["output"],
            "retry_count": t3.get("retry_count", 0), "error_message": t3.get("error_message"),
        },
        "audit_trail": state["audit_trail"],
        "errors": state["errors"],
    }
    return {"report": report}




checkpointer = MemorySaver()


def build_bgv_graph():
    builder = StateGraph(BGVState)

    builder.add_node("dispatch_node",       dispatch_node)
    builder.add_node("tool1_node",          tool1_node)
    builder.add_node("tool2_node",          tool2_node)
    builder.add_node("tool3_node",          tool3_node)
    builder.add_node("join",                join)
    builder.add_node("human_feedback_node", human_feedback_node)
    builder.add_node("generate_report",     generate_report)

    builder.set_entry_point("dispatch_node")

    # dispatch_node → parallel fan-out via route_tools condition
    builder.add_conditional_edges(
        "dispatch_node",
        route_tools,
        ["tool1_node", "tool2_node", "tool3_node", "join"]
    )

    builder.add_edge("tool1_node", "join")
    builder.add_edge("tool2_node", "join")
    builder.add_edge("tool3_node", "join")

    # join → conditional: pause (first pass) or report (re-run)
    builder.add_conditional_edges(
        "join", route_after_join,
        {"human_feedback_node": "human_feedback_node", "generate_report": "generate_report"}
    )

    # After HR feedback: back to dispatch for selective re-run
    builder.add_edge("human_feedback_node", "dispatch_node")
    builder.add_edge("generate_report",     END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_feedback_node"],
    )


_graph = build_bgv_graph()


# ─── State helpers ────────────────────────────────────────────────────────────

def initial_tool_state() -> ToolState:
    return ToolState(
        status="idle", output=None, timestamp=None,
        execution_time=None, data_hash=None, version=0,
        cached=False, retry_count=0, error_message=None,
    )

def default_human_feedback() -> HumanFeedback:
    return HumanFeedback(
        feedback_text=None, run_tool1=False, run_tool2=False, run_tool3=False,
        hr_name=None, hr_dob=None, hr_address=None, submitted=False,
    )


# ─── Public API ───────────────────────────────────────────────────────────────

async def run_initial_verification(
    aadhaar: str, hr_name: str, hr_dob: str, hr_address: str, thread_id: str
) -> dict:
    init = BGVState(
        aadhaar=aadhaar, hr_name=hr_name, hr_dob=hr_dob, hr_address=hr_address,
        tool1=initial_tool_state(), tool2=initial_tool_state(), tool3=initial_tool_state(),
        run_tool1=True, run_tool2=True, run_tool3=True,
        human_feedback=default_human_feedback(),
        hr_decision=None,
        report=None, audit_trail=[], errors=[],
    )
    config = {"configurable": {"thread_id": thread_id}}
    state  = await _graph.ainvoke(init, config=config)
    return _build_partial_report(state, thread_id)


async def resume_with_feedback(
    thread_id: str,
    feedback_text: Optional[str],
    run_tool1: bool, run_tool2: bool, run_tool3: bool,
    hr_name: Optional[str] = None,
    hr_dob: Optional[str]  = None,
    hr_address: Optional[str] = None,
) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    
    state_update = {
        "human_feedback": HumanFeedback(
            feedback_text=feedback_text,
            run_tool1=run_tool1, run_tool2=run_tool2, run_tool3=run_tool3,
            hr_name=hr_name, hr_dob=hr_dob, hr_address=hr_address,
            submitted=True,
        )
    }
    if hr_name:    state_update["hr_name"]    = hr_name
    if hr_dob:     state_update["hr_dob"]     = hr_dob
    if hr_address: state_update["hr_address"] = hr_address

    # Write run_tool flags directly so checkpointer stores them before resume.
    state_update["run_tool1"] = run_tool1
    state_update["run_tool2"] = run_tool2
    state_update["run_tool3"] = run_tool3


    if feedback_text:
        state_update["audit_trail"] = [{
            "tool": "HR Feedback",
            "timestamp": _now(),
            "version": 0,
            "hash": None,
            "execution_time": 0,
            "action": f"HR feedback: {feedback_text}",
        }]

    await _graph.aupdate_state(config, state_update, as_node="human_feedback_node")
    state = await _graph.ainvoke(None, config=config)
    report = state.get("report", {})
    # Signal to frontend: graph has completed, not paused
    report["thread_id"] = thread_id
    report["paused_for_review"] = False
    return report


async def submit_decision(
    thread_id: str,
    decision: str,    # "accepted" | "rejected"
    remarks: Optional[str] = None,
) -> dict:
    """
    HR's final Accept / Reject decision.

    This is the clean EXIT POINT of the graph — it resumes the paused graph,
    writes the decision into BGVState, logs it to the audit trail, and lets
    generate_report run to produce the final closed report.

    Without this, the graph would stay suspended at human_feedback_node
    indefinitely whenever HR is satisfied with the report and doesn't need
    a re-run. This gives the graph a proper terminal state.

    Flow:
      aupdate_state (inject decision + submitted=True, no tool re-runs)
      → graph resumes → human_feedback_node sets run_tool flags all False
      → dispatch_node → route_tools returns [] → join (no tools ran, all cached)
      → generate_report (writes hr_decision into report) → END
    """
    config  = {"configurable": {"thread_id": thread_id}}
    label   = decision.lower()  # normalise
    now     = _now()

    audit_entry = {
        "tool": "HR Decision",
        "timestamp": now,
        "version": 0,
        "hash": None,
        "execution_time": 0,
        "action": f"Candidate {label.upper()}" + (f" — {remarks}" if remarks else ""),
    }

   
    await _graph.aupdate_state(
        config,
        {
            "hr_decision": label,
            "audit_trail": [audit_entry],
            "human_feedback": HumanFeedback(
                feedback_text=remarks,
                run_tool1=False, run_tool2=False, run_tool3=False,
                hr_name=None, hr_dob=None, hr_address=None,
                submitted=True,
            ),
            "run_tool1": False,
            "run_tool2": False,
            "run_tool3": False,
        },
        as_node="human_feedback_node",
    )

    # Resume graph — flows: dispatch → join (all cached) → generate_report → END
    state  = await _graph.ainvoke(None, config=config)
    report = state.get("report", {})
    report["thread_id"]        = thread_id
    report["paused_for_review"] = False
    report["hr_decision"]       = label
    return report


def _build_partial_report(state: BGVState, thread_id: str) -> dict:
    t1, t2, t3 = state["tool1"], state["tool2"], state["tool3"]
    r1 = (t1["output"] or {}).get("risk_score", 0)
    r2 = (t2["output"] or {}).get("risk_score", 0)
    r3 = (t3["output"] or {}).get("risk_score", 0)
    overall_risk   = round((r1+r2+r3)/3, 1)
    any_flagged    = any((t["output"] or {}).get("flagged", False) for t in [t1,t2,t3])
    overall_status = "FLAGGED" if any_flagged else "CLEAR"
    risk_level     = "High" if overall_risk>=6 else "Medium" if overall_risk>=3 else "Low"
    db_name        = (t1["output"] or {}).get("db_name", state["hr_name"])

    return {
        "thread_id": thread_id, "paused_for_review": True, "generated_at": _now(),
        "aadhaar": state["aadhaar"],
        "candidate_name": db_name, "hr_entered_name": state["hr_name"],
        "hr_decision": state.get("hr_decision"),   # None until HR decides
        "executive_summary": {
            "overall_status": overall_status, "risk_level": risk_level,
            "overall_risk_score": overall_risk,
            "sections_completed": sum(1 for t in [t1,t2,t3] if t["status"]=="completed"),
            "sections_total": 3, "any_flagged": any_flagged,
        },
        "personal_identity": {
            "status": t1["status"], "cached": t1.get("cached", False),
            "version": t1.get("version", 0), "data": t1["output"],
            "retry_count": t1.get("retry_count", 0), "error_message": t1.get("error_message"),
        },
        "criminal_background": {
            "status": t2["status"], "cached": t2.get("cached", False),
            "version": t2.get("version", 0), "data": t2["output"],
            "retry_count": t2.get("retry_count", 0), "error_message": t2.get("error_message"),
        },
        "financial_fraud": {
            "status": t3["status"], "cached": t3.get("cached", False),
            "version": t3.get("version", 0), "data": t3["output"],
            "retry_count": t3.get("retry_count", 0), "error_message": t3.get("error_message"),
        },
        "audit_trail": state["audit_trail"],
        "errors": state["errors"],
    }