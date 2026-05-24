# BGV System — Aadhaar-Based Background Verification



---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [How It Works](#how-it-works)
6. [LangGraph Pipeline — Deep Dive](#langgraph-pipeline--deep-dive)
7. [State Management](#state-management)
8. [Verification Tools](#verification-tools)
9. [Selective Re-verification](#selective-re-verification)
10. [AI-Powered Suggestions](#ai-powered-suggestions)
11. [Audit Trail](#audit-trail)
12. [Demo Candidates](#demo-candidates)
13. [How to Run](#how-to-run)


---

## Overview

The BGV System automates candidate background verification for HR teams. HR enters a candidate's **Aadhaar number** as the primary key — the system pulls the official database record and cross-references it against the HR-entered details (name, DOB, address), running identity, criminal, and financial checks simultaneously.

**Key capabilities:**

- All three verification checks run **in parallel** 
- Every tool has **retry logic** with exponential back-off and **timeout protection**
- A tool failing in isolation **never crashes the report** — the other tools complete normally
- HR can **selectively re-run** only the checks that need correction without re-running everything
- An **AI suggestion engine** reads HR free-text feedback and recommends which checks to re-run
- A **complete audit trail** logs every execution with hash, version, timestamp, and HR feedback

---

## System Architecture




### LangGraph Graph — Actual Node Visualisation

> Generated directly from LangGraph. Every box is a real registered node. Dashed lines show conditional edges. The `__interrupt = before` annotation on `human_feedback_node` is the pause point.

![alt text](image.png)

> **Reading the diagram:**
> - `dispatch_node → tool1_node / tool2_node / tool3_node` — conditional fan-out via `route_tools()`, fired in parallel
> - `tool1/2/3_node → join` — all branches converge here
> - `join → human_feedback_node` — first pass (graph pauses here, `__interrupt = before`)
> - `join → generate_report` — re-run pass (after HR submits feedback or decision)
> - `human_feedback_node → dispatch_node` — loop back for selective re-run
> - `generate_report → __end__` — graph fully complete

### Detailed Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Browser  (HR Interface)                      │
│          index.html — Tailwind CSS + Vanilla JS                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │  POST /verify   POST /rerun
                               │  POST /decide   POST /suggest-rerun
┌──────────────────────────────▼──────────────────────────────────┐
│                    FastAPI Backend  (main.py)                     │
│  /verify  /rerun  /decide  /suggest-rerun  /demo-candidates      │
│  thread_id keyed — checkpointer owns state, no sessions dict     │
└──────────────────────────────┬──────────────────────────────────┘
                               │  run_initial_verification()
                               │  resume_with_feedback()
                               │  submit_decision()
┌──────────────────────────────▼──────────────────────────────────┐
│               LangGraph Pipeline  (bgv_graph.py)                 │
│                                                                  │
│  __start__                                                       │
│      │                                                           │
│  dispatch_node      ← thin pass-through (registered node)        │
│      │                                                           │
│  route_tools()      ← condition fn — returns list[Send]          │
│   /  │  \             LangGraph fires all three IN PARALLEL      │
│  T1  T2  T3                                                      │
│   \  │  /                                                        │
│    join             ← marks skipped tools cached,                │
│      │                applies HR-corrected values                │
│      │                                                           │
│      ├─(first run: submitted=False)──────────────────────────┐  │
│      │                                                        │  │
│  human_feedback_node  ← ⏸ interrupt_before PAUSE POINT      │  │
│      │                    graph suspended here after first run│  │
│      │                    HR reviews report                   │  │
│      │                                                        │  │
│      │  ┌─── HR chooses one of three paths ──────────────┐   │  │
│      │  │                                                 │   │  │
│      │  │  PATH A — Re-verify (POST /rerun)               │   │  │
│      │  │  submitted=True, run_tool flags selective        │   │  │
│      │  │  → dispatch → selected tools → join             │   │  │
│      │  │  → PAUSES again at human_feedback_node          │   │  │
│      │  │  → HR reviews updated report                    │   │  │
│      │  │                                                 │   │  │
│      │  │  PATH B — Accept (POST /decide, accepted)        │   │  │
│      │  │  submitted=True, ALL run_tool flags=False        │   │  │
│      │  │  → dispatch → join (all cached) ──────────────┐ │   │  │
│      │  │                                               │ │   │  │
│      │  │  PATH C — Reject (POST /decide, rejected)      │ │   │  │
│      │  │  submitted=True, ALL run_tool flags=False       │ │   │  │
│      │  │  → dispatch → join (all cached) ───────────────┘ │   │  │
│      │  └─────────────────────────────────────────────────┘│   │  │
│      │                                                      │   │  │
│      └──────────(re-run: submitted=True)────────────────────┘   │  │
│                                                              │   │  │
│  generate_report    ← aggregates all tool outputs            │   │  │
│      │                writes hr_decision into report         │   │  │
│      │                                                       │   │  │
│  __end__            ← ✓ GRAPH FULLY COMPLETE                 │   │  │
│                        hr_decision = "accepted"/"rejected"   │   │  │
│                                                              │   │  │
│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    │   │  │
│  MemorySaver checkpointer stores full BGVState at every      │   │  │
│  step — server restart = no data loss                        │   │  │
└──────────────────────────────┬──────────────────────────────────┘
                               │  lookup_by_aadhaar()
┌──────────────────────────────▼──────────────────────────────────┐
│               Candidate Database  (candidate_db.py)              │
│  Official records (4 hardcoded Aadhaar entries)                  │
│  Synthetic generator (deterministic for unknown Aadhaar)         │
└─────────────────────────────────────────────────────────────────┘
```

### Tool Subgraph Architecture

Each verification tool is an isolated subgraph:

```
execute_node ──(success)──────────────────────────→ END
     │
  (error)
     │
retry_node ──(retries left, back-off 0.3s/0.6s)──→ execute_node
     │
(max retries OR timeout)
     │
 fail_node ──(writes failed ToolState, never raises)──→ END
```

This means:
- Tool 1 failing does **not** stop Tool 2 or Tool 3
- The report always completes — failed sections are marked with `status: "failed"` and show an error message
- Each tool retries up to **2 times** with exponential back-off before marking as failed
- A **10-second timeout** per tool prevents hung DB calls from stalling the pipeline

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Pipeline orchestration | LangGraph 0.2+ |
| State checkpointing | LangGraph MemorySaver |
| Async execution | Python asyncio |
| HTTP client (AI calls) | httpx |
| Frontend | HTML + Tailwind CSS + Vanilla JS |
| Language | Python 3.11+ |

---

## Project Structure

```
bgv/
├── main.py              # FastAPI app — routes, request models
├── bgv_graph.py         # LangGraph pipeline — all agent logic
├── candidate_db.py      # Aadhaar record store + synthetic generator
├── templates/
│   └── index.html       # HR-facing single-page UI
├── requirements.txt
└── README.md
```

---

## How It Works

### Step 1 — HR Enters Candidate Details

HR opens the platform and enters:
- **Aadhaar Number** (primary key — `XXXX-XXXX-XXXX` format, auto-formatted)
- **Full Name** — what the candidate claims
- **Date of Birth** — what the candidate claims
- **Current Address** — what the candidate claims

The platform auto-formats the Aadhaar number and provides quick-fill demo buttons for testing.

### Step 2 — Parallel Verification

On clicking **Run Verification**, FastAPI generates a `thread_id` (UUID) and invokes the LangGraph pipeline. All three tools fire **simultaneously**:

| Tool | What it checks | Time |
|---|---|---|
| Tool 1 — Personal Identity | Name, DOB, address vs Aadhaar DB | ~1.2s |
| Tool 2 — Criminal Background | Criminal records, Interpol, sex offender registry | ~1.5s |
| Tool 3 — Financial & Fraud | Credit score, sanctions, bankruptcy, PEP | ~1.0s |

**Total wall time ≈ 1.5s** (slowest single tool), not 3.7s sequential.

### Step 3 — Report Display

The full report appears with:
- **Executive summary** — FLAGGED / CLEAR badge, overall risk score (0–10), risk level
- **Identity tab** — side-by-side DB record vs HR entry, mismatch highlighting
- **Criminal tab** — records with Conviction / Arrest/Pending / Arrest classification
- **Financial tab** — credit score, compliance status, active flag list
- **Audit Trail tab** — every execution logged with timestamp, hash, version

### Step 4 — Selective Re-verification (optional)

If the report flags a mismatch, HR can:
1. Correct the value in the form
2. Select only the affected check(s) in the **Re-verify Tools** panel
3. Click **Run Selected Checks**

Only the selected tools re-execute. Unaffected tools serve their **cached result instantly** — no wasted computation.

---

## LangGraph Pipeline — Deep Dive

### Why two separate functions for routing?

LangGraph has a strict rule:
- Registered **nodes** must return `dict` (state update)
- **Condition functions** passed to `add_conditional_edges` can return `list[Send]` for parallel fan-out

Mixing both in one function causes `InvalidUpdateError`. The solution:

```python
# dispatch_node — registered node, returns {} (no state change)
async def dispatch_node(state: BGVState) -> dict:
    return {}

# route_tools — condition function only, returns list[Send]
def route_tools(state: BGVState) -> list[Send]:
    targets = []
    if state["run_tool1"]: targets.append(Send("tool1_node", make_input(state["tool1"])))
    if state["run_tool2"]: targets.append(Send("tool2_node", make_input(state["tool2"])))
    if state["run_tool3"]: targets.append(Send("tool3_node", make_input(state["tool3"])))
    return targets

# Wired separately
builder.add_conditional_edges("dispatch_node", route_tools, [...])
```

### Why isolated SubState per subgraph?

When three subgraphs run in parallel via `Send()`, LangGraph merges their return dicts into the parent state. If all three subgraphs declare shared fields like `aadhaar` or `hr_name`, LangGraph sees three concurrent writes to the same channel → `InvalidUpdateError`.

**Fix:** Each subgraph has its own isolated `SubState` that only declares the channels it writes:

```python
class SubState(TypedDict):
    tool_input: ToolInput    # read-only input (from Send payload)
    tool_result: Optional[ToolState]   # what this subgraph writes
    audit_trail: Annotated[list[dict], operator.add]
    errors:      Annotated[list[str],  operator.add]
    _retry_count: int
    _last_error: str
```

`aadhaar`, `hr_name`, etc. are passed in `tool_input` (read-only, not a shared channel).

### Why `operator.add` on `audit_trail` and `errors`?

When parallel branches both append to a list, LangGraph needs to know how to merge them. `Annotated[list[dict], operator.add]` tells LangGraph to **concatenate** the lists from all parallel branches, not overwrite.

### The human-in-the-loop pattern

```python
# Graph compiled with:
builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["human_feedback_node"],
)
```

**First run:** Graph runs all tools → reaches `human_feedback_node` → **halts**. State is saved to checkpointer under `thread_id`. FastAPI returns the partial report.

**Re-run:** FastAPI calls `graph.aupdate_state()` to inject HR's feedback, then `graph.ainvoke(None, config)` with `None` input — this **resumes from the pause point**, not re-invokes from scratch.

**Why write flags directly to `aupdate_state`?**

`as_node="human_feedback_node"` marks that node as already completed — its body never re-executes. All state updates (HR values, run_tool flags, feedback audit entry) must be written **directly** in the `state_update` dict passed to `aupdate_state`.

```python
await _graph.aupdate_state(
    config,
    {
        "human_feedback": HumanFeedback(..., submitted=True),
        "hr_dob":     "1992-05-14",   # corrected value — direct to BGVState
        "run_tool1":  True,            # direct to BGVState — join reads this
        "run_tool2":  False,
        "run_tool3":  False,
        "audit_trail": [feedback_entry],  # operator.add appends it
    },
    as_node="human_feedback_node",
)
```

---

## State Management

### BGVState — the pipeline's single source of truth

```python
class BGVState(TypedDict):
    aadhaar: str          # primary key — never changes
    hr_name: str          # HR's claimed value — mutable (corrections)
    hr_dob: str
    hr_address: str

    tool1: ToolState      # identity check result
    tool2: ToolState      # criminal check result
    tool3: ToolState      # financial check result

    run_tool1: bool       # True = execute, False = serve cached
    run_tool2: bool
    run_tool3: bool

    human_feedback: HumanFeedback  # HR's re-run request
    report: Optional[dict]
    audit_trail: Annotated[list[dict], operator.add]
    errors:      Annotated[list[str],  operator.add]
```

### ToolState — per-tool execution metadata

```python
class ToolState(TypedDict):
    status: str           # idle | completed | failed | timeout
    output: Optional[dict]
    timestamp: str        # UTC ISO-8601
    execution_time: float # seconds
    data_hash: str        # SHA-256[:16] — change detection between versions
    version: int          # increments on every execution
    cached: bool          # True if this run skipped the tool
    retry_count: int      # how many retries happened
    error_message: str    # captured error if failed
```

---

## Verification Tools

### Tool 1 — Personal Identity

Compares HR-entered values against Aadhaar DB record:

**Name matching** (`_name_match`):
- First token (first name) must match exactly after lowercasing
- Overall token set overlap must be ≥ 80%
- Result: `Confirmed` (exact string) / `Partial` (token match) / `Mismatch`

**Address matching** (`_addr_match`):
- Strips punctuation, lowercases, splits into token sets
- Substring containment check first
- Token overlap ≥ 60% = match
- Also checks `prev_addresses[]` — candidates who moved are not penalised

**Risk scoring:**

| Mismatch | Points |
|---|---|
| Name mismatch | +4 |
| DOB mismatch | +3 |
| Address mismatch | +2 |
| Aadhaar expired | +3 |
| Max score | 10 |

### Tool 2 — Criminal Background

- Checks criminal records with type, year, jurisdiction, and court status
- Classifies each record: `Conviction` / `Arrest/Pending` / `Arrest`
- Checks Interpol watch-list and sex offender registry
- Covers 4 jurisdictions: Federal, State, County, Interpol
- Risk scoring: crimes +4, Interpol +4, sex offender +5

### Tool 3 — Financial & Fraud

- Credit score with band classification (Excellent ≥750 / Good ≥700 / Fair ≥600 / Poor)
- Fraud indicators: None / Low / High
- OFAC/UN/EU sanctions check
- Bankruptcy history
- PEP (Politically Exposed Person) status
- Adverse media and active litigation
- `compliance_status`: Non-Compliant if risk score ≥ 5

### Overall Risk Score

```
overall_risk = round((tool1.risk_score + tool2.risk_score + tool3.risk_score) / 3, 1)
```

| Score | Level |
|---|---|
| ≥ 6.0 | High |
| ≥ 3.0 | Medium |
| < 3.0 | Low |

Report is `FLAGGED` if **any** tool has `flagged: True`.

---

## Selective Re-verification

After the initial report, HR can re-run individual checks without re-running everything:

1. **Correct the form value** (e.g. fix a typo in DOB)
2. **Select only the affected check** (e.g. tick Tool 1 — Personal Identity)
3. **Click Run Selected Checks**

**What happens internally:**

- Selected tools re-execute with fresh DB lookup
- Skipped tools return their previous result with `cached: True` and `version` unchanged
- The corrected HR values are written directly into `BGVState` via `aupdate_state` **before** the tools run — so tools always receive the corrected values
- `generate_report` merges fresh and cached results into a new report

**Version tracking:**
- Each tool execution increments `version` (v1 → v2 → v3...)
- The `data_hash` (SHA-256[:16]) changes if the output changes between versions
- UI shows `FRESH v2` or `CACHED v1` badge per section

---

## HR Decision — Accept or Reject

After reviewing the report, HR makes a final **Accept** or **Reject** decision using the buttons shown below the report.

### Why this matters architecturally

Without a decision endpoint, the LangGraph graph stays **suspended at `human_feedback_node` indefinitely** whenever HR is satisfied with the result and doesn't need a re-run. The Accept/Reject button gives the graph a **clean exit point** — it resumes the paused graph, records the decision, and drives it to `__end__`.

```
HR clicks Accept / Reject
        │
   POST /decide
        │
  aupdate_state (decision + submitted=True, all run_tool flags=False)
        │
  graph resumes → dispatch_node → route_tools() → [] (no tools selected)
        │
       join (all tools marked cached)
        │
  generate_report (hr_decision written into report)
        │
      __end__   ← graph fully complete
```

### What HR sees

- **Accept Candidate** button (green) — candidate passes, verification closed
- **Reject Candidate** button (red) — candidate does not pass
- Optional **remarks** field — HR can add a note (e.g. "All checks passed, suitable for the role")
- Once decided, the buttons are replaced by a coloured result banner
- The re-verify panel is hidden — no further checks can be run on a closed verification

### What gets recorded

The decision is logged in the audit trail:
```
[HR Decision]  Candidate ACCEPTED — All checks passed, suitable for the role   11:36:45
```

And in the report:
```json
{
  "hr_decision": "accepted",   // or "rejected"
  ...
}
```

### Graph lifecycle summary

| Stage | Graph state | HR action |
|---|---|---|
| After `/verify` | Paused at `human_feedback_node` | Review report |
| After `/rerun` | Paused again at `human_feedback_node` | Review updated report |
| After `/decide` | Reached `__end__` — fully closed | None (verification complete) |

---

## AI-Powered Suggestions

The **AI Suggest** feature sends HR's free-text feedback to an LLM:

```
HR types: "The date of birth was entered incorrectly"
LLM returns: {"tool1": true, "tool2": false, "tool3": false, "reasoning": "DOB is verified by Tool 1"}
```

Two modes:
- **AI Suggest** — pre-fills the checkboxes, HR reviews and clicks Run
- **AI Suggest & Run** — immediately executes the AI-selected tools in one click

---

## Audit Trail

Every action is logged chronologically:

| Entry type | What it records |
|---|---|
| Tool execution | tool name, timestamp (UTC), version, SHA-256 hash, execution time |
| Retry | same as above with `executed (retry #N)` action |
| Tool failure | failure reason, retry count, timeout flag |
| HR Feedback | free-text feedback submitted by HR before re-run |

Example audit trail after a correction:
```
1. [Personal Identity]    executed        11:34:34  hash:ba8255ba...
2. [Criminal Background]  executed        11:34:34  hash:98204fb8...
3. [Financial & Fraud]    executed        11:34:33  hash:48baed62...
4. [HR Feedback]          HR feedback: DOB was entered incorrectly
5. [Personal Identity]    executed        11:36:20  hash:ba8255ba...
```

---

## Demo Candidates

Four pre-loaded candidates for testing — accessible via quick-fill buttons:

| Candidate | Aadhaar | Expected Result | Notable Flags |
|---|---|---|---|
| Rahul Sharma | `1234-5678-9012` | CLEAR | Clean record, credit 780 |
| Sneha Iyer | `9876-5432-1098` | CLEAR | Clean record, credit 820 |
| Priya Patel | `1111-2222-3333` | FLAGGED | Expired Aadhaar, bankruptcy, adverse media |
| Amit Verma | `4444-5555-6666` | FLAGGED (HIGH) | Fraud conviction, Interpol, sanctions, PEP |

Any **unknown Aadhaar** number triggers the synthetic record generator — deterministic (same Aadhaar always produces the same person), with 70% clean / 20% medium-risk / 10% high-risk probability distribution.

---

## How to Run

### Prerequisites

- Python 3.11 or higher
- pip

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd bgv
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
fastapi>=0.110.0
uvicorn>=0.29.0
langgraph>=0.2.0
langchain-core>=0.2.0
httpx>=0.27.0
pydantic>=2.0.0
```

### 3. Set up file structure

Ensure your files are laid out as:
```
bgv/
├── main.py
├── bgv_graph.py
├── candidate_db.py
├── templates/
│   └── index.html
└── requirements.txt
```

### 4. Run the server

```bash
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open the platform

Navigate to **http://localhost:8000** in your browser.

### 6. Optional — AI Suggest feature

The AI Suggest feature requires an API key. Enter it in the top-right field of the UI and click **SAVE**. The key is used only for `/suggest-rerun` calls.

---

## API Reference

### `POST /verify`

Run full verification for a new candidate.

**Request:**
```json
{
  "aadhaar": "1234-5678-9012",
  "hr_name": "Rahul Sharma",
  "hr_dob": "1992-05-14",
  "hr_address": "42 Marine Lines, Mumbai, MH 400001"
}
```

**Response:**
```json
{
  "thread_id": "uuid-v4",
  "report": {
    "candidate_name": "Rahul Sharma",
    "executive_summary": {
      "overall_status": "CLEAR",
      "risk_level": "Low",
      "overall_risk_score": 0.0,
      "sections_completed": 3,
      "any_flagged": false
    },
    "personal_identity": { "status": "completed", "version": 1, "cached": false, "data": {...} },
    "criminal_background": { "status": "completed", "version": 1, "cached": false, "data": {...} },
    "financial_fraud": { "status": "completed", "version": 1, "cached": false, "data": {...} },
    "audit_trail": [...],
    "errors": []
  }
}
```

---

### `POST /rerun`

Re-run selected checks for an existing verification session.

**Request:**
```json
{
  "thread_id": "uuid-from-verify",
  "rerun_tool1": true,
  "rerun_tool2": false,
  "rerun_tool3": false,
  "feedback": "DOB was entered incorrectly",
  "hr_dob": "1992-05-14"
}
```

**Response:** Same structure as `/verify`, with updated tool versions and cached flags.

---

### `POST /suggest-rerun`

Use AI to determine which tools need re-running based on HR feedback.

**Request:**
```json
{
  "feedback": "The candidate's address does not seem correct",
  "openai_key": "ak-..."
}
```

**Response:**
```json
{
  "tool1": true,
  "tool2": false,
  "tool3": false,
  "reasoning": "Address is verified by Tool 1 — Personal Identity"
}
```

---

### `POST /decide`

HR's final Accept / Reject decision. Resumes the paused graph and drives it to `__end__`.

**Request:**
```json
{
  "thread_id": "uuid-from-verify",
  "decision": "accepted",
  "remarks": "All checks passed, suitable for the role"
}
```

**Response:** Full final report with `hr_decision` field set and all tools showing `cached: true`.

```json
{
  "thread_id": "...",
  "hr_decision": "accepted",
  "paused_for_review": false,
  "audit_trail": [
    { "tool": "Personal Identity", "action": "executed", ... },
    { "tool": "Criminal Background", "action": "executed", ... },
    { "tool": "Financial & Fraud", "action": "executed", ... },
    { "tool": "HR Decision", "action": "Candidate ACCEPTED — All checks passed", ... }
  ]
}
```

---

### `GET /demo-candidates`

Returns the four pre-loaded demo candidates for quick-fill.

---

## Key Design Decisions

### Why LangGraph over plain asyncio?

LangGraph provides a typed state machine with explicit node boundaries, built-in checkpointing, interrupt/resume for human-in-the-loop patterns, and automatic audit of state transitions. Plain asyncio would require manually implementing all of this.

### Why `interrupt_before` instead of FastAPI managing the loop?

In Approach 2 (the previous version), FastAPI held `BGVState` in a `sessions{}` dict and re-invoked `graph.ainvoke()` from scratch on every re-run. This meant:
- State was lost on server restart
- The feedback loop was an external workaround, not a first-class graph concern

With `interrupt_before` + `MemorySaver`, the graph **owns its lifecycle** — it pauses, the checkpointer stores state durably, and `ainvoke(None, same_thread_id)` resumes from the exact pause point.

### Why dispatch_node + route_tools as separate functions?

LangGraph nodes must return `dict`. The `list[Send]` return needed for parallel fan-out is only valid as a condition function passed to `add_conditional_edges`. Combining both in one function causes `InvalidUpdateError`. Two separate functions keeps the API contract clear.

### Why isolated SubState per subgraph?

Three parallel subgraphs writing to shared BGVState channels causes concurrent write conflicts. Each subgraph declares only the channels it writes — the rest is passed as a read-only `ToolInput` payload via `Send`. The parent graph maps `tool_result` back to the correct `tool1/2/3` key in wrapper nodes.

### Why write corrected values directly in `aupdate_state`?

`as_node="human_feedback_node"` marks the node as already completed — its body never re-executes on resume. Any state updates (corrected HR values, run_tool flags, feedback audit entry) must be written directly in the `state_update` dict passed to `aupdate_state`, not inside the node function body.

### Complete graph lifecycle (with decision)

```
╔══════════════════════════════════════════════════════════════╗
║                   COMPLETE LIFECYCLE                         ║
╚══════════════════════════════════════════════════════════════╝

1. POST /verify
      │
      ▼
   All 3 tools run IN PARALLEL (T1 + T2 + T3 simultaneously)
      │
      ▼
   ⏸ PAUSED at human_feedback_node  ←── MemorySaver stores state
      │
      ▼
   Report returned to HR  (paused_for_review: true)
      │
      ├──── HR sees a mismatch ──────────────────────────────────┐
      │                                                          │
      │     2. POST /rerun (correct value + select tool)         │
      │           │                                              │
      │           ▼                                              │
      │        Selected tools re-run in PARALLEL                 │
      │           │                                              │
      │           ▼                                              │
      │        ⏸ PAUSED again at human_feedback_node            │
      │           │                                              │
      │           ▼                                              │
      │        Updated report returned to HR                     │
      │           │                                              │
      │           └──────────────────────────────────────────────┘
      │                (can repeat re-run as many times as needed)
      │
      └──── HR is satisfied with report ────────────────────────┐
                                                                │
           3a. POST /decide  { decision: "accepted" }           │
                 │                                              │
                 ▼                                              │
              submitted=True, ALL run_tool=False                │
              graph resumes → dispatch → join (all cached)      │
              → generate_report (hr_decision="accepted")        │
              → ✅ __end__  (graph fully complete)              │
                                                                │
           3b. POST /decide  { decision: "rejected" }           │
                 │                                              │
                 ▼                                              │
              submitted=True, ALL run_tool=False                │
              graph resumes → dispatch → join (all cached)      │
              → generate_report (hr_decision="rejected")        │
              → ✅ __end__  (graph fully complete)              │
                                                                │
           ◀──────────────────────────────────────────────────┘

Final report contains:
  • All tool results (fresh or cached with version numbers)
  • hr_decision: "accepted" | "rejected"
  • Complete audit trail: tool executions + HR feedback + HR decision
  • paused_for_review: false
```

### Production upgrade path

| Feature | Current (prototype) | Production upgrade |
|---|---|---|
| State persistence | `MemorySaver` (in-memory) | `SqliteSaver` or `RedisSaver` |
| Authentication | None | JWT middleware on all routes |
| DB layer | Python dict + generator | Real Aadhaar API integration |
| Tool parallelism | 3 fixed tools | Dynamic tool registry |
| Sessions | Lost on restart | Durable via checkpointer swap |