# Production LLM support agent — tool-calling, prompt-injection-resistant, cost-aware failover

The chat engine behind the two AI agents of a live two-sided marketplace: a
customer-facing support advisor and an operator/admin assistant. It runs the
full function-calling loop against an LLM provider, with a security model that
holds even against prompt injection, trilingual replies, and cost-aware
provider failover.

Extracted from the production system with domain details genericized. It runs
**offline** with a scripted model — `python demo.py` shows the whole loop and
the security boundary with no API key and no network.

## The three decisions that carry the value

**1. The security boundary is the tool, not the prompt.**
A system prompt saying "never reveal another user's data" is a *suggestion* —
it helps the model behave, but it is not what makes the system safe. What makes
it safe: the customer agent's tools receive the user id of the **real
authenticated session**, injected by the caller — never anything the model
decides. The `find_my_orders` tool takes *no arguments at all*. So even if a
user convinces the model to "fetch account X's orders", the tool still returns
only the session owner's data, because the id never travels through the model.
`demo.py` demonstrates exactly this.

**2. Customer and admin agents are strictly separate.**
Two prompts, two tool lists, never shared or combined. A single agent with
"permissions conditional on who's asking" is the easiest path for a
manipulation attempt to get a customer's agent to reach an admin tool. The
admin tool list is also **read-only by construction** — there is deliberately
no tool that moves money or changes a record, so no prompt can make the agent
act, only inform.

**3. Failover is cost-aware.**
A fast/cheap model handles normal traffic. When it actually goes down (one day
it queued requests for hours while the stronger model answered in seconds), the
engine switches to a stronger model — but only for the messages inside a short
cooldown window, so the higher cost is paid only when it buys something. The
failover distinguishes a real outage (timeout, dropped connection, 429, 5xx)
from a request/account error (4xx) that the fallback would fail on too.

## Also handled

- **Trilingual (es/en/fr) driven by the UI language, not the message.** Someone
  may write briefly, ambiguously, or in a different language than the site is
  set to — the interface language is the reliable signal, so the prompt is
  built per language and the agent answers in the one the user chose.
- **Bounded tool loop and trimmed history** — a hard cap on function-calling
  round trips (no infinite loop if the model keeps calling tools) and a cap on
  messages re-sent to the API (bounded cost on long sessions).
- **Graceful degradation** — if the API key is unset the engine returns a
  "being configured" notice instead of failing, so the rest of the system can
  deploy before the key exists. Every failure path returns a user-facing
  message in the right language; the technical detail goes to the log.

## Files

| File | What it is |
|---|---|
| `agent_core.py` | The engine: prompts, tool schemas, the function-calling loop, provider failover. Pure — no database, no framework. The real tools are passed in as a callback. |
| `demo.py` | Offline run with a scripted fake model: shows the user→assistant→tool→assistant cycle and the session-scoped security boundary. |

## Run it

```bash
pip install requests
python demo.py        # fully offline, no API key needed
```

To wire it to a real backend, the caller provides `run_tool(name, arguments)`
with real data access and the authenticated session's user id, and points
`LLM_URL` / `LLM_MODEL` / `LLM_API_KEY` at an OpenAI-compatible provider.
