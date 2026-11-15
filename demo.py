# -*- coding: utf-8 -*-
"""
demo.py — runs the agent loop fully offline, with a scripted fake LLM, so you
can see the function-calling cycle and the security model without an API key.

It demonstrates the point that matters: the tool filters by the REAL session
user, so even when the (fake) model is told to look at someone else's data, it
physically cannot — the user id never comes from the model.

Run:  python demo.py
"""
import agent_core as ac


# ── A scripted "LLM": first turn asks for a tool, second turn answers. ───────
# This replaces the network call so the demo needs no key and no connection.
class FakeLLM:
    def __init__(self):
        self.turn = 0

    def __call__(self, model, body, headers, timeout):
        self.turn += 1
        if self.turn == 1:
            # The model decides to look up the user's orders. Note it does NOT
            # pass any user id — it has no argument for one.
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "find_my_orders", "arguments": "{}"},
                }],
            }
        # Second turn: the model writes the final reply using the tool result.
        return {"role": "assistant",
                "content": "You have one order in progress, ref 260913-A1B2C3D4, "
                           "waiting for payment. Want me to check how long is left on it?"}


# ── The caller-provided tool: bound to the authenticated session user. ───────
# This is the security boundary. The user id is captured here from the real
# session (SESSION_USER), never taken from anything the model sends.
SESSION_USER = "user_42"

FAKE_DB = {
    "user_42": [{"ref": "260913-A1B2C3D4", "status": "waiting for payment"}],
    "user_99": [{"ref": "999999-DEADBEEF", "status": "completed"}],
}


def run_tool(name, arguments):
    if name == "find_my_orders":
        # Always and only this session's user — arguments is ignored on purpose.
        orders = FAKE_DB.get(SESSION_USER, [])
        return "; ".join(f"{o['ref']} ({o['status']})" for o in orders) or "no orders"
    return f"(tool {name} not implemented in the demo)"


def main():
    # Make the agent think it's configured, and swap the network call for the script.
    ac.LLM_API_KEY = "demo-key"
    ac._call = FakeLLM()

    print("=== Customer agent, scripted round trip ===\n")
    result = ac.chat(
        messages=[{"role": "user", "content": "where's my order?"}],
        system_prompt=ac.customer_system_prompt("en"),
        tools=ac.CUSTOMER_TOOLS,
        run_tool=run_tool,
        lang="en",
    )
    print("final reply:\n ", result["reply"])
    print("\nconversation steps:", [m.get("role") for m in result["messages"]])

    # The security point, shown: even a user telling the model to fetch another
    # account's data gets only their own, because run_tool ignores the model and
    # uses SESSION_USER. (The scripted model calls the same tool regardless.)
    print("\n=== Security model ===")
    print("SESSION_USER =", SESSION_USER, "-> the tool returns only this user's orders.")
    print("No prompt can change that: the id is injected by the caller, not the model.")


if __name__ == "__main__":
    main()
