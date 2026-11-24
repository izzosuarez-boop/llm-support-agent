# KamasMarket AI agent: the chat engine

**Portfolio:** [ivanboot.dev](https://ivanboot.dev) · see it in [Web & API Integration](https://ivanboot.dev/web-api)

On top of KamasMarket, an AI agent connected to the site and to every order, so each user always knows where their
product and their money are. The agent looks after the users: it tracks their orders, tells them how a payment is
going, and explains any section of the site, for example how to register their BEP-20 address to receive their
payouts.

## The problem

The client asked for a complete product: a site with user accounts that works as a peer-to-peer marketplace, where
sellers list their products (here, an online game's items and currency) and buyers purchase and pay without leaving
the platform. The site is the middleman of every trade and earns a commission on each sale.

Users want a system that solves things, not a chat that answers with simple rules.

## How I built it

DeepSeek powers the site's agent. This repo is its engine, extracted from production with the domain details made
generic: the function-calling loop, the tools, and the switch to a stronger model when the main one goes down. The
tools receive the user of the real session from the backend, never from the model, so each user can only see their
own orders. The customer agent and the admin agent are kept separate, and the admin one can only read.

## Result

An AI agent connected to the site and to every order, so each user always knows where their product and their money are. A constant flow of buying and selling in a secure environment.

## Files

| File | What it is |
|---|---|
| `agent_core.py` | The engine: prompts, tool schemas, the function-calling loop, provider failover. No database and no framework: the real tools are passed in as a callback |
| `demo.py` | Offline run with a scripted model: the user → assistant → tool → assistant cycle and the session-scoped tools |

## Run it

```bash
pip install requests
python demo.py        # fully offline, no API key needed
```

To wire it to a real backend, the caller provides `run_tool(name, arguments)` with real data access and the
authenticated session's user id, and points `LLM_URL` / `LLM_MODEL` / `LLM_API_KEY` at an OpenAI-compatible provider.
