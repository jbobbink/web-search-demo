import os
import json
from typing import Any, Dict, List
import streamlit as st
from openai import OpenAI

# ---- Pricing Config ----
TOKEN_PRICING_PER_MILLION = {
    "gpt-5": {"inputTokens": 1.25, "inputReadCache": 0.1250, "outputTokens": 10.00},
    "gpt-5-mini": {"inputTokens": 0.25, "inputReadCache": 0.025, "outputTokens": 2.00},
    "gpt-4.1": {"inputTokens": 2.00, "inputReadCache": 0.50, "outputTokens": 8.00},
    "gpt-4.1-mini": {"inputTokens": 0.40, "inputReadCache": 0.10, "outputTokens": 1.60},
}
WEB_SEARCH_CALL_COST = {
    "gpt-4.1": 0.025,
    "gpt-4.1-mini": 0.025,
    "gpt-5": 0.01,
    "gpt-5-mini": 0.01,
}
MODEL_OPTIONS = ["gpt-4.1", "gpt-4.1-mini", "gpt-5", "gpt-5-mini"]

MAX_OUTPUT_TOKENS = 4000
DEFAULT_MAX_TOOL_CALLS = 1
DEFAULT_SEARCH_CONTEXT_SIZE = "low"
REASONING_EFFORT = "low"

# ---- Helpers ----
def compute_token_cost(model: str, usage: Dict[str, Any]) -> float:
    pricing = TOKEN_PRICING_PER_MILLION.get(model, {})
    if not pricing or not usage:
        return 0.0
    in_tokens = int((usage.get("input_tokens") or 0) or 0)
    cached = int(((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0) or 0)
    uncached = max(in_tokens - cached, 0)
    out_tokens = int((usage.get("output_tokens") or 0) or 0)
    cost_uncached = (uncached * pricing["inputTokens"]) / 1_000_000
    cost_cached = (cached * pricing.get("inputReadCache", pricing["inputTokens"])) / 1_000_000
    cost_output = (out_tokens * pricing["outputTokens"]) / 1_000_000
    return round(cost_uncached + cost_cached + cost_output, 6)

def web_search_call_cost(model: str, calls: int) -> float:
    per_call = WEB_SEARCH_CALL_COST.get(model, 0.0)
    return round(calls * per_call, 6)

def run_query(client, model: str, query: str, max_tool_calls: int, search_context_size: str):
    system_msg = {
        "role": "system",
        "content": [
            {"type": "input_text", "text": (
                f"You can use provider-native web_search. "
                f"Make at most {max_tool_calls} queries. "
                f"Cite key sources when helpful."
            )}
        ],
    }
    user_msg = {"role": "user", "content": [{"type": "input_text", "text": query}]}
    tools = [{"type": "web_search_preview", "search_context_size": search_context_size}]
    req_kwargs = {
        "model": model,
        "input": [system_msg, user_msg],
        "tools": tools,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_tool_calls": max_tool_calls,
        "parallel_tool_calls": False,
    }
    if str(model).startswith("gpt-5"):
        req_kwargs["reasoning"] = {"effort": REASONING_EFFORT}
    resp = client.responses.create(**req_kwargs)

    # Extract usage
    usage = {}
    if hasattr(resp, "usage") and resp.usage:
        if hasattr(resp.usage, "model_dump"):
            usage = resp.usage.model_dump()
        else:
            usage = {
                "input_tokens": getattr(resp.usage, "input_tokens", 0),
                "output_tokens": getattr(resp.usage, "output_tokens", 0),
                "input_tokens_details": getattr(resp.usage, "input_tokens_details", {}),
                "output_tokens_details": getattr(resp.usage, "output_tokens_details", {}),
            }

    text = getattr(resp, "output_text", "") or ""
    token_cost = compute_token_cost(model, usage)
    search_cost = web_search_call_cost(model, getattr(resp, "max_tool_calls", max_tool_calls))
    total_cost = token_cost + search_cost

    return {
        "model": model,
        "answer": text,
        "usage": usage,
        "token_cost_usd": token_cost,
        "web_search_cost_usd": search_cost,
        "total_cost_usd": round(total_cost, 6),
    }

# ---- Streamlit UI ----
st.set_page_config(page_title="OpenAI Web Search Cost Demo", layout="wide")
st.title("🔎 OpenAI Web Search Cost Comparison")

# API key input
api_key = st.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    client = OpenAI(api_key=api_key)
else:
    st.warning("⚠️ Please enter your API key to continue.")
    st.stop()

# Query & settings
query = st.text_input("Enter your web query", "Latest news on AI regulation in Europe")
selected_models = st.multiselect("Choose models", MODEL_OPTIONS, default=["gpt-4.1", "gpt-5-mini"])
max_tool_calls = st.slider("Max web tool calls", 1, 5, DEFAULT_MAX_TOOL_CALLS)
search_context_size = st.radio("Search context size", ["low", "medium", "high"], index=0)

if st.button("Run Comparison"):
    results = []
    for m in selected_models:
        with st.spinner(f"Running {m}..."):
            try:
                info = run_query(client, m, query, max_tool_calls, search_context_size)
                results.append(info)
            except Exception as e:
                st.error(f"Error with {m}: {e}")

    if results:
        st.subheader("Results")
        for r in results:
            with st.expander(f"📌 {r['model']} — ${r['total_cost_usd']:.6f}"):
                st.write("**Answer:**")
                st.write(r["answer"] or "_No answer_")
                st.json(r["usage"])
                st.write(f"💰 Token cost: ${r['token_cost_usd']:.6f}")
                st.write(f"💰 Web search surcharge: ${r['web_search_cost_usd']:.6f}")
                st.write(f"💰 **Total estimated cost: ${r['total_cost_usd']:.6f}**")

        # Summary table
        st.subheader("Summary")
        st.table([
            {
                "Model": r["model"],
                "Token Cost ($)": f"{r['token_cost_usd']:.6f}",
                "Web Cost ($)": f"{r['web_search_cost_usd']:.6f}",
                "Total ($)": f"{r['total_cost_usd']:.6f}",
            }
            for r in results
        ])
