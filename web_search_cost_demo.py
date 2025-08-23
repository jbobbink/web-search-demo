#!/usr/bin/env python3
"""
Simple CLI to compare web search costs across OpenAI models using the Responses API.

- Prompts for a web query
- Lets you choose which models to run: gpt-4.1, gpt-4.1-mini, gpt-5, gpt-5-mini, or ALL
- Performs a single non-streaming call per selected model with web_search_preview enabled
- Prints the API's raw usage object, counts web search calls, and estimates cost

Assumptions
- OPENAI_API_KEY is set in the environment (use a .env file if you like)
- Uses search_context_size = "low" to reduce token usage
- We nudge the model to limit web calls via instruction and also offer a max_tool_calls hint

Pricing notes (Aug 2025)
- Web search: $0.025/call for gpt-4.1 family (gpt-4.1, gpt-4.1-mini); search content tokens are free
- Web search: $0.01/call for gpt-5 family (gpt-5, gpt-5-mini); search content tokens are billed at model token rates
- Token pricing per 1M tokens (USD) is embedded for the 4 models we compare, based on your model-settings

References
- https://platform.openai.com/docs/pricing#built-in-tools
- https://platform.openai.com/docs/api-reference/responses/create
"""

import json
import os
import sys
import argparse
from datetime import datetime
from typing import Any, Dict, List, Tuple

try:
    from dotenv import load_dotenv  # Optional convenience
    load_dotenv()
except Exception:
    pass

try:
    from openai import OpenAI
except Exception as e:
    print("This script requires the openai Python package. Install with: pip install openai python-dotenv")
    raise

# ---- Configuration ----
# Per 1M tokens pricing (USD) for the selected models
TOKEN_PRICING_PER_MILLION = {
    "gpt-5": {"inputTokens": 1.25, "inputReadCache": 0.1250, "outputTokens": 10.00},
    "gpt-5-mini": {"inputTokens": 0.25, "inputReadCache": 0.025, "outputTokens": 2.00},
    "gpt-4.1": {"inputTokens": 2.00, "inputReadCache": 0.50, "outputTokens": 8.00},
    "gpt-4.1-mini": {"inputTokens": 0.40, "inputReadCache": 0.10, "outputTokens": 1.60},
}

# Web search call surcharges (USD per call)
WEB_SEARCH_CALL_COST = {
    # 4.1 family (also 4o) flat $25 / 1k calls
    "gpt-4.1": 0.025,
    "gpt-4.1-mini": 0.025,
    # 5 family (and o-series) $10 / 1k calls
    "gpt-5": 0.01,
    "gpt-5-mini": 0.01,
}

MODEL_OPTIONS = [
    ("gpt-4.1", "GPT 4.1"),
    ("gpt-4.1-mini", "GPT 4.1-mini"),
    ("gpt-5", "GPT 5"),
    ("gpt-5-mini", "GPT 5-mini"),
]

DEFAULT_SEARCH_CONTEXT_SIZE = "low"  # reduce token usage by default
DEFAULT_MAX_TOOL_CALLS = 1
MAX_OUTPUT_TOKENS = 4000  # leave room for reasoning
REASONING_EFFORT = 'low'
# Max characters of the visible answer we print. Set to 0 or negative to disable truncation.
MAX_ANSWER_CHARS = 1800
SAVE_FULL_RESPONSES = True
LOG_DIR = "logs"


def count_web_search_calls(resp: Any) -> Tuple[int, List[Dict[str, Any]]]:
    """Count provider-native web_search_call items from a Responses API result.

    This inspects response.output items where item.type == 'web_search_call'.
    Returns (count, details) where details includes the query if present.
    """
    details: List[Dict[str, Any]] = []
    seen_keys = set()

    try:
        output = getattr(resp, "output", None)
        if not output:
            return 0, []
        for item in output:
            item_type = getattr(item, "type", None)
            if item_type == "web_search_call":
                # Try to extract unique key and attributes
                iid = getattr(item, "id", None)
                act = getattr(item, "action", None)
                q = getattr(act, "query", None) if act else None
                status = getattr(item, "status", None)
                key = ("web_search_call", iid) if iid else ("web_search_call", (q or "").strip().lower())
                if key not in seen_keys:
                    seen_keys.add(key)
                    details.append({"type": "web_search_call", "id": iid, "query": q, "status": status})
            # Sometimes web search calls might appear as provider_tool_call blocks in adapters,
            # but in raw SDK Responses it should be 'web_search_call'. Keeping just in case:
            if item_type == "provider_tool_call":
                tool_type = getattr(item, "tool_type", None)
                if tool_type == "web_search_call":
                    iid = getattr(item, "id", None)
                    act = getattr(item, "action", None)
                    q = getattr(act, "query", None) if act else None
                    status = getattr(item, "status", None)
                    key = ("web_search_call", iid) if iid else ("web_search_call", (q or "").strip().lower())
                    if key not in seen_keys:
                        seen_keys.add(key)
                        details.append({"type": "web_search_call", "id": iid, "query": q, "status": status})
    except Exception:
        pass

    return len(details), details


def extract_text(resp: Any) -> str:
    """Get the best-effort text content from a Responses API result."""
    try:
        # New SDKs often expose output_text directly
        txt = getattr(resp, "output_text", None)
        if isinstance(txt, str) and txt.strip():
            return txt
    except Exception:
        pass

    # Fallback: scan output items
    try:
        output = getattr(resp, "output", None)
        if output:
            for item in output:
                if getattr(item, "type", None) == "message":
                    for c in getattr(item, "content", []) or []:
                        ttype = getattr(c, "type", None)
                        if ttype in ("output_text", "text"):
                            t = getattr(c, "text", "")
                            if t:
                                return t
    except Exception:
        pass

    return ""


def usage_to_dict(resp: Any) -> Dict[str, Any]:
    """Return the API usage object as a plain dict for printing."""
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    try:
        # pydantic models have model_dump
        if hasattr(u, "model_dump"):
            return u.model_dump()
    except Exception:
        pass

    # Fallback manual mapping
    try:
        return {
            "input_tokens": getattr(u, "input_tokens", None),
            "output_tokens": getattr(u, "output_tokens", None),
            "total_tokens": getattr(u, "total_tokens", None),
            "input_tokens_details": getattr(u, "input_tokens_details", {}),
            "output_tokens_details": getattr(u, "output_tokens_details", {}),
        }
    except Exception:
        return {}


def extract_url_citations(resp: Any) -> List[Dict[str, Any]]:
    """Best-effort extraction of web citations from a Responses API result.

    Returns a list of {url, title?} dicts, de-duplicated by URL.
    Handles both annotation-style citations and web_search_tool_result content.
    """
    def sget(obj, name, default=None):
        if hasattr(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return default

    cites: List[Dict[str, Any]] = []
    seen: set = set()

    try:
        output = sget(resp, "output", []) or []
        for item in output:
            # 1) Look for annotations on the item itself
            anns = sget(item, "annotations", []) or []
            for a in anns:
                atype = sget(a, "type")
                url = sget(a, "url")
                title = sget(a, "title")
                if url and (atype in ("url_citation", "url-citation", None)):
                    if url not in seen:
                        cites.append({"url": url, **({"title": title} if title else {})})
                        seen.add(url)

            # 2) Look inside message content blocks
            content = sget(item, "content", []) or []
            for c in content:
                # Annotations on content items
                cann = sget(c, "annotations", []) or []
                for a in cann:
                    atype = sget(a, "type")
                    url = sget(a, "url")
                    title = sget(a, "title")
                    if url and (atype in ("url_citation", "url-citation", None)):
                        if url not in seen:
                            cites.append({"url": url, **({"title": title} if title else {})})
                            seen.add(url)

                # web_search_tool_result with web_search_result items
                ctype = sget(c, "type")
                if ctype in ("web_search_tool_result", "web_search_result", "tool_result"):
                    # try content list (may contain results)
                    ccontent = sget(c, "content", []) or []
                    for r in ccontent:
                        rurl = sget(r, "url")
                        rtitle = sget(r, "title")
                        if rurl and rurl not in seen:
                            cites.append({"url": rurl, **({"title": rtitle} if rtitle else {})})
                            seen.add(rurl)
    except Exception:
        pass

    return cites


def _safe_serialize(obj: Any) -> Any:
    """Best-effort conversion of SDK objects to JSON-serializable structures."""
    try:
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, list):
            return [_safe_serialize(x) for x in obj]
        if isinstance(obj, tuple):
            return [_safe_serialize(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _safe_serialize(v) for k, v in obj.items()}
        # pydantic-like
        if hasattr(obj, "model_dump"):
            try:
                return _safe_serialize(obj.model_dump())
            except Exception:
                pass
        if hasattr(obj, "model_dump_json"):
            try:
                import json as _json
                return _json.loads(obj.model_dump_json())
            except Exception:
                pass
        # generic object
        if hasattr(obj, "__dict__"):
            try:
                return {k: _safe_serialize(v) for k, v in vars(obj).items() if not k.startswith("_")}
            except Exception:
                pass
    except Exception:
        pass
    # fallback
    try:
        return str(obj)
    except Exception:
        return None


def write_response_log(
    model: str,
    query: str,
    max_tool_calls: int,
    search_context_size: str,
    resp: Any,
    usage: Dict[str, Any],
    counted_calls: int,
    call_details: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
    attempted_total: int = None,
) -> str:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%fZ")
    model_tag = (model or "model").replace("/", "-").replace(":", "-")
    filename = f"{ts}_{model_tag}.json"
    path = os.path.join(LOG_DIR, filename)
    payload = {
        "timestamp_utc": ts,
        "request": {
            "model": model,
            "query": query,
            "max_tool_calls": max_tool_calls,
            "search_context_size": search_context_size,
        },
        "usage": usage,
        "web_search": {
            "counted_calls": counted_calls,
            "attempted_total": attempted_total,
            "details": call_details,
        },
        "citations": citations,
        "raw_response": _safe_serialize(resp),
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  [debug] failed to write log {path}: {e}")
    return path


def compute_token_cost(model: str, usage: Dict[str, Any]) -> float:
    """Compute token costs using per-1M pricing.

    - Uncached input tokens are billed at inputTokens
    - Cached input tokens (if usage.input_tokens_details.cached_tokens present) billed at inputReadCache
    - Output tokens billed at outputTokens
    """
    pricing = TOKEN_PRICING_PER_MILLION.get(model)
    if not pricing or not usage:
        return 0.0

    # Input tokens breakdown
    input_total = int((usage.get("input_tokens") or 0) or 0)
    cached = 0
    try:
        cached = int(((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0) or 0)
    except Exception:
        cached = 0
    uncached = max(input_total - cached, 0)

    output_tokens = int((usage.get("output_tokens") or 0) or 0)

    cost_uncached = (uncached * pricing["inputTokens"]) / 1_000_000.0
    cost_cached = (cached * pricing.get("inputReadCache", pricing["inputTokens"])) / 1_000_000.0
    cost_output = (output_tokens * pricing["outputTokens"]) / 1_000_000.0

    return round(float(cost_uncached + cost_cached + cost_output), 6)


def web_search_call_cost(model: str, calls: int) -> float:
    per_call = WEB_SEARCH_CALL_COST.get(model, 0.0)
    return round(float(calls) * per_call, 6)


def run_query(client: OpenAI, model: str, query: str, max_tool_calls: int, search_context_size: str) -> Dict[str, Any]:
    """Execute one Responses API call with web_search_preview and return gathered info."""
    # Provide a short system instruction to limit web queries
    system_msg = {
        "role": "system",
        "content": [
            {"type": "input_text", "text": (
                f"You can use provider-native web_search to retrieve up-to-date information. "
                f"Make at most {max_tool_calls} web search queries before answering. "
                f"Cite key sources in-line when helpful."
            )}
        ],
    }
    user_msg = {"role": "user", "content": [{"type": "input_text", "text": query}]}

    # Enable web search tool with low context to reduce token usage
    tools = [{"type": "web_search_preview", "search_context_size": search_context_size}]
    # Optionally provide a coarse location if desired
    # tools[0]["user_location"] = {"type": "approximate", "country": "US", "city": "Los Angeles", "region": "CA", "timezone": "America/Los_Angeles"}

    # Build request with conditional reasoning effort (supported on GPT-5 family only)
    req_kwargs: Dict[str, Any] = {
        "model": model,
        "input": [system_msg, user_msg],
        "tools": tools,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_tool_calls": max_tool_calls,
        "parallel_tool_calls": False,
    }
    if str(model).startswith("gpt-5") and REASONING_EFFORT:
        req_kwargs["reasoning"] = {"effort": REASONING_EFFORT}

    resp = client.responses.create(**req_kwargs)

    # Safety check: log what the server reports back and compute billable vs attempted calls
    total_found = 0
    details_all: List[Dict[str, Any]] = []
    billable_count = 0
    ignored_attempts = 0
    try:
        resp_mtc = getattr(resp, "max_tool_calls", None)
        if resp_mtc is None and hasattr(resp, "model_dump"):
            try:
                md = resp.model_dump()
                resp_mtc = md.get("max_tool_calls")
            except Exception:
                pass
        total_found, details_all = count_web_search_calls(resp)
        completed = [d for d in (details_all or []) if (d or {}).get("status") == "completed"]
        billable_count = min(len(completed), int(max_tool_calls or 0))
        ignored_attempts = max(0, total_found - billable_count)
        if ignored_attempts > 0:
            print(
                f"  [debug] tool-calls: requested={max_tool_calls} response.max_tool_calls={resp_mtc} "
                f"counted={billable_count} ({ignored_attempts} additional tool call(s) attempted, assumed ignored according to documentation and not counting as billable)"
            )
    except Exception:
        pass

    usage = usage_to_dict(resp)
    # Use previously computed counts; fallback if unavailable
    if not details_all:
        total_found, details_all = count_web_search_calls(resp)
        completed = [d for d in (details_all or []) if (d or {}).get("status") == "completed"]
        billable_count = min(len(completed), int(max_tool_calls or 0))
        ignored_attempts = max(0, total_found - billable_count)
    text = extract_text(resp)
    citations = extract_url_citations(resp)

    # Optionally save a full response log for analysis
    log_path = None
    if SAVE_FULL_RESPONSES:
        try:
            log_path = write_response_log(
                model=model,
                query=query,
                max_tool_calls=max_tool_calls,
                search_context_size=search_context_size,
                resp=resp,
                usage=usage,
                counted_calls=billable_count,
                attempted_total=total_found,
                call_details=details_all,
                citations=citations,
            )
        except Exception as e:
            print(f"  [debug] failed to serialize response: {e}")

    token_cost = compute_token_cost(model, usage)
    search_cost = web_search_call_cost(model, billable_count)
    total_cost = round(token_cost + search_cost, 6)

    return {
        "model": model,
        "answer": text,
        "usage": usage,
        "web_search_calls": billable_count,
        "web_search_attempted": total_found,
        "web_search_details": details_all,
        "citations": citations,
        "token_cost_usd": token_cost,
        "web_search_cost_usd": search_cost,
        "total_cost_usd": total_cost,
    }


def choose_models() -> List[str]:
    print("Which models would you like to run?")
    for idx, (name, label) in enumerate(MODEL_OPTIONS, start=1):
        print(f"  {idx}. {label} ({name})")
    print("  5. All of the above")
    print("You can select:")
    print("  - A single number (1-4)")
    print("  - Comma-separated numbers or model names (e.g., 1,3 or gpt-5,gpt-4.1-mini)")
    print("  - 5 for All")

    while True:
        sel = input("Enter selection (e.g., 1 or 1,3 or gpt-5,gpt-4.1-mini, or 5 for All): ").strip()
        models = parse_models_arg(sel)
        if models:
            return models
        print("Please enter a valid selection (single number, comma-separated numbers/names, or 5 for All).")


def prompt_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        if v <= 0:
            raise ValueError()
        return v
    except Exception:
        print(f"Invalid number; using default {default}.")
        return default


def parse_models_arg(arg: str) -> List[str]:
    """Parse --models argument into a list of model names.
    Accepts: 'all', comma-separated names, or numeric indices (1-5).
    """
    if not arg:
        return []
    s = arg.strip().lower()
    if s in {"all", "5"}:
        return [m for m, _ in MODEL_OPTIONS]
    # split by comma
    out: List[str] = []
    mapping = {str(i + 1): name for i, (name, _) in enumerate(MODEL_OPTIONS)}
    valid_names = {name for name, _ in MODEL_OPTIONS}
    for tok in (t.strip() for t in s.split(",")):
        if not tok:
            continue
        if tok in mapping:
            out.append(mapping[tok])
        elif tok in valid_names:
            out.append(tok)
        else:
            print(f"Warning: unknown model selector '{tok}' - ignoring")
    # de-dup but preserve order
    seen = set()
    uniq: List[str] = []
    for m in out:
        if m not in seen:
            uniq.append(m)
            seen.add(m)
    return uniq


def choose_search_context_size() -> str:
    print("Select search context size:")
    print("  1. low")
    print("  2. medium")
    print("  3. high")
    while True:
        sel = input("Enter a number (1-3): ").strip()
        if sel == "1":
            return "low"
        if sel == "2":
            return "medium"
        if sel == "3":
            return "high"
        print("Please enter a valid option (1-3).")


def parse_search_context_size_arg(arg: str) -> str:
    if not arg:
        return ""
    s = arg.strip().lower()
    if s in {"1", "low"}:
        return "low"
    if s in {"2", "medium", "med"}:
        return "medium"
    if s in {"3", "high"}:
        return "high"
    print(f"Warning: unknown search_context_size '{arg}', using default '{DEFAULT_SEARCH_CONTEXT_SIZE}'")
    return DEFAULT_SEARCH_CONTEXT_SIZE

def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set. Create a .env with OPENAI_API_KEY=... or export it in your shell.")

    client = OpenAI()

    print("Web Search Cost Comparison Demo")
    print("--------------------------------")

    # CLI args to support non-interactive usage
    parser = argparse.ArgumentParser(description="Web Search Cost Comparison Demo (OpenAI Responses API)")
    parser.add_argument("--query", "-q", type=str, help="Web query to run")
    parser.add_argument("--models", "-m", type=str, help="Models to run: 'all', comma-separated names, or numbers 1-5")
    parser.add_argument(
        "--max-tool-calls",
        "-t",
        type=int,
        default=None,
        help="Max built-in tool calls to allow (default prompt).")
    parser.add_argument("--search-context-size", "-s", type=str, help="Search context size: 1|2|3 or low|medium|high")
    args = parser.parse_args()

    # Query
    if args.query:
        query = args.query.strip()
    else:
        query = input("Enter your web query: ").strip()
        if not query:
            print("No query provided. Exiting.")
            sys.exit(1)

    # Models
    models = parse_models_arg(args.models) if args.models else choose_models()
    if not models:
        print("No valid models selected. Exiting.")
        sys.exit(1)

    # Max tool calls: if not provided as a flag, ask interactively, default to 1
    if args.max_tool_calls is not None and args.max_tool_calls > 0:
        max_tool_calls = args.max_tool_calls
    else:
        max_tool_calls = prompt_int("Max web tool calls to allow (enforced)", DEFAULT_MAX_TOOL_CALLS)

    # Search context size
    search_context_size = parse_search_context_size_arg(args.search_context_size) if args.search_context_size else choose_search_context_size()
    if not search_context_size:
        search_context_size = DEFAULT_SEARCH_CONTEXT_SIZE

    print("")
    results: List[Dict[str, Any]] = []
    for m in models:
        print(f"Running {m}...")
        try:
            info = run_query(client, m, query, max_tool_calls, search_context_size)
        except Exception as e:
            print(f"  Error calling {m}: {e}")
            continue
        results.append(info)
        # Pretty-print per-model result
        print(f"  Web search calls: {info['web_search_calls']}")
        print(f"  Usage (raw): {json.dumps(info['usage'], indent=2)}")
        if info["answer"]:
            ans = info["answer"].strip()
            if MAX_ANSWER_CHARS and MAX_ANSWER_CHARS > 0 and len(ans) > MAX_ANSWER_CHARS:
                ans = ans[:MAX_ANSWER_CHARS] + "..."
            print("  Answer:")
            print("  " + "\n  ".join(ans.splitlines()))
        # Print citations if present
        cites = info.get("citations") or []
        if cites:
            print("  Citations:")
            for i, c in enumerate(cites, 1):
                url = c.get("url")
                title = c.get("title")
                label = f"{title} - {url}" if title else (url or "")
                print(f"    {i}. {label}")
        # Token cost and breakdown
        print(f"  Token cost: ${info['token_cost_usd']:.6f}")
        try:
            usage_dbg = info.get("usage") or {}
            pricing_dbg = TOKEN_PRICING_PER_MILLION.get(info.get("model"), {}) or {}
            in_price = float(pricing_dbg.get("inputTokens", 0.0))
            read_price = float(pricing_dbg.get("inputReadCache", in_price))
            out_price = float(pricing_dbg.get("outputTokens", 0.0))
            in_tokens = int((usage_dbg.get("input_tokens") or 0) or 0)
            cached_tokens = int(((usage_dbg.get("input_tokens_details") or {}).get("cached_tokens") or 0) or 0)
            uncached_tokens = max(in_tokens - cached_tokens, 0)
            out_tokens = int((usage_dbg.get("output_tokens") or 0) or 0)
            cost_uncached = (uncached_tokens * in_price) / 1_000_000.0
            cost_cached = (cached_tokens * read_price) / 1_000_000.0
            cost_output = (out_tokens * out_price) / 1_000_000.0
            # Variable-formula line
            print("  Token cost formula: ((uncached_input_tokens × inputPrice) + (cached_input_tokens × cachedPrice) + (output_tokens × outputPrice)) / 1,000,000")
            print("  Where: uncached_input_tokens = input_tokens − cached_tokens")
            # Numeric substitution line
            print(
                f"  = (({uncached_tokens} × {in_price}) + ({cached_tokens} × {read_price}) + ({out_tokens} × {out_price})) / 1,000,000"
            )
            # Component totals and final
            print(
                f"  = (${cost_uncached:.6f} + ${cost_cached:.6f} + ${cost_output:.6f}) = ${info['token_cost_usd']:.6f}"
            )
        except Exception:
            pass
        print(f"  Web search surcharge: ${info['web_search_cost_usd']:.6f}")
        print(f"  Total estimated cost: ${info['total_cost_usd']:.6f}")
        # Inform about saved log path if logging is enabled
        if SAVE_FULL_RESPONSES:
            print(f"  [debug] saved response log in {LOG_DIR}/ (per-model JSON)")
        print("")

    if not results:
        print("No successful results.")
        return

    # Summary table
    print("Summary")
    print("-------")
    headers = ["Model", "Calls", "Token Cost ($)", "Web Cost ($)", "Total ($)"]
    rows = []
    for r in results:
        rows.append([
            r.get('model', ''),
            str(r.get('web_search_calls', '')),
            f"${r.get('token_cost_usd', 0.0):.6f}",
            f"${r.get('web_search_cost_usd', 0.0):.6f}",
            f"${r.get('total_cost_usd', 0.0):.6f}",
        ])
    # Compute column widths
    widths = []
    for i, h in enumerate(headers):
        maxlen = len(h)
        for row in rows:
            if i < len(row):
                maxlen = max(maxlen, len(str(row[i])))
        widths.append(maxlen)
    def fmt_row(cols):
        return "  " + "  ".join(str(col).ljust(widths[i]) for i, col in enumerate(cols))
    print(fmt_row(headers))
    print("  " + "  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


if __name__ == "__main__":
    main()