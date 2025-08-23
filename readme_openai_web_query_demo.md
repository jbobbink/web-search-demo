# OpenAI Responses API: Web Search Cost Comparison Demo

This is a tiny, standalone CLI that demonstrates the cost differences of running a single web search with OpenAI models using the Responses API. It compares:

- gpt-4.1
- gpt-4.1-mini
- gpt-5
- gpt-5-mini

The script:
- Runs your query with provider-native web_search_preview enabled
- Lets you pick one model or run them all
- Enforces a cap on built-in tool usage via max_tool_calls
- Prints the raw usage object returned by the API
- Counts the number of web search calls
- Computes token costs and web search surcharges, then shows a summary table

Why this demo?
People often ask why token counts appear high when using GPT-5 with built-in web search. The reason is pricing is different for 4.1 vs. 5 families. This demo makes that difference easy to see in one run.

- Pricing docs: https://platform.openai.com/docs/pricing#built-in-tools
- In short:
  - gpt-4.1 family (includes 4o): $25 per 1,000 web search calls, and search content tokens are free
  - gpt-5 family (and o-series): $10 per 1,000 web search calls, and search content tokens are billed at the model's text token rates

As a result, GPT-5 runs that use web search can show large input token usage because the retrieved web content is tokenized and billed in addition to the per-call surcharge. GPT-4.1 runs won't include those content tokens in the usage billed for web search—only the flat per-call surcharge applies.

## Files

- web_search_cost_demo.py — the CLI script
- requirements.txt — minimal dependencies for this demo

If you're looking at this in a larger repo, these files may be in backend/scripts. If you copy them into a new public repo, they can live at the top level.

## Prerequisites

- Python 3.9+
- An OpenAI API key with access to the listed models

## Setup

1) Create and activate a virtual environment

- macOS / Linux
  ```bash
  cd backend/scripts  # if you keep the files here; otherwise cd into their folder
  python3 -m venv .venv
  source .venv/bin/activate
  ```

- Windows (PowerShell)
  ```powershell
  cd backend/scripts  # if you keep the files here; otherwise cd into their folder
  py -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

- Windows (cmd)
  ```bat
  cd backend\scripts  
  py -m venv .venv
  .\.venv\Scripts\activate.bat
  ```

2) Install requirements

```bash
pip install -r requirements.txt
```

3) Provide your API key

- Create a .env file next to the script with:
  ```env
  OPENAI_API_KEY=sk-...
  ```
  or export it in your shell:
  - macOS / Linux: `export OPENAI_API_KEY=sk-...`
  - Windows (PowerShell): `$env:OPENAI_API_KEY = "sk-..."`

## How to use

Interactive (asks for query, model selection, max tool calls, and search context size):
```bash
python web_search_cost_demo.py
```

Non-interactive examples:
- Run all models, cap built-in tool calls at 1, use low search context
  ```bash
  python web_search_cost_demo.py -q "What was the high temperature in Los Angeles today?" -m all -t 1 -s low
  ```

- Run a subset of models by name
  ```bash
  python web_search_cost_demo.py -q "Latest US CPI report?" -m "gpt-5,gpt-4.1-mini" -t 2 -s medium
  ```

- Run using numeric indices (1=gpt-4.1, 2=gpt-4.1-mini, 3=gpt-5, 4=gpt-5-mini, 5=All)
  ```bash
  python web_search_cost_demo.py -q "NVIDIA earnings summary" -m "1,3" -t 2 -s 3
  ```

Flags
- -q/--query: your web query
- -m/--models: model selection: all | 1..5 | comma-separated names (e.g., gpt-5,gpt-4.1-mini)
- -t/--max-tool-calls: enforced max number of built-in tool calls in the response (default prompt uses 1)
- -s/--search-context-size: 1|2|3 or low|medium|high (default prompt)

What you’ll see
- Per model: web search calls, raw usage object, computed token cost, web search surcharge, total cost
- A summary table comparing all selected models

Notes
- The script enforces the tool call limit via the Responses API parameter `max_tool_calls`.
- The `search_context_size` (low/medium/high) influences how much of the search content is fed into the model and therefore token usage.
- You can optionally add a coarse `user_location` to the web_search_preview tool if you want to tailor results.

Enjoy, and feel free to adapt this script for your own analysis or demos.
