# OpenAI Web Search Demo (Responses API)

This small CLI demonstrates the usage and cost differences of OpenAI's built-in web search across these models:

- gpt-4.1
- gpt-4.1-mini
- gpt-5
- gpt-5-mini

It performs a single Responses API call per selected model with the provider-native `web_search_preview` tool enabled and shows:
- The raw `usage` object from the API
- How many web searches the model actually performed
- Estimated token cost + web-search surcharge cost
- A compact answer and a final table summarizing all models

Why this demo?
- There’s confusion about why GPT-5 web searches can show high token counts. Pricing for built-in tools differs across model families:
  - gpt-4.1 family (includes 4o): $25 / 1k web search calls; retrieved search content tokens are included (not billed separately)
  - gpt-5 family (and o-series): $10 / 1k web search calls; retrieved search content tokens are billed at the model’s token rates

See pricing details from OpenAI:
- https://platform.openai.com/docs/pricing#built-in-tools

This demo makes the difference visible in one run.

## Files
- `web_search_cost_demo.py` — the CLI script
- `requirements.txt` — minimal dependencies
- `.env.example` — example environment file (copy to `.env`)

## Prerequisites
- Python 3.9+
- An OpenAI API key with access to the listed models

## Setup

1) Clone or copy this folder and create a virtual environment

- macOS / Linux
  ```bash
  cd web-search-demo
  python3 -m venv .venv
  source .venv/bin/activate
  ```

- Windows (PowerShell)
  ```powershell
  cd web-search-demo
  py -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```

- Windows (cmd)
  ```bat
  cd web-search-demo
  py -m venv .venv
  .\.venv\Scripts\activate.bat
  ```

2) Install requirements
```bash
pip install -r requirements.txt
```

3) Configure your API key
- Copy `.env.example` to `.env` and paste your key:
  ```env
  OPENAI_API_KEY=sk-...
  ```
  Or set it in your shell instead:
  - macOS / Linux: `export OPENAI_API_KEY=sk-...`
  - Windows (PowerShell): `$env:OPENAI_API_KEY = "sk-..."`

## Usage

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
- Run by numeric indices (1=gpt-4.1, 2=gpt-4.1-mini, 3=gpt-5, 4=gpt-5-mini, 5=All)
  ```bash
  python web_search_cost_demo.py -q "NVIDIA earnings summary" -m "1,3" -t 2 -s 3
  ```

Flags
- `-q/--query` — your web query
- `-m/--models` — `all`, `1..5`, or comma-separated names (e.g., `gpt-5,gpt-4.1-mini`)
- `-t/--max-tool-calls` — enforced cap on built-in tool calls in a single response (default prompt uses 1)
- `-s/--search-context-size` — `1|2|3` or `low|medium|high`

Output
- Per model: number of web calls, raw usage dict, token cost, web search surcharge, total
- Citations list (URLs and titles when available)
- Final table with per-model summary

Notes
- The script passes `max_tool_calls` to the Responses API (enforced) and sets `web_search_preview` with `search_context_size` (low/medium/high). We also set `parallel_tool_calls=False` to reduce the chance of concurrent tool calls when a strict cap is desired.
- The Responses API may still show more than `max_tool_calls` web_search_call items in the output. Based on our tests and the OpenAI dashboard, only up to `max_tool_calls` are billed; extra attempts appear to be ignored by the service. Our cost calculator self-enforces the cap (counts only completed calls up to the limit) and logs the total attempted calls for transparency.
- You can change the default visible answer length by editing the constant `MAX_ANSWER_CHARS` in `web_search_cost_demo.py` (set to 0 or negative to disable truncation). `MAX_OUTPUT_TOKENS` controls the model's maximum output tokens.
- GPT-5 family can accumulate more input tokens because the content retrieved from the web search is billed at input token rates in addition to the per-call surcharge, while in the 4.1 family the search content tokens are included.
- Logs: each run writes a JSON file under `logs/` with the full serialized response, usage, citations, and both `counted_calls` (billable, capped) and `attempted_total` (raw events seen).

## License
This project is provided under the MIT License (see `LICENSE`). It permits reuse and distribution, and includes an "AS IS" warranty disclaimer to limit liability.

