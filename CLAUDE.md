# brazingtoncompiler — Claude Code Project Context

## What this project is

A Streamlit stock analysis web application (`streamlit_app.py`, ~10,800 lines). All application logic lives in that single file. There are no modules or packages to import.

**Run the app:** `streamlit run streamlit_app.py`

## Architecture

- **Storage:** PostgreSQL (set `DATABASE_URL` env var). Schema: `analysis` table (109 columns), `portfolio_memberships`, `decision_log`.
- **Data sources:** yfinance (prices, fundamentals, news), SEC EDGAR REST API (10-K/10-Q for DCF cash-flow history), US Treasury API (10-year yield for WACC).
- **Key classes:** `DatabaseManager` (~line 5500) — all DB I/O. `PortfolioBot` — efficient frontier + Monte Carlo.
- **Key constants:** `FETCH_CACHE` (in-memory TTL cache, bounded FIFO), `PEER_METRIC_MAP`, `FETCH_CACHE_MAX_ENTRIES`.

## UI structure

```
Analyst tab
├── New Analyst        — quick single-stock entry
├── Comparison         — side-by-side multi-ticker
├── AI Reports         — Claude-powered reports (equity research, comps, thesis, IC memo)
├── Methodology        — scoring explanation
├── ReadMe             — documentation
├── Controls           — model preset (Balanced / Conservative / Aggressive)
└── Senior Analyst (pw-gated)
    ├── Single Stock   — deep-dive with DCF, technicals, sentiment
    ├── Sensitivity    — assumption stress-test
    ├── Backtest       — historical strategy replay
    └── Library        — saved analyses database

Sector Leader tab      — sector-level briefings and trade flags
Portfolio Manager tab  — efficient frontier, rebalancing, decision log
```

## Data exports (already built into the app)

| Export | Where | File |
|--------|-------|------|
| Company snapshot | Senior Analyst → Single Stock | `{TICKER}_analysis_snapshot.json` |
| DCF model | Senior Analyst → Single Stock → DCF tab | `{TICKER}_dcf.json` |
| Full library | Library tab | `stock_engine_library.csv` |
| Database backup | Library tab | `stock_engine_library.db` |
| Skill-input briefs | Senior Analyst → "Export for Claude Code Skills" expander | `{TICKER}_earnings_brief.md`, `{TICKER}_comps_brief.md`, `{TICKER}_dcf_brief.md`, `{TICKER}_ic_memo_brief.md` |
| Portfolio brief | Portfolio Manager → "Export for Claude Code Skills" expander | `portfolio_rebalance_brief.md` |

## Installed financial skills — mapping to app workflows

| App workflow | Best skill(s) | Input to use |
|---|---|---|
| Single stock analysis | `/equity-research:earnings`, `/equity-research:thesis` | `{TICKER}_earnings_brief.md` |
| Valuation / DCF | `/financial-analysis:dcf-model`, `/financial-analysis:3-statement-model` | `{TICKER}_dcf_brief.md` or `{TICKER}_dcf.json` |
| Peer group (5 peers) | `/financial-analysis:comps-analysis` | `{TICKER}_comps_brief.md` |
| Portfolio holdings | `/wealth-management:rebalance`, `/wealth-management:tlh` | `portfolio_rebalance_brief.md` |
| Saved library screening | `/equity-research:screen`, `/private-equity:screen-deal` | `stock_engine_library.csv` |
| Sector briefing | `/equity-research:sector`, `/equity-research:morning-note` | Sector Leader tab context |
| Single holding write-up | `/private-equity:ic-memo`, `/investment-banking:one-pager` | `{TICKER}_ic_memo_brief.md` |
| M&A / deal analysis | `/investment-banking:buyer-list`, `/investment-banking:cim` | `{TICKER}_analysis_snapshot.json` |

## How to use skills with app data

**Quickest path — skill brief export:**
1. Analyze a ticker in the Senior Analyst tab
2. Open the "Export for Claude Code Skills" expander below the Download buttons
3. Download the relevant brief (e.g. `AAPL_earnings_brief.md`)
4. In a Claude Code session in this directory: invoke the skill with the brief as context

**Example invocations:**
```bash
# Earnings analysis report
/equity-research:earnings
# (then attach or paste AAPL_earnings_brief.md when prompted for company data)

# Comps table
/financial-analysis:comps-analysis
# (attach AAPL_comps_brief.md)

# DCF model in Excel
/financial-analysis:dcf-model
# (attach AAPL_dcf_brief.md)

# IC memo for a portfolio holding
/private-equity:ic-memo
# (attach AAPL_ic_memo_brief.md)

# Portfolio rebalance
/wealth-management:rebalance
# (attach portfolio_rebalance_brief.md from Portfolio Manager tab)
```

**In-app reports (AI Reports tab):**
The "AI Reports" tab in the Analyst section generates markdown reports directly in the browser using the Claude API (`claude-opus-4-6`). Requires `ANTHROPIC_API_KEY` env var (already used by the app for SEC guidance extraction). Reports render in 20–60 seconds.

## Environment variables

| Var | Purpose |
|-----|---------|
| `ANTHROPIC_API_KEY` | Claude API — used for SEC guidance extraction and AI Reports tab |
| `DATABASE_URL` | PostgreSQL DSN — if set, overrides SQLite |
| `SENIOR_ANALYST_PASSWORD` | Gates the Senior Analyst sub-tab |

## Key file locations

- `streamlit_app.py` — entire application
- `stocks_data.db` — default SQLite database (gitignored)
- `peer_universe.csv` — ticker universe for peer group search
- `requirements.txt` — Python dependencies (includes `anthropic`)
- `.claude/` — Claude Code settings and memory
