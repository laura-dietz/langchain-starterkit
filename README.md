# LangChain Preference-Nugget AutoJudge

A [TREC AutoJudge](https://trec-auto-judge.cs.unh.edu/) built on [LangChain](https://python.langchain.com/), imitating the [prefnugget-starterkit](https://github.com/laura-dietz/prefnugget-starterkit) `best-decide-plum` variant — with an added twist: **decision points that create more preference pairs on demand** when nugget extraction stalls before its target.

The canonical guide for setup, LLM configuration, running, caching, meta-evaluation, and submission is the [TREC AutoJudge Participant HowTo](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/README.md); in [Claude Code](https://docs.anthropic.com/en/docs/claude-code), the `/autojudge-setup`, `/autojudge-develop`, and `/autojudge-submit` skills walk through the activities interactively.

## How the judge works

| Phase | What happens |
|-------|--------------|
| 1 — Preference ranking | Pairwise `must_decide` preference judgments (both passage orders; inconsistent verdicts dropped) over a deterministic pool of comparisons; Borda scores rank the responses. |
| 2 — Contrastive nugget extraction | Winner/loser pairs, strongest first (`borda(winner) + 0.99·borda(loser)`), are mined iteratively for *differentiating* questions — one pair per round, one question per pair ("plum"), deduplicated, until the bank holds `target_nuggets` (20). |
| 3 — Grading | Every (response, nugget) pair is graded 0–5; a nugget counts as covered at grade ≥ 4. Measures: `NUGGET_COVERAGE`, `AVG_GRADE`, `MAX_GRADE`, `COVERED_COUNT`. |

**The decision points (new over prefnugget):** Phase 1 starts with a deliberately small pool (`initial_num_others: 2` comparisons per response). Whenever Phase 2 runs out of unconsumed pairs — or the last remaining pair yields nothing new — while the bank sits below target, the judge *decides* to judge additional preference pairs (one more comparison offset per response), recomputes Borda, and continues extracting. It gives up only when the pool cannot grow further (all pairs judged) or the `max_pairs_considered` budget is spent.

LLM access uses LangChain's `ChatOpenAI` against the endpoint injected via `llm_config` (never hardcoded), with lenient regex/JSON parsing so no provider-specific tool-calling is required. Prompt caching uses an **endpoint-agnostic** SQLite cache under `$CACHE_DIR` (LangChain's default key includes `openai_api_base`, which would break TIRA's deterministic re-execution with a disabled endpoint — the wrapper normalizes it out), per the [prompt-cache contract](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/05-prompt-cache.md).

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[all]'

export OPENAI_BASE_URL=... OPENAI_MODEL=... OPENAI_API_KEY=...
export CACHE_DIR=./cache

auto-judge run \
    --workflow judges/langchain_pref/workflow.yml \
    --variant best-decide-plum \
    --rag-responses data/kiddie/runs/repgen/ \
    --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
    --out-dir ./output-kiddie/
```

Variants: `best-decide-plum` (default parameters) and `smoke` (small targets for fast iteration). Handy overrides: `-N initial_num_others=1` (exercise the growth decision points), `-N target_nuggets=10`, `-J grade_threshold=3`.

On the tiny kiddie dataset (4 runs → at most 6 pairs), the 20-nugget target is intentionally unreachable — the judge extracts what the pool supports and stops gracefully; expect ~5–6 nuggets per topic.

## Layout

```
judges/langchain_pref/
  langchain_judge.py   # prompts, parsers, PrefPool, decision loop, judge class
  workflow.yml         # phases, settings, variants
tests/test_langchain_pref.py  # parser + pool unit tests (no LLM)
```

## License

MIT
