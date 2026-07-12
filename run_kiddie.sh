#!/usr/bin/env bash
#
# End-to-end smoke test: run a judge on the kiddie dataset,
# then meta-evaluate against the (fake) kiddie truth leaderboard.
#
# Usage:
#   source .venv/bin/activate
#   bash run_kiddie.sh
#
set -euo pipefail

OUTDIR="./output-kiddie"
TOPICS="data/kiddie/topics/kiddie-topics.jsonl"
RESPONSES="data/kiddie/runs/repgen/"
TRUTH="data/kiddie/eval/kiddie_fake.eval.ir_measures.txt"

# --- Run the LangChainPrefJudge (requires OPENAI_* env vars; smoke variant for speed) ---
WORKFLOW="judges/langchain_pref/workflow.yml"
VARIANT="smoke"

echo "=== Running LangChainPrefJudge (${VARIANT}) on kiddie ==="
auto-judge run \
    --workflow "${WORKFLOW}" \
    --variant "${VARIANT}" \
    --rag-responses "${RESPONSES}" \
    --rag-topics "${TOPICS}" \
    --out-dir "${OUTDIR}"

echo ""
echo "=== Output files ==="
ls -1 "${OUTDIR}/"

# --- Local meta-evaluation ---
if command -v auto-judge-evaluate &>/dev/null; then
    echo ""
    echo "=== Meta-evaluation (correlation with kiddie truth) ==="
    auto-judge-evaluate meta-evaluate \
        --truth-leaderboard "${TRUTH}" \
        --truth-format ir_measures --truth-header \
        --eval-format ir_measures \
        --on-missing default \
        "${OUTDIR}"/*.eval.txt
else
    echo ""
    echo "Skipping meta-evaluation (auto-judge-evaluate not installed)."
    echo "Install with: uv pip install -e '.[evaluate]'"
fi

echo ""
echo "Done."
