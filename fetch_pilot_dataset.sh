#!/usr/bin/env bash
#
# fetch_pilot_dataset.sh - download the TREC AutoJudge pilot (v0.2) data into ./local-data/
#
# The pilot/training runs are password-protected (HTTP basic auth). Provide credentials via the
# environment (never commit them):
#
#   export TREC_AUTOJUDGE_USER=trec2025        # default: trec2025
#   export TREC_AUTOJUDGE_PASSWORD=...          # required
#
# Usage:
#   ./fetch_pilot_dataset.sh                        # fetch all tracks (dragun25, rag25, ragtime25)
#   ./fetch_pilot_dataset.sh --dataset dragun-repgen   # fetch the track for one dataset (repeatable)
#   ./fetch_pilot_dataset.sh --keep-archive        # keep the .tar.gz after extracting
#
# Note: this fetches the PILOT/training runs only. The TREC 2026 AutoJudge test data releases in
# August; a sibling fetch_test_dataset.sh will handle it. Corpora/topics for some tracks come from
# the host tracks (see the data release page); this script fetches the released run tarballs.

set -euo pipefail

BASE_URL="https://trec-auto-judge.cs.unh.edu/datareleases/v0.2"
DEST="./local-data"
USER_NAME="${TREC_AUTOJUDGE_USER:-trec2025}"

# dataset-name (or track) -> tarball track key. rag25-gen and rag25-auggen share the rag25 tarball.
declare -A TRACK_OF=(
  [dragun-repgen]=dragun25 [dragun25]=dragun25
  [rag25-gen]=rag25 [rag25-auggen]=rag25 [rag25]=rag25
  [ragtime25]=ragtime25
)

usage() { grep '^#' "$0" | sed 's/^#\s\?//'; exit "${1:-0}"; }

TARGETS=()
KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) shift; [ $# -gt 0 ] || { echo "--dataset needs a value" >&2; exit 1; }; TARGETS+=("$1");;
    --keep-archive) KEEP=1;;
    -h|--help) usage 0;;
    *) echo "Unknown argument: $1" >&2; usage 1;;
  esac
  shift
done

if [ -z "${TREC_AUTOJUDGE_PASSWORD:-}" ]; then
  echo "Error: set TREC_AUTOJUDGE_PASSWORD (HTTP basic-auth password for the data release)." >&2
  echo "  export TREC_AUTOJUDGE_PASSWORD=...   # ask the organizers; do not commit it" >&2
  exit 1
fi

# Resolve requested targets into a unique set of track tarballs.
declare -A WANT=()
if [ ${#TARGETS[@]} -eq 0 ]; then
  WANT[dragun25]=1; WANT[rag25]=1; WANT[ragtime25]=1
else
  for t in "${TARGETS[@]}"; do
    track="${TRACK_OF[$t]:-}"
    if [ -z "$track" ]; then
      echo "Unknown dataset/track: $t (known: ${!TRACK_OF[*]})" >&2
      exit 1
    fi
    WANT[$track]=1
  done
fi

mkdir -p "$DEST"
for track in "${!WANT[@]}"; do
  tarball="anonymized-runs-${track}-v2.tar.gz"
  url="$BASE_URL/$tarball"
  out="$DEST/$track"
  echo "==> Fetching $tarball"
  mkdir -p "$out"
  tmp="$(mktemp)"
  curl -fSL --user "$USER_NAME:$TREC_AUTOJUDGE_PASSWORD" -o "$tmp" "$url"
  echo "==> Extracting into $out"
  tar -xzf "$tmp" -C "$out"
  if [ "$KEEP" -eq 1 ]; then
    cp "$tmp" "$DEST/$tarball"
  fi
  rm -f "$tmp"
  echo "Layout:"
  find "$out" -maxdepth 3 -type d | sed 's/^/  /'
done

echo
echo "Done -> $DEST/. Verify the extracted paths match datasets.yml (runs/<task>/ and topics/<file>);"
echo "if a tarball wraps its contents in a top-level directory, adjust the datasets.yml paths to match."
