#!/usr/bin/env bash
set -euo pipefail

# Demo workflow: build sample binary, extract instructions/functions, predict boundaries and signatures.
# Usage: bash demo/run_demo.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python3}"
PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export PYTHONPATH

BOUNDARY_MODEL="$ROOT_DIR/boundary/trained_model/lstm_classifier.pt"
EMBED_PATH="$ROOT_DIR/signature/embedding/embed_output/embed_1.emb"
SIG_MODEL="$ROOT_DIR/signature/rnn/output/sig_model.pt"

if [ ! -f "$BOUNDARY_MODEL" ]; then
  echo "Boundary model not found at $BOUNDARY_MODEL. Train boundary model first."
  exit 1
fi
if [ ! -f "$EMBED_PATH" ] || [ ! -f "$SIG_MODEL" ]; then
  echo "Signature embedding/model not found. Ensure $EMBED_PATH and $SIG_MODEL exist."
  exit 1
fi

echo "Building demo binaries..."
make -C "$SCRIPT_DIR" clean all >/dev/null

STRIPPED="$SCRIPT_DIR/demo_program_stripped"
SYMTAB="$SCRIPT_DIR/symtab.txt"

echo "Extracting instructions for boundary prediction..."
BOUNDARY_INSTRUCTIONS="$SCRIPT_DIR/instructions.jsonl"
"$PYTHON" "$ROOT_DIR/boundary/extract_instructions.py" "$STRIPPED" --jsonl -o "$BOUNDARY_INSTRUCTIONS"

echo "Running boundary prediction..."
BOUNDARY_PREDS="$SCRIPT_DIR/boundary_preds.jsonl"
"$PYTHON" "$ROOT_DIR/boundary/predict_boundaries.py" \
  --instructions "$BOUNDARY_INSTRUCTIONS" \
  --model-path "$BOUNDARY_MODEL" \
  --output "$BOUNDARY_PREDS"

echo "Extracting functions for signature prediction..."
FUNCTIONS_JSONL="$SCRIPT_DIR/functions.jsonl"
"$PYTHON" "$ROOT_DIR/signature/extract_functions.py" \
  --binary "$STRIPPED" \
  --symtab "$SYMTAB" \
  --output "$FUNCTIONS_JSONL"

echo "Running signature prediction..."
SIG_PREDS="$SCRIPT_DIR/sig_preds.jsonl"
"$PYTHON" "$ROOT_DIR/signature/rnn/predict_signatures.py" \
  --input "$FUNCTIONS_JSONL" \
  --embed-path "$EMBED_PATH" \
  --model-path "$SIG_MODEL" \
  --output "$SIG_PREDS"

echo "Demo complete."
echo "Boundary predictions: $BOUNDARY_PREDS"
echo "Signature predictions: $SIG_PREDS"
