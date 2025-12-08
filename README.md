# AI Reverse Engineering Toolkit

This repository contains two main pipelines:
- **Boundary detection** (`boundary/`): Palmtree embeddings + LSTM to find function boundaries from disassembly.
- **Signature prediction** (`signature/`): Skip-gram instruction embeddings + GRU classifier to predict function signatures (e.g., number of arguments).

## Getting Started
1) Install dependencies:
   ```bash
   ./install_deps.sh
   ```
2) Boundary workflow:
   - Generate boundary embeddings (already present under `boundary/data/palmtree_embeddings`).
   - Train boundary model: `python boundary/train.py`
   - Predict boundaries on instructions: `python boundary/predict_boundaries.py --instructions path/to/instructions.jsonl --model-path boundary/trained_model/lstm_classifier.pt --output preds.jsonl`
3) Signature workflow:
   - Prepare embedding input: `python signature/embedding/prep_embed.py -i data/pickles`
   - Train embeddings: `python signature/embedding/train_embed.py -i signature/embedding/embed_input -o signature/embedding/embed_output`
   - Train signature GRU: `python signature/rnn/sig_train.py -d data/pickles -e signature/embedding/embed_output/embed_1.emb -o signature/rnn/output`
   - Predict signatures: `python signature/rnn/predict_signatures.py --input signature/functions.jsonl --embed-path signature/embedding/embed_output/embed_1.emb --model-path signature/rnn/output/sig_model.pt`
4) Demo:
   ```bash
   bash demo/run_demo.sh
   ```

## Notes
- Ensure `PYTHONPATH=.` when using module syntax (e.g., `python -m signature.rnn.predict_signatures ...`).
- Palmtree assets live in `boundary/palmtree/`.
- Data pickles default to `data/pickles/`.
