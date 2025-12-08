# Signature Prediction Pipeline

Instruction embeddings + GRU classifier to predict function signatures (e.g., number of arguments).

## Steps
1) Prepare embedding input from pickles:
   ```bash
   python signature/embedding/prep_embed.py -i data/pickles -o signature/embedding/embed_input
   ```
2) Train instruction embeddings:
   ```bash
   python signature/embedding/train_embed.py -i signature/embedding/embed_input -o signature/embedding/embed_output
   ```
3) Train signature classifier:
   ```bash
   python signature/rnn/sig_train.py \
     -d data/pickles \
     -e signature/embedding/embed_output/embed_1.emb \
     -o signature/rnn/output
   ```
4) Predict signatures:
   ```bash
   python signature/rnn/predict_signatures.py \
     --input signature/functions.jsonl \
     --embed-path signature/embedding/embed_output/embed_1.emb \
     --model-path signature/rnn/output/sig_model.pt \
     --output signature/rnn/output/sig_preds.jsonl
   ```

## Function extraction
Use `signature/extract_functions.py` to build JSONL inputs for prediction from a stripped binary plus a symtab or boundary file:
```bash
python signature/extract_functions.py \
  --binary path/to/stripped.bin \
  --symtab path/to/symtab.txt \
  --output signature/functions.jsonl
```
or with boundary predictions:
```bash
python signature/extract_functions.py \
  --binary path/to/stripped.bin \
  --boundaries path/to/boundary_preds.jsonl \
  --output signature/functions.jsonl
```
