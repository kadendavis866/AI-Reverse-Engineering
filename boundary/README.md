# Boundary Detection Pipeline

Detect function boundaries from disassembly using Palmtree embeddings + LSTM. Includes a demo workflow on a sample C program.

## Layout
- `config.py`: Central settings (device, batch size, paths). `BASE_DIR` points to this folder.
- `create_embeddings.py`: Generate Palmtree embedding shards directly from pickles in `data/pickles` (or override with `--pickle-glob`).
- `data/palmtree_embeddings/`: Embedding shards produced by the script above.
- `train.py`: Trains the LSTM on cached embeddings, saves to `trained_model/lstm_classifier.pt`.
- `predict_boundaries.py`: Sliding-window inference on instructions (text/JSON/JSONL); outputs boundary indices and addresses.
- `extract_instructions.py`: Disassembles a binary’s `.text` (via objdump), outputs instructions (optionally JSONL with idx/addr).

## Training (pickles → embeddings → model)
1) Put pickles under `../data/pickles/` (default glob `../data/pickles/**/*.pkl`).
2) Create embeddings:
   ```bash
   python create_embeddings.py
   ```
   Shards are written to `data/palmtree_embeddings/`.
3) Train:
   ```bash
   python train.py
   ```
   Model saved to `trained_model/lstm_classifier.pt`.

## Inference (binary → instructions → boundaries)
1) Extract instructions from a binary:
   ```bash
   python extract_instructions.py --binary path/to/binary --output instructions.jsonl
   ```
2) Predict boundaries:
   ```bash
    python predict_boundaries.py \
      --instructions instructions.jsonl \
      --model-path trained_model/lstm_classifier.pt \
      --output boundary_preds.jsonl
    ```
   
## Notes
- Palmtree assets expected at `palmtree/transformer.ep19` and `palmtree/vocab`.
- For CPU-only runs, set `USE_CUDA=0` in the environment.
- `requirements.txt` lists minimal Python deps (pick the correct torch wheel for your CUDA setup).
