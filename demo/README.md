# Demo

End-to-end demo for boundary detection and signature prediction on a sample C program.

## Files
- `demo_program.c`: Sample program.
- `Makefile`: Builds `demo_program` (with debug symbols) and `demo_program_stripped`, plus `symtab.txt`.
- `run_demo.sh`: Runs the full pipeline (build, boundary prediction, function extraction, signature prediction).

## Usage
From repo root:
```bash
bash demo/run_demo.sh
```
Requires pretrained models:
- `boundary/trained_model/lstm_classifier.pt`
- `signature/embedding/embed_output/embed_1.emb`
- `signature/rnn/output/sig_model.pt`
