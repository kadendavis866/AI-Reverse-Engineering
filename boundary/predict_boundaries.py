import argparse
import json

from config import *
from model import NN
import eval_utils as utils


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model = NN(
        checkpoint["hidden_size"],
        checkpoint["sequence_len"],
        checkpoint["num_classes"],
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_instructions(path):
    # Accept JSON array of strings, JSONL with {"idx","addr","inst"}, or plain text (one instruction per line).
    with open(path, "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            if isinstance(data, list):
                return [{"inst": str(x)} for x in data]
        lines = f.read().splitlines()
    records = []
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "inst" in obj:
                records.append(obj)
                continue
        except json.JSONDecodeError:
            pass
        records.append({"inst": line.strip()})
    return records


def slide_windows(instructions, window_size=20):
    windows = []
    starts = []
    for i in range(0, len(instructions) - window_size + 1):
        windows.append(instructions[i:i + window_size])
        starts.append(i)
    return windows, starts


def aggregate_probs(probs_list, starts, total_len, window_size):
    sums = torch.zeros(total_len)
    counts = torch.zeros(total_len)
    for prob, start in zip(probs_list, starts):
        end = start + window_size
        sums[start:end] += prob[:window_size]
        counts[start:end] += 1
    counts[counts == 0] = 1  # avoid div by zero
    return (sums / counts).tolist()


def main():
    parser = argparse.ArgumentParser(
        description="Predict function boundaries from a list of disassembled instructions.")
    parser.add_argument("--model-path", default=str(BASE_DIR / "trained_model" / "lstm_classifier.pt"),
                        help="Path to the saved model checkpoint.")
    parser.add_argument("--instructions", required=True,
                        help="Path to JSON array/text/JSONL (one instruction per line).")
    parser.add_argument("--output", default=str(BASE_DIR / "output.jsonl"),
                        help="Optional path to write predictions as JSONL.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for boundary classification.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference windows.")
    args = parser.parse_args()

    device = CUDA_DEVICE if USE_CUDA else torch.device("cpu")
    model = load_model(args.model_path, device)
    palmtree = utils.UsableTransformer(
        model_path=str(BASE_DIR / "palmtree" / "transformer.ep19"),
        vocab_path=str(BASE_DIR / "palmtree" / "vocab"),
    )

    records = load_instructions(args.instructions)
    instructions = [r["inst"] for r in records]
    if len(instructions) < 20:
        raise ValueError("Need at least 20 instructions to run sliding window predictions.")

    windows, starts = slide_windows(instructions, window_size=20)
    probs_list = []

    for i in range(0, len(windows), args.batch_size):
        batch_windows = windows[i:i + args.batch_size]
        with torch.no_grad():
            embs = palmtree.encode(batch_windows)
            embs = embs.to(device, non_blocking=True, dtype=torch.float32)
            logits = model(embs)
            probs = torch.softmax(logits, dim=-1)[..., 1].cpu()  # (batch, seq_len)
            probs_list.extend([p for p in probs])

    agg_probs = aggregate_probs(probs_list, starts, total_len=len(instructions), window_size=20)
    boundary_indices = [i - 1 for i, p in enumerate(agg_probs) if p >= args.threshold]

    idx_to_addr = {}
    for i, rec in enumerate(records):
        if "addr" in rec:
            idx_to_addr[i] = rec["addr"]

    result = {
        "instruction_count": len(instructions),
        "boundary_probs": agg_probs,
        "predicted_boundary_indices": boundary_indices,
        "predicted_boundary_addrs": [idx_to_addr.get(i) for i in boundary_indices],
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as outf:
            outf.write(json.dumps(result) + "\n")
        print(f"Wrote predictions to {args.output}")
    else:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
