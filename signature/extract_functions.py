"""
Extract per-function instruction bytes/strings from a stripped binary using supplied boundaries.

Inputs:
  --binary: path to stripped binary (required)
  --symtab: path to symbol table (text from nm -a -C) OR
  --boundaries: path to JSONL/JSON with function boundaries (e.g., output of boundary predictor)

Output:
  JSONL with one function per line:
  {"name": ..., "addr_start": ..., "addr_end": ..., "inst_bytes": [...], "inst_strings": [...]}
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple


def parse_symtab(path: str) -> List[Dict]:
    funcs = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            addr_hex, sym_type, name = parts[0], parts[1], parts[-1]
            if sym_type.lower() in ("t", "t*", "t."):  # text symbols
                try:
                    addr = int(addr_hex, 16)
                except ValueError:
                    continue
                funcs.append({"name": name, "addr_start": addr})
    funcs.sort(key=lambda x: x["addr_start"])
    for i in range(len(funcs) - 1):
        funcs[i]["addr_end"] = funcs[i + 1]["addr_start"]
    if funcs:
        funcs[-1]["addr_end"] = None
    return funcs


def parse_boundaries(path: str) -> List[Dict]:
    funcs = []
    with open(path, "r") as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
        if isinstance(data, list):
            funcs = data
        elif isinstance(data, dict) and "functions" in data:
            funcs = data["functions"]
    except json.JSONDecodeError:
        funcs = []
    if not funcs:
        # Try JSONL
        with open(path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        funcs.append(obj)
                except json.JSONDecodeError:
                    continue
    for f in funcs:
        for key in ("addr_start", "addr_end"):
            if key in f and isinstance(f[key], str) and f[key].startswith("0x"):
                try:
                    f[key] = int(f[key], 16)
                except ValueError:
                    pass
    funcs = [f for f in funcs if "addr_start" in f]
    funcs.sort(key=lambda x: x["addr_start"])
    return funcs


def disassemble(binary: str) -> List[Tuple[int, bytes, str]]:
    """
    Return list of (addr, bytes, asm_str) for .text using objdump.
    """
    cmd = ["objdump", "-d", "-M", "intel", binary]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"objdump failed: {e}")

    lines = result.stdout.splitlines()
    insts = []
    pattern = re.compile(r"^\s*([0-9a-fA-F]+):\s+((?:[0-9a-fA-F]{2}\s+)+)\s*(.+)$")
    for line in lines:
        m = pattern.match(line)
        if not m:
            continue
        addr = int(m.group(1), 16)
        bytes_str = m.group(2).strip().split()
        asm = m.group(3).strip()
        try:
            b = bytes(int(x, 16) for x in bytes_str)
        except ValueError:
            continue
        insts.append((addr, b, asm))
    return insts


def slice_functions(insts: List[Tuple[int, bytes, str]], funcs: List[Dict]) -> List[Dict]:
    sliced = []
    insts.sort(key=lambda x: x[0])
    for func in funcs:
        start = func["addr_start"]
        end = func.get("addr_end")
        fn_insts = []
        for addr, b, asm in insts:
            if addr < start:
                continue
            if end is not None and addr >= end:
                break
            fn_insts.append({"addr": addr, "bytes": list(b), "asm": asm})
        if not fn_insts:
            continue
        sliced.append({
            "name": func.get("name"),
            "addr_start": start,
            "addr_end": end,
            "inst_bytes": [inst["bytes"] for inst in fn_insts],
            "inst_strings": [inst["asm"] for inst in fn_insts],
        })
    return sliced


def main():
    parser = argparse.ArgumentParser(description="Extract per-function instructions from a stripped binary using supplied boundaries.")
    parser.add_argument("--binary", required=True, help="Path to stripped binary.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--symtab", help="Path to symbol table (nm -a -C output).")
    group.add_argument("--boundaries", help="Path to JSON/JSONL with function boundaries.")
    parser.add_argument("--output", default="functions.jsonl", help="Output JSONL path.")
    args = parser.parse_args()

    if args.symtab:
        funcs = parse_symtab(args.symtab)
    else:
        funcs = parse_boundaries(args.boundaries)
    if not funcs:
        raise ValueError("No function boundaries parsed.")

    insts = disassemble(args.binary)
    sliced = slice_functions(insts, funcs)
    out_path = Path(args.output)
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w") as f:
        for fn in sliced:
            f.write(json.dumps(fn) + "\n")
    print(f"Wrote {len(sliced)} functions to {out_path}")


if __name__ == "__main__":
    main()
