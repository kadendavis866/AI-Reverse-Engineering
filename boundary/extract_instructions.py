"""
Extract human-readable instruction lines from a stripped binary's .text section
using objdump, and write them to a plain text file (one instruction per line).

Dependencies: binutils objdump available on PATH (e.g., `objdump` or `llvm-objdump`).
"""

import argparse
import json
import os
import re
import subprocess
import sys


def run_objdump(binary_path: str, objdump_bin: str = "objdump") -> str:
    commands = [
        [objdump_bin, "-d", "--section", ".text", "-M", "intel", binary_path],
        [objdump_bin, "-d", "-M", "intel", binary_path],
    ]
    for cmd in commands:
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            return result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    raise RuntimeError(f"Failed to disassemble {binary_path} with objdump (tried {len(commands)} commands)")


def parse_instructions(objdump_output: str) -> list[tuple]:
    instructions: list[tuple] = []
    # Matches: address:  bytes    mnemonic operands
    pattern = re.compile(r"^\s*([0-9a-fA-F]+):\s+(?:[0-9a-fA-F]{2}\s+)+\s*(.+)$")
    for line in objdump_output.splitlines():
        m = pattern.match(line)
        if not m:
            continue
        addr_hex = m.group(1)
        inst = m.group(2).strip()
        # Drop inline comments/annotations (e.g., "# 2004 <...>").
        inst = re.split(r"\s*[#;].*", inst)[0].strip()
        # Normalize whitespace to single spaces for tokenizer consistency.
        inst = " ".join(inst.split())
        # Skip entries that are only hex bytes or empty (objdump sometimes emits ".byte" style lines).
        if inst and re.search(r"[A-Za-z]", inst):
            instructions.append((addr_hex, inst))
    return instructions


def main():
    parser = argparse.ArgumentParser(description="Extract .text instructions from a binary using objdump.")
    parser.add_argument("binary", help="Path to the stripped binary")
    parser.add_argument("--output", "-o", default="instructions.jsonl", help="Output path for instruction lines")
    parser.add_argument("--jsonl", action="store_true", help="Write JSONL with idx, addr, inst instead of plain text")
    parser.add_argument("--objdump", default="objdump", help="Objdump binary to use (e.g., objdump or llvm-objdump)")
    args = parser.parse_args()

    if not os.path.isfile(args.binary):
        print(f"Binary not found: {args.binary}", file=sys.stderr)
        sys.exit(1)

    objdump_output = run_objdump(args.binary, objdump_bin=args.objdump)
    instructions = parse_instructions(objdump_output)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.jsonl:
        with open(args.output, "w") as f:
            for idx, (addr, inst) in enumerate(instructions):
                f.write(json.dumps({"idx": idx, "addr": f"0x{addr}", "inst": inst}) + "\n")
    else:
        with open(args.output, "w") as f:
            for _, inst in instructions:
                f.write(inst + "\n")
    print(f"Wrote {len(instructions)} instructions to {args.output}")


if __name__ == "__main__":
    main()
