# Data Acquisition Pipeline

This directory contains a self-contained builder for paired stripped/unstripped GNU utilities (coreutils, binutils, findutils) plus symbol tables for downstream ML labeling.

## What you get
- Installed binaries for each package staged under `dataset/<pkg>-<version>/...`
- For every executable: `<name>.full` (with symbols), `<name>.stripped`, and `<name>.symtab.txt` from `objdump -t`
- A machine-readable `dataset/manifest.json` describing all artifacts

## Prerequisites
- Linux toolchain: `gcc`, `g++`, `make`, `binutils` (`strip`, `objdump`)
- Build helpers: `tar`, `gzip`, `xz`, and typical development headers
- Python 3.9+ with the standard library (no external dependencies)
- Network access to fetch GNU source tarballs (from `https://ftp.gnu.org/gnu/`)

## Usage
Run the builder from this directory:
```bash
python3 build_dataset.py
```
Optional flags:
- `--output-dir dataset` change where artifacts and manifest are written
- `--work-dir .build` temporary build tree (deleted between package builds)
- `--downloads-dir .downloads` caches downloaded tarballs
- `--jobs N` parallelism for `make`
- `--packages coreutils binutils findutils` build only specific packages

Example restricting to coreutils with a custom output location:
```bash
python3 build_dataset.py --packages coreutils --output-dir /tmp/gnu-dataset --jobs 8
```

## Notes
- Builds are configured with `-g -O2` to retain symbol information in the `.full` binaries.
- Stripping is performed on a copied binary, leaving the unstripped original intact.
- If you re-run the script it will reuse cached tarballs but rebuild from source each time.
- The manifest paths are relative to the chosen `output-dir` for portability.
