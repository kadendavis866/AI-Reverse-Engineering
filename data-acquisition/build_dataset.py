#!/usr/bin/env python3
"""
Build and collect stripped/unstripped binaries for GNU coreutils, binutils, and findutils.

The script downloads sources, builds with debug symbols, installs into a staging
prefix, and then emits paired binaries plus objdump symbol tables suitable for
label extraction.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Sequence


PackageSpec = Mapping[str, object]


PACKAGES: List[PackageSpec] = [
    {
        "name": "coreutils",
        "version": "9.4",
        "url": "https://ftp.gnu.org/gnu/coreutils/coreutils-9.4.tar.xz",
        "configure_args": [],
    },
    {
        "name": "binutils",
        "version": "2.42",
        "url": "https://ftp.gnu.org/gnu/binutils/binutils-2.42.tar.xz",
        "configure_args": ["--disable-multilib"],
    },
    {
        "name": "findutils",
        "version": "4.9.0",
        "url": "https://ftp.gnu.org/gnu/findutils/findutils-4.9.0.tar.xz",
        "configure_args": [],
    },
]


def run_cmd(cmd: Sequence[str], cwd: Path | None = None, env: MutableMapping[str, str] | None = None) -> None:
    """
    Run a shell command in a subprocess.

    This function executes a given command using `subprocess.run` and ensures that
    the command completes successfully. If the command fails, a `subprocess.CalledProcessError`
    is raised.

    Args:
        cmd (Sequence[str]): The command to execute as a sequence of strings.
        cwd (Path | None, optional): The working directory to execute the command in.
            Defaults to None, which means the current working directory is used.
        env (MutableMapping[str, str] | None, optional): A dictionary of environment
            variables to use during execution. Defaults to None, which means the
            current environment is used.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero status.
    """
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def download(url: str, dest: Path) -> None:
    """
    Download a file from a URL and save it to a specified destination.

    This function fetches a file from the given URL and writes it to the specified
    destination path. If the file already exists, the function does nothing.

    Args:
        url (str): The URL of the file to download.
        dest (Path): The destination path where the file should be saved.

    Raises:
        OSError: If there is an error creating directories or writing the file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with urllib.request.urlopen(url) as response, open(dest, "wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_archive(archive: Path, dest_dir: Path) -> Path:
    """
    Extract a tar archive to a specified directory.

    This function extracts the contents of a tar archive into the given destination
    directory. It also verifies that the archive contains a single top-level directory.

    Args:
        archive (Path): The path to the tar archive to extract.
        dest_dir (Path): The destination directory where the archive should be extracted.

    Returns:
        Path: The path to the top-level directory extracted from the archive.

    Raises:
        RuntimeError: If the archive layout is unexpected or if extraction fails.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        tf.extractall(dest_dir)
        top_level = {member.name.split("/")[0] for member in tf.getmembers() if "/" in member.name}
    if len(top_level) != 1:
        raise RuntimeError(f"Unexpected archive layout for {archive}")
    return dest_dir / next(iter(top_level))


def is_elf(path: Path) -> bool:
    """
    Check if a file is an ELF (Executable and Linkable Format) binary.

    This function reads the first four bytes of a file to determine if it is an ELF
    binary by checking for the ELF magic number.

    Args:
        path (Path): The path to the file to check.

    Returns:
        bool: True if the file is an ELF binary, False otherwise.

    Raises:
        OSError: If there is an error opening or reading the file.
    """
    try:
        with open(path, "rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


def find_executables(root: Path) -> Iterable[Path]:
    """
    Find all executable ELF binaries in a directory tree.

    This function recursively searches a directory tree for files that are both
    executable and in the ELF format.

    Args:
        root (Path): The root directory to search.

    Yields:
        Path: Paths to ELF executables found in the directory tree.
    """
    for p in root.rglob("*"):
        if p.is_file() and os.access(p, os.X_OK) and is_elf(p):
            yield p


def build_and_collect(spec: PackageSpec, args: argparse.Namespace, manifest: MutableMapping[str, object]) -> None:
    """
    Build and collect stripped/unstripped binaries for a given package specification.

    This function handles the process of downloading, extracting, building, and collecting
    binaries for a specified package. It also generates stripped binaries and symbol tables
    for each executable found in the package.

    Args:
        spec (PackageSpec): A dictionary containing the package specification, including
            name, version, URL, and optional configure arguments.
        args (argparse.Namespace): Parsed command-line arguments, including directories
            for downloads, work, and output, as well as the number of parallel jobs.
        manifest (MutableMapping[str, object]): A dictionary to store metadata about the
            collected binaries, including their paths and associated symbol tables.

    Raises:
        RuntimeError: If the archive layout is unexpected or if any subprocess command fails.
    """
    name = str(spec["name"])
    version = str(spec["version"])
    url = str(spec["url"])
    configure_args = [str(x) for x in spec.get("configure_args", [])]

    # Define paths for downloads, working directory, and output dataset
    downloads_dir = Path(args.downloads_dir)
    work_dir = Path(args.work_dir)
    dataset_root = Path(args.output_dir)

    # Determine the archive file extension based on the URL
    archive_path = downloads_dir / f"{name}-{version}.tar"
    if url.endswith(".tar.gz") or url.endswith(".tgz"):
        archive_path = archive_path.with_suffix(".tar.gz")
    elif url.endswith(".tar.xz"):
        archive_path = archive_path.with_suffix(".tar.xz")
    else:
        archive_path = archive_path.with_suffix(".tar")

    # Download the package source archive
    print(f"[+] Fetching {name}-{version} from {url}")
    download(url, archive_path)

    # Create a temporary directory for building the package
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        source_root = extract_archive(archive_path, tmp_dir)

        # Create a build directory and installation prefix
        build_dir = tmp_dir / f"{name}-build"
        build_dir.mkdir()
        prefix = tmp_dir / "install"

        # Set environment variables for the build process
        env = os.environ.copy()
        env["CFLAGS"] = env.get("CFLAGS", "") + " -g -O2"
        env["CXXFLAGS"] = env.get("CXXFLAGS", "") + " -g -O2"

        # Configure the package
        configure_script = source_root / "configure"
        configure_cmd: List[str] = [str(configure_script), f"--prefix={prefix}", *configure_args]
        print(f"[+] Configuring {name}-{version}")
        run_cmd(configure_cmd, cwd=build_dir, env=env)

        # Build the package using `make`
        print(f"[+] Building {name}-{version}")
        run_cmd(["make", f"-j{args.jobs}"], cwd=build_dir, env=env)

        # Install the package to the staging prefix
        print(f"[+] Installing {name}-{version} to staging prefix")
        run_cmd(["make", "install"], cwd=build_dir, env=env)

        # Prepare the output directory for the package
        pkg_root = dataset_root / f"{name}-{version}"
        pkg_root.mkdir(parents=True, exist_ok=True)

        # Initialize the manifest entry for the package
        manifest_entry = {
            "name": name,
            "version": version,
            "source_url": url,
            "binaries": [],
        }

        # Find all ELF executables in the installed package
        for exe in find_executables(prefix):
            rel = exe.relative_to(prefix)
            dest_dir = pkg_root / rel.parent
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Define paths for the full, stripped, and symbol table versions of the binary
            binary_base = dest_dir / rel.name
            full_path = binary_base.with_suffix(binary_base.suffix + ".full")
            stripped_path = binary_base.with_suffix(binary_base.suffix + ".stripped")
            symtab_path = binary_base.with_suffix(binary_base.suffix + ".symtab.txt")

            # Copy the full binary and create the stripped version
            shutil.copy2(exe, full_path)
            shutil.copy2(exe, stripped_path)
            run_cmd(["strip", "--strip-all", str(stripped_path)])

            # Generate the symbol table for the full binary
            with open(symtab_path, "w", encoding="utf-8") as symfile:
                subprocess.run(["objdump", "-t", str(full_path)], stdout=symfile, check=True)

            # Add binary metadata to the manifest entry
            manifest_entry["binaries"].append(
                {
                    "relative": str(rel),
                    "full": str(full_path.relative_to(dataset_root)),
                    "stripped": str(stripped_path.relative_to(dataset_root)),
                    "symtab": str(symtab_path.relative_to(dataset_root)),
                }
            )

        # Append the manifest entry to the overall manifest
        manifest.setdefault("packages", []).append(manifest_entry)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the script.

    This function defines and parses the command-line arguments required to configure
    the script's behavior, such as specifying directories, the number of parallel jobs,
    and the packages to build.

    Returns:
        argparse.Namespace: A namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Build stripped/unstripped GNU utilities for ML labeling.")
    parser.add_argument("--output-dir", default="dataset", help="Where to place collected binaries and symtabs.")
    parser.add_argument("--work-dir", default=".build", help="Temporary working directory for builds.")
    parser.add_argument("--downloads-dir", default=".downloads", help="Directory to cache source archives.")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 4, help="Parallel make jobs.")
    parser.add_argument(
        "--packages",
        nargs="*",
        choices=[spec["name"] for spec in PACKAGES],
        help="Restrict build to specific packages (default: all).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the script.

    This function orchestrates the process of building and collecting binaries for
    the specified packages. It parses command-line arguments, prepares directories,
    and invokes the `build_and_collect` function for each selected package. Finally,
    it writes a manifest file summarizing the collected binaries.

    Raises:
        RuntimeError: If any subprocess command fails during the build process.
    """
    args = parse_args()
    manifest: MutableMapping[str, object] = {"packages": []}

    # Filter the packages to build based on user input
    selected = PACKAGES
    if args.packages:
        selected = [spec for spec in PACKAGES if spec["name"] in set(args.packages)]

    # Ensure output, work, and downloads directories exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)
    Path(args.downloads_dir).mkdir(parents=True, exist_ok=True)

    # Build and collect binaries for each selected package
    for spec in selected:
        build_and_collect(spec, args, manifest)

    # Write the manifest file to the output directory
    manifest_path = Path(args.output_dir) / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[+] Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()