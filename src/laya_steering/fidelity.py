from __future__ import annotations

import argparse
from pathlib import Path

from .artifact import verify_fidelity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--backend")
    args = parser.parse_args()
    verify_fidelity(args.path, args.backend)


if __name__ == "__main__":
    main()
