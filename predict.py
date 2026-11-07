"""Predict binary labels for each valid trajectory point."""

import sys

from pointwise import main

if __name__ == "__main__":
    raise SystemExit(main(["predict", *sys.argv[1:]]))
