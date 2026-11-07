"""Train the binary point-label Transformer model."""

import sys

from pointwise import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
