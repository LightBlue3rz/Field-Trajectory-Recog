"""Command-line entry points for binary turn=0 / straight=1 recognition."""

from cli import main as _main
from cli import make_parser as _make_parser


def make_parser():
    return _make_parser("point")


def main(argv=None):
    return _main("point", argv)


if __name__ == "__main__":
    raise SystemExit(main())
