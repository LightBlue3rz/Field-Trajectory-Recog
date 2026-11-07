"""Run operation classification with python -m agri."""

from cli import main, make_parser

__all__ = ["main", "make_parser"]

if __name__ == "__main__":
    raise SystemExit(main("pattern"))
