"""Command line:

python -m launchmon <capture_dir> [--session S.json] [--ball golf|football|rugby] [--json out.json]
python -m launchmon --gui
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SESSION = os.path.join(HERE, "sessions", "session_13oct.json")


def main():
    parser = argparse.ArgumentParser(prog="launchmon")
    parser.add_argument("capture", nargs="?")
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--ball", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if args.gui or not args.capture:
        from .gui import run

        sys.exit(run(args.session))
    from . import process
    from .config import guess_ball

    ball = args.ball or guess_ball(args.capture)
    result = process(args.capture, args.session, ball, progress=None)
    print(result.summary())
    if args.json:
        result.save_json(args.json)


if __name__ == "__main__":
    main()
