"""CLI entry: python -m vidmcp | vidmcp"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="vidmcp",
        description="VidMCP — SAM 3.1 AI video editing MCP server (stdio)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Check ffmpeg / Python deps and exit",
    )
    args, _unknown = parser.parse_known_args(argv)

    if args.version:
        from vidmcp import __version__

        print(__version__)
        return

    if args.doctor:
        # lightweight inline so package works without scripts/
        import importlib
        import shutil

        fails = 0
        for name in ("ffmpeg", "ffprobe"):
            path = shutil.which(name)
            print(("OK" if path else "MISSING"), name, path or "")
            fails += 0 if path else 1
        for mod in ("vidmcp", "mcp", "cv2", "numpy", "pydantic", "fastmcp"):
            try:
                m = importlib.import_module(mod)
                print("OK", mod, getattr(m, "__version__", ""))
            except Exception as exc:  # noqa: BLE001
                print("MISSING", mod, exc)
                fails += 1
        if fails:
            sys.exit(1)
        return

    from vidmcp.config import get_settings
    from vidmcp.server import create_server
    from vidmcp.utils.logging import setup_logging

    setup_logging(get_settings().log_level)
    mcp = create_server()
    # stdio is default for MCP hosts (Claude Desktop, Cursor, Claude Code)
    mcp.run()


if __name__ == "__main__":
    main()
