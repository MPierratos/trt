#!/usr/bin/env python3
"""Wrapper script for backward compatibility with uv run client.py"""

if __name__ == "__main__":
    from inference_perf.client import main
    exit(main())

