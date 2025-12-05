#!/usr/bin/env python3
"""Test script to parse solver log files."""

import json
import sys
from pathlib import Path

from search import SearchTree

# Increase recursion limit for deep trees
sys.setrecursionlimit(10000)


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_log.py <log_file> [output.json]")
        sys.exit(1)

    log_path = Path(sys.argv[1])

    if not log_path.exists():
        print(f"Error: File not found: {log_path}")
        sys.exit(1)

    output_path = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else log_path.with_suffix(".json")
    )

    print(f"Parsing log file: {log_path}")

    log_text = log_path.read_text()

    try:
        tree = SearchTree.from_log(log_text)

        tree_dict = tree.to_dict()

        with open(output_path, "w") as f:
            json.dump(tree_dict, f, indent=2)

        print(f"Tree saved to: {output_path}")

        # Count nodes
        def count_nodes(node_dict):
            total = 1
            for child in node_dict.get("children", []):
                total += count_nodes(child)
            return total

        print(f"Total nodes: {count_nodes(tree_dict)}")

    except Exception as e:
        print(f"Error parsing log: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
