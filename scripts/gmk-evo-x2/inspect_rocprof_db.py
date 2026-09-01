#!/usr/bin/env python3
"""Inspect a ROCm rocprofv3 SQLite trace without requiring the sqlite3 CLI.

This GMKtec EVO-X2 helper is deliberately read-only.  It inventories the database
schema first, then prints one representative row from each trace table so a
subsequent aggregation can use the exact ROCm-version-specific column names.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Accept the immutable profiler database to inspect."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="rocprofv3 *_results.db artifact")
    parser.add_argument(
        "--tail-seconds",
        type=float,
        default=0.0,
        help="aggregate only the final positive-duration kernel window, zero prints schema only",
    )
    return parser.parse_args()


def main() -> int:
    """Print a compact schema inventory and representative records, then exit."""

    args = parse_args()
    if not args.database.is_file():
        raise SystemExit(f"missing profiler database: {args.database}")

    # Open the evidence database in immutable read-only mode so inspection cannot
    # create journal files or change the captured trace under any circumstances.
    uri = f"file:{args.database.resolve()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        # SQLite's catalog is the authoritative list of ROCm trace tables.
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        tables = [row["name"] for row in rows]
        for table in tables:
            # Quote table names defensively even though rocprof creates them.
            quoted = '"' + table.replace('"', '""') + '"'
            columns = connection.execute(f"PRAGMA table_info({quoted})").fetchall()
            column_names = [column["name"] for column in columns]
            count = connection.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
            print(f"TABLE {table} rows={count} columns={','.join(column_names)}")
            # A single row gives the names and units needed for a version-safe
            # aggregate without dumping the large raw trace into the terminal.
            sample = connection.execute(f"SELECT * FROM {quoted} LIMIT 1").fetchone()
            if sample is not None:
                values = ";".join(f"{key}={sample[key]!r}" for key in sample.keys())
                print(f"SAMPLE {table} {values}")
        if args.tail_seconds > 0:
            # rocprof version-stamps every table name with one UUID.  Selecting
            # by prefix keeps this analysis portable across ROCm trace versions.
            dispatch = next(name for name in tables if name.startswith("rocpd_kernel_dispatch_"))
            symbols = next(name for name in tables if name.startswith("rocpd_info_kernel_symbol_"))
            quoted_dispatch = '"' + dispatch.replace('"', '""') + '"'
            quoted_symbols = '"' + symbols.replace('"', '""') + '"'
            # Timestamps are nanoseconds.  The final active dispatch is a stable
            # anchor because the profiler may remain alive after request work ends.
            last_end = connection.execute(
                f"SELECT MAX(end) FROM {quoted_dispatch} WHERE end > start"
            ).fetchone()[0]
            cutoff = last_end - int(args.tail_seconds * 1_000_000_000)
            aggregate = connection.execute(
                f"""
                SELECT s.kernel_name AS kernel,
                       COUNT(*) AS calls,
                       SUM(d.end - d.start) AS gpu_ns
                FROM {quoted_dispatch} AS d
                JOIN {quoted_symbols} AS s ON s.id = d.kernel_id
                WHERE d.end > d.start AND d.end >= ?
                GROUP BY s.kernel_name
                ORDER BY gpu_ns DESC
                LIMIT 40
                """,
                (cutoff,),
            ).fetchall()
            print(f"TAIL_WINDOW seconds={args.tail_seconds:g} cutoff_ns={cutoff} last_end_ns={last_end}")
            for row in aggregate:
                print(f"KERNEL calls={row['calls']} gpu_ms={row['gpu_ns'] / 1e6:.3f} name={row['kernel']}")
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
