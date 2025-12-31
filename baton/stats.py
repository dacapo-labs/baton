"""Baton Stats - Analyze judge decisions and routing patterns."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def main():
    """Main CLI entrypoint for stats."""
    parser = argparse.ArgumentParser(
        prog="baton-stats",
        description="Analyze Baton logs for judge decisions and routing patterns",
    )

    parser.add_argument(
        "--log-dir",
        default="/data/baton/logs",
        help="Log directory (default: /data/baton/logs)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to analyze (default: 7)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("summary", help="Show overall summary")

    judge_parser = subparsers.add_parser("judge", help="Analyze judge decisions")
    judge_parser.add_argument(
        "--by-query-type",
        action="store_true",
        help="Group by query type",
    )

    subparsers.add_parser("models", help="Model usage statistics")
    subparsers.add_parser("errors", help="Error analysis")
    subparsers.add_parser("winners", help="Show model win rates from judge decisions")

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        sys.exit(1)

    entries = load_logs(log_dir, args.days)

    if args.command == "summary":
        show_summary(entries)
    elif args.command == "judge":
        show_judge_analysis(entries, args.by_query_type)
    elif args.command == "models":
        show_model_stats(entries)
    elif args.command == "errors":
        show_error_analysis(entries)
    elif args.command == "winners":
        show_winners(entries)
    else:
        parser.print_help()
        sys.exit(1)


def load_logs(log_dir: Path, days: int) -> list[dict[str, Any]]:
    """Load log entries from the last N days."""
    entries = []
    cutoff = datetime.now() - timedelta(days=days)

    for log_file in sorted(log_dir.glob("baton-*.jsonl")):
        try:
            date_str = log_file.stem.replace("baton-", "")
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
            if file_date < cutoff:
                continue
        except ValueError:
            continue

        with open(log_file) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue

    return entries


def show_summary(entries: list[dict[str, Any]]):
    """Show overall summary."""
    requests = [e for e in entries if e.get("type") == "request"]
    responses = [e for e in entries if e.get("type") == "response"]
    errors = [e for e in entries if e.get("type") == "error"]
    judge_decisions = [e for e in entries if e.get("type") == "judge_decision"]
    fanouts = [e for e in entries if e.get("type") == "fanout"]

    print("=== Baton Stats Summary ===\n")
    print(f"Total Requests: {len(requests)}")
    print(f"Total Responses: {len(responses)}")
    print(f"Total Errors: {len(errors)}")
    print(f"Judge Decisions: {len(judge_decisions)}")
    print(f"Fan-out Operations: {len(fanouts)}")

    if responses:
        latencies = [r.get("latency_ms", 0) for r in responses]
        avg_latency = sum(latencies) / len(latencies)
        print(f"\nAverage Latency: {avg_latency:.0f}ms")

        total_tokens_in = sum(r.get("tokens_in", 0) or 0 for r in responses)
        total_tokens_out = sum(r.get("tokens_out", 0) or 0 for r in responses)
        print(f"Total Input Tokens: {total_tokens_in:,}")
        print(f"Total Output Tokens: {total_tokens_out:,}")

    zones = defaultdict(int)
    for r in requests:
        zone = r.get("zone") or "unset"
        zones[zone] += 1

    if zones:
        print("\nRequests by Zone:")
        for zone, count in sorted(zones.items(), key=lambda x: -x[1]):
            print(f"  {zone}: {count}")


def show_judge_analysis(entries: list[dict[str, Any]], by_query_type: bool):
    """Show judge decision analysis."""
    decisions = [e for e in entries if e.get("type") == "judge_decision"]

    if not decisions:
        print("No judge decisions found")
        return

    print("=== Judge Decision Analysis ===\n")
    print(f"Total Decisions: {len(decisions)}")

    winners = defaultdict(int)
    for d in decisions:
        winner = d.get("winner", "unknown")
        winners[winner] += 1

    print("\nWinner Distribution:")
    for model, count in sorted(winners.items(), key=lambda x: -x[1]):
        pct = count / len(decisions) * 100
        print(f"  {model}: {count} ({pct:.1f}%)")

    if by_query_type:
        by_type = defaultdict(lambda: defaultdict(int))
        for d in decisions:
            query_type = d.get("query_type") or "unknown"
            winner = d.get("winner", "unknown")
            by_type[query_type][winner] += 1

        print("\nWinners by Query Type:")
        for query_type, model_counts in sorted(by_type.items()):
            print(f"\n  {query_type}:")
            for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
                print(f"    {model}: {count}")


def show_model_stats(entries: list[dict[str, Any]]):
    """Show model usage statistics."""
    responses = [e for e in entries if e.get("type") == "response"]

    if not responses:
        print("No responses found")
        return

    print("=== Model Usage Statistics ===\n")

    by_model = defaultdict(lambda: {"count": 0, "tokens_in": 0, "tokens_out": 0, "latency": []})
    for r in responses:
        model = r.get("model", "unknown")
        by_model[model]["count"] += 1
        by_model[model]["tokens_in"] += r.get("tokens_in", 0) or 0
        by_model[model]["tokens_out"] += r.get("tokens_out", 0) or 0
        if r.get("latency_ms"):
            by_model[model]["latency"].append(r["latency_ms"])

    for model, stats in sorted(by_model.items(), key=lambda x: -x[1]["count"]):
        avg_latency = sum(stats["latency"]) / len(stats["latency"]) if stats["latency"] else 0
        print(f"{model}:")
        print(f"  Requests: {stats['count']}")
        print(f"  Tokens In: {stats['tokens_in']:,}")
        print(f"  Tokens Out: {stats['tokens_out']:,}")
        print(f"  Avg Latency: {avg_latency:.0f}ms")
        print()


def show_error_analysis(entries: list[dict[str, Any]]):
    """Show error analysis."""
    errors = [e for e in entries if e.get("type") == "error"]

    if not errors:
        print("No errors found")
        return

    print("=== Error Analysis ===\n")
    print(f"Total Errors: {len(errors)}")

    by_type = defaultdict(int)
    for e in errors:
        error_type = e.get("error_type", "unknown")
        by_type[error_type] += 1

    print("\nBy Error Type:")
    for error_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {error_type}: {count}")

    by_model = defaultdict(int)
    for e in errors:
        model = e.get("model") or "unknown"
        by_model[model] += 1

    print("\nBy Model:")
    for model, count in sorted(by_model.items(), key=lambda x: -x[1]):
        print(f"  {model}: {count}")


def show_winners(entries: list[dict[str, Any]]):
    """Show model win rates from judge decisions."""
    decisions = [e for e in entries if e.get("type") == "judge_decision"]

    if not decisions:
        print("No judge decisions found")
        return

    print("=== Model Win Rates ===\n")

    appearances = defaultdict(int)
    wins = defaultdict(int)

    for d in decisions:
        winner = d.get("winner")
        candidates = d.get("candidates", [])

        for c in candidates:
            model = c.get("model")
            if model:
                appearances[model] += 1
                if model == winner:
                    wins[model] += 1

    print(f"{'Model':<40} {'Appearances':>12} {'Wins':>8} {'Win Rate':>10}")
    print("-" * 72)

    for model in sorted(appearances.keys(), key=lambda m: -wins[m]):
        app = appearances[model]
        w = wins[model]
        rate = w / app * 100 if app > 0 else 0
        print(f"{model:<40} {app:>12} {w:>8} {rate:>9.1f}%")


if __name__ == "__main__":
    main()
