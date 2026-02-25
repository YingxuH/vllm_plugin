#!/usr/bin/env python3
"""Generate compatibility markdown summary from artifacts/logs only."""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path
from typing import Any


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_version_from_dir(name: str, prefix: str) -> str:
    value = name[len(prefix):]
    return value.replace("_", ".")


def parse_last_exit(text: str) -> int | None:
    matches = re.findall(r"EXIT:\s*(\d+)", text)
    if not matches:
        return None
    return int(matches[-1])


def extract_signature(text: str) -> str:
    patterns = [
        r"ValueError: 'aimv2' is already used by a Transformers config, pick another name\.",
        r"AttributeError: .*",
        r"ImportError: .*",
        r"ModuleNotFoundError: .*",
        r"RuntimeError: .*",
        r"AssertionError: .*",
        r"startup-timeout: .*",
        r"FAILED .*",
        r"ERROR .*",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:240] if lines else ""


def mode_status(log_path: Path) -> tuple[str, str]:
    if not log_path.exists():
        return "NA", ""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    last_exit = parse_last_exit(text)
    if last_exit == 0:
        return "PASS", ""
    return "FAIL", extract_signature(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate compatibility markdown from artifacts/logs.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--output", default="artifacts/compatibility_summary.md")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    artifacts_dir = (repo_root / args.artifacts_dir).resolve()
    logs_root = artifacts_dir / "logs"
    output_path = (repo_root / args.output).resolve()

    if not logs_root.exists():
        raise SystemExit(f"Logs directory not found: {logs_root}")

    by_vllm: dict[str, list[dict[str, Any]]] = {}
    for vdir in sorted(logs_root.glob("vllm_*")):
        if not vdir.is_dir():
            continue
        vllm_version = parse_version_from_dir(vdir.name, "vllm_")
        for tdir in sorted(vdir.glob("transformers_*")):
            if not tdir.is_dir():
                continue
            tf_version = parse_version_from_dir(tdir.name, "transformers_")

            install_status, install_sig = mode_status(tdir / "install.log")
            general_status, general_sig = mode_status(tdir / "general.log")
            latency_status, latency_sig = mode_status(tdir / "latency.log")
            asr_status, asr_sig = mode_status(tdir / "asr.log")

            mode_statuses = [s for s in [general_status, latency_status, asr_status] if s != "NA"]
            if install_status == "FAIL":
                overall = "FAIL"
            elif not mode_statuses:
                overall = "INCOMPLETE"
            elif all(s == "PASS" for s in mode_statuses):
                overall = "PASS"
            elif any(s == "FAIL" for s in mode_statuses):
                overall = "FAIL"
            else:
                overall = "INCOMPLETE"

            signature = install_sig or general_sig or latency_sig or asr_sig
            by_vllm.setdefault(vllm_version, []).append(
                {
                    "transformers": tf_version,
                    "install": install_status,
                    "general": general_status,
                    "latency": latency_status,
                    "asr": asr_status,
                    "overall": overall,
                    "signature": signature,
                }
            )

    lines: list[str] = []
    lines.append("# vLLM + Transformers Compatibility (From Artifacts)")
    lines.append("")
    lines.append(f"Generated: `{now_utc()}`")
    lines.append("")
    lines.append("Legend: `PASS` = test mode succeeded, `FAIL` = mode failed, `NA` = mode not run, `INCOMPLETE` = partial run.")
    lines.append("")

    for vllm_version in sorted(by_vllm.keys(), key=lambda s: [int(x) for x in s.split(".")]):
        rows = sorted(by_vllm[vllm_version], key=lambda x: [int(p) for p in x["transformers"].split(".")])
        lines.append(f"## vLLM {vllm_version}")
        lines.append("")
        lines.append("| transformers | install | general | latency | asr | overall | signature |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in rows:
            sig = (row["signature"] or "").replace("|", "\\|")
            lines.append(
                f"| {row['transformers']} | {row['install']} | {row['general']} | {row['latency']} | "
                f"{row['asr']} | {row['overall']} | {sig} |"
            )
        lines.append("")

        full_pass = [r["transformers"] for r in rows if r["overall"] == "PASS"]
        if full_pass:
            lines.append(f"- Fully passing transformers: {', '.join(full_pass)}")
        else:
            lines.append("- No fully passing transformers in current artifacts.")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
