#!/usr/bin/env python3
"""Local compatibility matrix runner for MERaLiON2 plugin.

This publishable copy lives outside private scripts so it can be versioned.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def slugify(version: str) -> str:
    return version.replace(".", "_").replace("-", "_")


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate repository root (pyproject.toml missing).")


def append_log(log_file: Path, message: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{now_utc()}] {message}\n")


def run_cmd(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    log_file: Path,
    timeout: int | None = None,
) -> tuple[int, str]:
    rendered = " ".join(shlex.quote(part) for part in cmd)
    append_log(log_file, f"CMD: {rendered}")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    output = proc.stdout or ""
    with log_file.open("a", encoding="utf-8") as f:
        f.write(output)
        f.write(f"\n[{now_utc()}] EXIT: {proc.returncode}\n")
    return proc.returncode, output


def extract_error_signature(text: str) -> str:
    patterns = [
        r"ValueError: 'aimv2' is already used by a Transformers config",
        r"AttributeError: .*",
        r"ImportError: .*",
        r"ModuleNotFoundError: .*",
        r"RuntimeError: .*",
        r"AssertionError: .*",
        r"FAILED .*",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1][:240] if lines else "unknown-error"


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def wait_port_closed(host: str, port: int, timeout_s: int = 12, poll_s: float = 0.5) -> bool:
    end_time = time.time() + timeout_s
    while time.time() < end_time:
        if not is_port_open(host, port):
            return True
        time.sleep(poll_s)
    return not is_port_open(host, port)


def _listener_pids(port: int) -> list[int]:
    pids: set[int] = set()
    if shutil.which("lsof"):
        proc = subprocess.run(
            ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        pids.update(int(x) for x in proc.stdout.split() if x.isdigit())
    if not pids and shutil.which("fuser"):
        proc = subprocess.run(
            ["fuser", "-n", "tcp", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        pids.update(int(x) for x in re.findall(r"\b\d+\b", proc.stdout))
    return sorted(pids)


def force_cleanup_port(port: int, log_file: Path) -> None:
    pids = _listener_pids(port)
    if not pids:
        return
    append_log(log_file, f"Force cleanup port {port} pids={pids}")
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(1.5)
        if not _listener_pids(port):
            return


def wait_server_ready(base_url: str, health_path: str, timeout_s: int, poll_s: int) -> tuple[bool, str]:
    end_time = time.time() + timeout_s
    url = base_url.rstrip("/") + health_path
    reason = "server not ready"
    while time.time() < end_time:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True, "ready"
                reason = f"status={resp.status}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            reason = f"url-error: {e}"
        time.sleep(poll_s)
    return False, reason


def terminate_group(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    for sig, wait_s in ((signal.SIGTERM, 8), (signal.SIGKILL, 2)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        end = time.time() + wait_s
        while time.time() < end:
            if process.poll() is not None:
                return
            time.sleep(0.2)


def build_summary(payload: dict[str, Any]) -> str:
    rows = payload.get("results", [])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["vllm_version"], []).append(row)

    lines = ["# Local Compatibility Summary", "", f"Generated: `{payload.get('generated_at')}`", ""]
    for vllm_ver in sorted(grouped):
        entries = sorted(grouped[vllm_ver], key=lambda x: x["transformers_version"])
        lines.append(f"## vLLM {vllm_ver}")
        lines.append("")
        lines.append("| transformers | install | general | latency | asr | overall | signature |")
        lines.append("|---|---|---|---|---|---|---|")
        for e in entries:
            modes = e.get("modes", {})
            install = "PASS" if e.get("install", {}).get("ok") else "FAIL"
            general = "PASS" if modes.get("general", {}).get("ok") else "FAIL"
            latency = "PASS" if modes.get("latency", {}).get("ok") else "FAIL"
            asr = "PASS" if modes.get("asr", {}).get("ok") else "FAIL"
            overall = "PASS" if e.get("overall_ok") else "FAIL"
            sig = (e.get("error_signature", "") or "").replace("|", "\\|")
            lines.append(
                f"| {e['transformers_version']} | {install} | {general} | {latency} | {asr} | {overall} | {sig} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def default_env(repo_root: Path, venv: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = f"{venv / 'bin'}:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("VLLM_TEST_BASE_URL", "http://localhost:8063/v1")
    return env


def run_mode(
    repo_root: Path,
    combo_dir: Path,
    mode: str,
    serve_script: str,
    tests: list[str],
    env: dict[str, str],
    server_cfg: dict[str, Any],
) -> dict[str, Any]:
    mode_log = combo_dir / f"{mode}.log"
    data = {
        "ok": False,
        "started_at": now_utc(),
        "ended_at": None,
        "error_signature": "",
        "startup_ready": False,
        "pytest_exit_code": None,
        "serve_script": serve_script,
        "pytest_targets": tests,
    }
    base_url = str(server_cfg["base_url"])
    health_path = str(server_cfg["health_path"])
    timeout_s = int(server_cfg.get("startup_timeout_seconds", 180))
    poll_s = int(server_cfg.get("poll_interval_seconds", 2))
    host = base_url.split("://", 1)[-1].split("/", 1)[0].split(":", 1)[0]
    port = int(base_url.rsplit(":", 1)[1].split("/", 1)[0])

    if is_port_open(host, port):
        force_cleanup_port(port, mode_log)
        if not wait_port_closed(host, port, timeout_s=8):
            data["error_signature"] = f"port {port} already in use before mode {mode}"
            data["ended_at"] = now_utc()
            return data

    proc: subprocess.Popen[Any] | None = None
    try:
        append_log(mode_log, f"MODE START {mode}")
        append_log(mode_log, f"SERVE CMD: bash {serve_script}")
        with mode_log.open("a", encoding="utf-8") as f:
            proc = subprocess.Popen(
                ["bash", serve_script],
                cwd=str(repo_root),
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        ready, reason = wait_server_ready(base_url, health_path, timeout_s, poll_s)
        data["startup_ready"] = ready
        if not ready:
            data["error_signature"] = f"startup-timeout: {reason}"
            return data
        cmd = [str(Path(env["VIRTUAL_ENV"]) / "bin" / "python"), "-m", "pytest", "-q", *tests]
        rc, out = run_cmd(cmd, cwd=repo_root, env=env, log_file=mode_log, timeout=7200)
        data["pytest_exit_code"] = rc
        if rc != 0:
            data["error_signature"] = extract_error_signature(out)
            return data
        data["ok"] = True
        return data
    finally:
        terminate_group(proc)
        if not wait_port_closed(host, port, timeout_s=8):
            force_cleanup_port(port, mode_log)
            wait_port_closed(host, port, timeout_s=8)
        data["ended_at"] = now_utc()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run vLLM/transformers local matrix.")
    parser.add_argument("--config", default="scripts/compatibility/version_matrix_candidates.yaml")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--venvs-dir", default=".venvs/matrix")
    parser.add_argument("--only-vllm", nargs="*", default=[])
    parser.add_argument("--only-transformers", nargs="*", default=[])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    repo_root = find_repo_root(Path(__file__).resolve())
    config_path = (repo_root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    artifacts_dir = (repo_root / args.artifacts_dir).resolve()
    logs_dir = artifacts_dir / "logs"
    results_path = artifacts_dir / "matrix_results.json"
    summary_path = artifacts_dir / "matrix_summary.md"
    venvs_dir = (repo_root / args.venvs_dir).resolve()

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    vllm_versions: list[str] = cfg["vllm_versions"]
    tf_map: dict[str, list[str]] = cfg["transformers_candidates"]
    mode_order: list[str] = cfg["mode_order"]
    serve_scripts: dict[str, str] = cfg["serve_scripts"]
    pytest_by_mode: dict[str, list[str]] = cfg["pytest_by_mode"]
    server_cfg: dict[str, Any] = cfg["server"]
    install_cfg: dict[str, Any] = cfg["install"]

    if args.only_vllm:
        allow = set(args.only_vllm)
        vllm_versions = [x for x in vllm_versions if x in allow]

    existing: list[dict[str, Any]] = []
    if args.resume and results_path.exists():
        existing = json.loads(results_path.read_text(encoding="utf-8")).get("results", [])
    done = {(r["vllm_version"], r["transformers_version"]) for r in existing}
    results = list(existing)

    for vllm_ver in vllm_versions:
        tf_candidates = tf_map.get(vllm_ver, [])
        if args.only_transformers:
            allow_tf = set(args.only_transformers)
            tf_candidates = [x for x in tf_candidates if x in allow_tf]

        for tf_ver in tf_candidates:
            if args.resume and (vllm_ver, tf_ver) in done:
                continue
            combo_dir = logs_dir / f"vllm_{slugify(vllm_ver)}" / f"transformers_{slugify(tf_ver)}"
            install_log = combo_dir / "install.log"
            combo = {
                "vllm_version": vllm_ver,
                "transformers_version": tf_ver,
                "started_at": now_utc(),
                "ended_at": None,
                "install": {"ok": False, "error_signature": ""},
                "modes": {},
                "overall_ok": False,
                "error_signature": "",
            }

            venv = venvs_dir / f"vllm_{slugify(vllm_ver)}__tf_{slugify(tf_ver)}"
            py = venv / "bin" / "python"
            if not py.exists():
                rc, out = run_cmd([sys.executable, "-m", "venv", str(venv)], repo_root, os.environ.copy(), install_log)
                if rc != 0:
                    combo["install"]["error_signature"] = extract_error_signature(out)
                    combo["error_signature"] = combo["install"]["error_signature"]
                    combo["ended_at"] = now_utc()
                    results.append(combo)
                    continue
            env = default_env(repo_root, venv)
            install_cmds = [
                [str(py), "-m", "pip", "install", "--upgrade", "pip"],
                [str(py), "-m", "pip", "install", f"vllm=={vllm_ver}", f"transformers=={tf_ver}"],
                [str(py), "-m", "pip", "install", *shlex.split(str(install_cfg.get("plugin_install", "-e .")))],
                [str(py), "-m", "pip", "install", *list(install_cfg.get("extra_packages", []))],
            ]
            install_ok = True
            for cmd in install_cmds:
                rc, out = run_cmd(cmd, repo_root, env, install_log, timeout=1200)
                if rc != 0:
                    install_ok = False
                    combo["install"]["error_signature"] = extract_error_signature(out)
                    combo["error_signature"] = combo["install"]["error_signature"]
                    break
            combo["install"]["ok"] = install_ok
            if install_ok:
                all_ok = True
                for mode in mode_order:
                    mode_result = run_mode(
                        repo_root=repo_root,
                        combo_dir=combo_dir,
                        mode=mode,
                        serve_script=serve_scripts[mode],
                        tests=pytest_by_mode[mode],
                        env=env,
                        server_cfg=server_cfg,
                    )
                    combo["modes"][mode] = mode_result
                    if not mode_result["ok"]:
                        all_ok = False
                        combo["error_signature"] = mode_result["error_signature"]
                        break
                combo["overall_ok"] = all_ok

            combo["ended_at"] = now_utc()
            results.append(combo)
            payload = {"generated_at": now_utc(), "config_path": str(config_path), "results": results}
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            summary_path.write_text(build_summary(payload), encoding="utf-8")

    payload = {"generated_at": now_utc(), "config_path": str(config_path), "results": results}
    results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(build_summary(payload), encoding="utf-8")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
