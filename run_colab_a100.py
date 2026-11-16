"""Autonomous Google Colab A100 runner for Lexsi interpretability suites.

Provisions a Colab session per suite via google-colab-cli, executes notebook
cells sequentially with live log streaming and executed-notebook checkpointing,
enforces a strict compute-unit spend ceiling, downloads all artifacts locally,
and guarantees VM termination in a finally block.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parent
COLAB_EXE = ROOT / ".venv_colab" / "Scripts" / "colab.exe"
ENV_FILE = ROOT / ".env_colab"
OUTPUTS_ROOT = ROOT / "colab_outputs"
LEDGER_PATH = OUTPUTS_ROOT / "budget_ledger.json"

SUITE_NOTEBOOKS: dict[str, Path] = {
    "exp6": ROOT / "exp6_overrefusal_compliance_suite_circuitkit.ipynb",
    "exp7": ROOT / "exp7_instruction_hierarchy_compliance_circuitkit.ipynb",
    "exp8": ROOT / "exp8_contextual_epistemic_faitheval_circuitkit.ipynb",
    "exp9": ROOT / "exp9_normative_ethics_entanglement_circuitkit.ipynb",
    "exp10": ROOT / "exp10_operator_rank_decomposition_circuitkit.ipynb",
    "exp11": ROOT / "exp11_regional_alignment_transfer_circuitkit.ipynb",
    "exp12": ROOT / "exp12_cross_behavior_orthogonality_circuitkit.ipynb",
    "exp13": ROOT / "exp13_probe_gated_steering_circuitkit.ipynb",
}


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def run_colab(
    args: list[str],
    timeout: int = 120,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    cmd = [str(COLAB_EXE), "--auth=oauth2", *args]
    res = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if check and res.returncode != 0:
        raise RuntimeError(
            f"colab command failed ({res.returncode}): {' '.join(args)}\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    return res


def get_usage() -> dict[str, Any]:
    res = run_colab(["usage"], timeout=60, check=True)
    text = (res.stdout or "") + "\n" + (res.stderr or "")
    bal_m = re.search(r"Current balance:\s*([\d.]+)\s*compute units", text)
    rate_m = re.search(r"Usage rate:\s*([\d.]+)/hr", text)
    act_m = re.search(r"Active assignments:\s*(\d+)", text)
    return {
        "balance": float(bal_m.group(1)) if bal_m else None,
        "rate_per_hr": float(rate_m.group(1)) if rate_m else None,
        "active_assignments": int(act_m.group(1)) if act_m else None,
        "raw": text.strip(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def load_or_init_ledger(initial_usage: dict[str, Any], max_cu_spend: float) -> dict[str, Any]:
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    if LEDGER_PATH.exists():
        ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    else:
        ledger = {
            "initial_balance": initial_usage["balance"],
            "max_cu_spend": max_cu_spend,
            "created_at": initial_usage["timestamp"],
            "runs": [],
        }
        LEDGER_PATH.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return ledger


def save_ledger(ledger: dict[str, Any]) -> None:
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(json.dumps(ledger, indent=2), encoding="utf-8")


def spent_cu(ledger: dict[str, Any], current_balance: float | None) -> float:
    init_bal = ledger.get("initial_balance")
    if init_bal is None or current_balance is None:
        return 0.0
    return max(0.0, float(init_bal) - float(current_balance))


def sanitize_cell_source(src: str) -> str:
    lines = []
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("%matplotlib"):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def _exec_once(
    session_name: str,
    tmp_path: Path,
    log_fp: Any,
    timeout_sec: int,
    extra_env: list[str] | None,
) -> tuple[int, str, bool]:
    cmd = [str(COLAB_EXE), "--auth=oauth2", "exec", "-s", session_name, "--timeout", "0", "-f", str(tmp_path)]
    for kv in extra_env or []:
        cmd.extend(["--env", kv])

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    q: queue.Queue[str | None] = queue.Queue()

    def _reader():
        assert proc.stdout is not None
        try:
            for line in iter(proc.stdout.readline, ""):
                q.put(line)
        finally:
            q.put(None)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    chunks: list[str] = []
    start = time.monotonic()
    conn_lost = False
    while True:
        try:
            item = q.get(timeout=1.0)
        except queue.Empty:
            if proc.poll() is not None and not t.is_alive():
                break
            if time.monotonic() - start > timeout_sec:
                proc.kill()
                msg = f"\n[RUNNER ERROR] Cell timed out after {timeout_sec}s.\n"
                chunks.append(msg)
                log_fp.write(msg)
                log_fp.flush()
                return 124, "".join(chunks), False
            if conn_lost and time.monotonic() - start > 15:
                proc.kill()
                return 1, "".join(chunks), True
            continue

        if item is None:
            break
        chunks.append(item)
        sys.stdout.write(item)
        sys.stdout.flush()
        log_fp.write(item)
        log_fp.flush()
        if "Connection was lost" in item:
            conn_lost = True
            # Give CLI up to 3s to flush & exit cleanly, else kill hung websocket thread
            try:
                rc = proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = 1
            return rc, "".join(chunks), True

    rc = proc.wait()
    return rc, "".join(chunks), conn_lost


def exec_python_in_session(
    session_name: str,
    code: str,
    log_fp: Any,
    timeout_sec: int = 2400,
    extra_env: list[str] | None = None,
    max_reconnects: int = 3,
) -> tuple[int, str]:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        encoding="utf-8",
        delete=False,
    ) as tmp:
        tmp.write(code)
        tmp_path = Path(tmp.name)

    try:
        for attempt in range(1, max_reconnects + 1):
            rc, out, conn_lost = _exec_once(session_name, tmp_path, log_fp, timeout_sec, extra_env)
            if not conn_lost:
                return rc, out
            if attempt < max_reconnects:
                msg = f"\n[runner] Websocket connection dropped; sleeping 6s and retrying (attempt {attempt + 1}/{max_reconnects})...\n"
                print(msg, end="", flush=True)
                log_fp.write(msg)
                log_fp.flush()
                time.sleep(6)
        return rc, out
    finally:
        tmp_path.unlink(missing_ok=True)


def provision_session(session_name: str, preferred_gpu: str = "A100") -> str:
    gpus = [preferred_gpu] + (["L4"] if preferred_gpu != "L4" else [])
    for gpu in gpus:
        for attempt in range(1, 3):
            print(f"[runner] Provisioning session '{session_name}' with --gpu {gpu} (attempt {attempt})...", flush=True)
            res = run_colab(["new", "-s", session_name, "--gpu", gpu], timeout=180, check=False)
            out = (res.stdout or "") + "\n" + (res.stderr or "")
            print(out.strip(), flush=True)
            if res.returncode == 0:
                return gpu
            time.sleep(10)
    raise RuntimeError(f"Failed to provision Colab GPU session '{session_name}' across {gpus}")


def stop_session_safely(session_name: str) -> None:
    for attempt in range(1, 4):
        print(f"[runner] Stopping session '{session_name}' (attempt {attempt})...", flush=True)
        res = run_colab(["stop", "-s", session_name], timeout=90, check=False)
        out = ((res.stdout or "") + " " + (res.stderr or "")).strip()
        if out:
            print(out, flush=True)
        if res.returncode == 0 or "not found" in out.lower():
            return
        time.sleep(5)


def run_single_notebook(
    suite_key: str,
    nb_path: Path,
    hf_token: str,
    preferred_gpu: str,
    cell_timeout: int,
    ledger: dict[str, Any],
    max_cu_spend: float,
    resume_session: str | None = None,
    start_cell: int = 0,
) -> dict[str, Any]:
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])

    # Resolve SUITE_ID from Cell 1 if available, else fall back to suite_key
    suite_id = suite_key
    for c in cells:
        if c.get("cell_type") == "code":
            src = "".join(c.get("source", []))
            m = re.search(r'SUITE_ID\s*=\s*"([^"]+)"', src)
            if m:
                suite_id = m.group(1)
                break

    out_dir = OUTPUTS_ROOT / suite_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "execution.log"
    executed_nb_path = out_dir / f"{nb_path.stem}_executed.ipynb"
    if resume_session and executed_nb_path.exists():
        nb = json.loads(executed_nb_path.read_text(encoding="utf-8"))
        cells = nb.get("cells", [])

    pre_usage = get_usage()
    cur_spent = spent_cu(ledger, pre_usage["balance"])
    print(
        f"\n[runner] === Starting {suite_key} ({suite_id}) | Balance={pre_usage['balance']} CU | Spent={cur_spent:.2f}/{max_cu_spend:.2f} CU ===",
        flush=True,
    )
    if cur_spent >= max_cu_spend - 2.0:
        raise RuntimeError(f"Compute budget cap reached ({cur_spent:.2f} >= {max_cu_spend:.2f} CU).")

    session_name = resume_session or f"lexsi_{suite_key}"
    t0 = time.monotonic()
    actual_gpu = preferred_gpu if resume_session else "unknown"
    status = "FAILED"
    failed_cell = None
    download_ok = False

    log_mode = "a" if resume_session else "w"
    with open(log_path, log_mode, encoding="utf-8") as log_fp:
        log_fp.write(f"=== Run {'resumed' if resume_session else 'started'} {datetime.now(timezone.utc).isoformat()} | {nb_path.name} (start_cell={start_cell}) ===\n")
        try:
            if not resume_session:
                actual_gpu = provision_session(session_name, preferred_gpu=preferred_gpu)

            # Bootstrap runtime env vars and headless matplotlib backend cleanly
            boot_code = (
                "import os, matplotlib\n"
                "os.environ['COLAB_HEADLESS'] = '1'\n"
                "os.environ['MPLBACKEND'] = 'Agg'\n"
                "matplotlib.use('Agg')\n"
                "import torch\n"
                "print('BOOTSTRAP_OK GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')\n"
            )
            rc, _ = exec_python_in_session(
                session_name,
                boot_code,
                log_fp,
                timeout_sec=180,
                extra_env=[f"HF_TOKEN={hf_token}", "COLAB_HEADLESS=1"],
            )
            if rc != 0:
                raise RuntimeError(f"Bootstrap failed with exit code {rc}")

            exec_count = 0
            for idx, cell in enumerate(cells):
                if cell.get("cell_type") != "code":
                    continue
                exec_count += 1
                if idx < start_cell:
                    continue
                raw_src = "".join(cell.get("source", []))
                src = sanitize_cell_source(raw_src)
                banner = f"\n--- [Cell {idx} / Exec #{exec_count}] ---\n"
                print(banner, end="", flush=True)
                log_fp.write(banner)
                log_fp.flush()

                c_t0 = time.monotonic()
                rc, cell_out = exec_python_in_session(
                    session_name,
                    src,
                    log_fp,
                    timeout_sec=cell_timeout,
                )
                c_dt = time.monotonic() - c_t0
                cell["execution_count"] = exec_count
                cell["outputs"] = [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": cell_out.splitlines(keepends=True),
                    }
                ]
                executed_nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
                done_msg = f"[Cell {idx} finished in {c_dt:.1f}s | rc={rc}]\n"
                print(done_msg, end="", flush=True)
                log_fp.write(done_msg)
                log_fp.flush()

                has_tb = (
                    "Traceback (most recent call last)" in cell_out
                    or bool(re.search(r"-{10,}\s*[A-Za-z_]*(Error|Exception)\b", cell_out))
                )
                if rc != 0 or has_tb:
                    failed_cell = idx
                    raise RuntimeError(f"Cell {idx} failed (rc={rc}, has_traceback={has_tb})")

            status = "SUCCESS"
        finally:
            # Always attempt to capture logs & artifacts generated so far before stopping VM
            try:
                tar_code = f"""
import os, tarfile
from pathlib import Path
base = ARTIFACT_DIR if 'ARTIFACT_DIR' in globals() else Path('./lexsi_hackathon_artifacts/{suite_id}')
out_tar = Path('/content/{suite_id}_artifacts.tar.gz')
if base.exists():
    with tarfile.open(out_tar, 'w:gz') as tf:
        tf.add(base, arcname='{suite_id}')
    print('TAR_READY', out_tar, out_tar.stat().st_size)
else:
    print('TAR_SKIPPED_NO_DIR', base)
"""
                rc_tar, _ = exec_python_in_session(session_name, tar_code, log_fp, timeout_sec=180)
                if rc_tar == 0:
                    local_tar = out_dir / f"{suite_id}_artifacts.tar.gz"
                    dl = run_colab(
                        ["download", "-s", session_name, f"/content/{suite_id}_artifacts.tar.gz", str(local_tar)],
                        timeout=300,
                        check=False,
                    )
                    if dl.returncode == 0 and local_tar.exists() and local_tar.stat().st_size > 0:
                        with tarfile.open(local_tar, "r:gz") as tf:
                            if sys.version_info >= (3, 12):
                                tf.extractall(OUTPUTS_ROOT, filter="data")
                            else:
                                tf.extractall(OUTPUTS_ROOT)
                        download_ok = True
                        print(f"[runner] Downloaded & extracted artifacts to {out_dir}", flush=True)
                # Also export session log notebook via colab log
                run_colab(
                    ["log", "-s", session_name, "-o", str(out_dir / "colab_session_export.ipynb")],
                    timeout=90,
                    check=False,
                )
            except Exception as dl_exc:
                print(f"[runner] Warning during artifact harvest: {dl_exc}", flush=True)

            stop_session_safely(session_name)

    elapsed = time.monotonic() - t0
    post_usage = get_usage()
    record = {
        "suite_key": suite_key,
        "suite_id": suite_id,
        "notebook": nb_path.name,
        "gpu": actual_gpu,
        "status": status,
        "failed_cell": failed_cell,
        "download_ok": download_ok,
        "elapsed_sec": round(elapsed, 2),
        "pre_balance": pre_usage["balance"],
        "post_balance": post_usage["balance"],
        "cu_delta": (
            round(pre_usage["balance"] - post_usage["balance"], 3)
            if pre_usage["balance"] is not None and post_usage["balance"] is not None
            else None
        ),
        "total_cu_spent": round(spent_cu(ledger, post_usage["balance"]), 3),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    ledger["runs"].append(record)
    save_ledger(ledger)
    print(f"[runner] Summary for {suite_key}: {json.dumps(record)}", flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Lexsi interpretability suites on Colab A100.")
    parser.add_argument("--suites", nargs="+", default=["exp6", "exp7", "exp8", "exp9"])
    parser.add_argument("--gpu", default="A100", choices=["A100", "L4", "T4"])
    parser.add_argument("--max-cu", type=float, default=120.0)
    parser.add_argument("--cell-timeout", type=int, default=2400)
    parser.add_argument("--resume-session", default=None)
    parser.add_argument("--start-cell", type=int, default=0)
    args = parser.parse_args()

    if not COLAB_EXE.exists():
        print(f"ERROR: colab.exe missing at {COLAB_EXE}", file=sys.stderr)
        return 2

    env = load_env_file(ENV_FILE)
    hf_token = env.get("HF_TOKEN", "")
    if not hf_token:
        print("ERROR: HF_TOKEN missing in .env_colab", file=sys.stderr)
        return 2

    usage0 = get_usage()
    ledger = load_or_init_ledger(usage0, args.max_cu)
    any_failed = False

    for i_s, suite_key in enumerate(args.suites):
        nb_path = SUITE_NOTEBOOKS.get(suite_key) or (ROOT / suite_key)
        try:
            rec = run_single_notebook(
                suite_key=suite_key,
                nb_path=Path(nb_path),
                hf_token=hf_token,
                preferred_gpu=args.gpu,
                cell_timeout=args.cell_timeout,
                ledger=ledger,
                max_cu_spend=args.max_cu,
                resume_session=args.resume_session if i_s == 0 else None,
                start_cell=args.start_cell if i_s == 0 else 0,
            )
            if rec["status"] != "SUCCESS":
                any_failed = True
        except Exception as exc:
            any_failed = True
            print(f"[runner] Suite {suite_key} aborted: {exc}", file=sys.stderr, flush=True)

    # Safety check: verify zero active sessions remain
    sess = run_colab(["sessions"], timeout=60, check=False)
    print("[runner] Active Colab sessions after run:\n" + ((sess.stdout or "") + (sess.stderr or "")), flush=True)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
