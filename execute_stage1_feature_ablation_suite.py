import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SEEDS = (20260616, 20260617, 20260618)


def parse_args():
    parser = argparse.ArgumentParser("Execute the frozen Stage1 feature-ablation suite")
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--val_records_csv", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "execution_state.json"
    state = {
        "started_utc": utc_now(),
        "status": "running",
        "seeds": list(SEEDS),
        "conditions": ["rgb", "dust3r"],
        "runs": [],
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    run_dirs = {"rgb": [], "dust3r": []}
    for condition in ("rgb", "dust3r"):
        for seed in SEEDS:
            run_dir = output_root / f"{condition}_seed_{seed}"
            run_dirs[condition].append(str(run_dir))
            summary_path = run_dir / "summary.json"
            if args.resume and summary_path.exists():
                state["runs"].append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "status": "reused_complete",
                        "run_dir": str(run_dir),
                    }
                )
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                continue
            command = [
                sys.executable,
                "run_stage1_feature_ablation.py",
                "--condition",
                condition,
                "--root_dir",
                args.root_dir,
                "--weights_path",
                args.weights_path,
                "--val_records_csv",
                args.val_records_csv,
                "--save_dir",
                str(run_dir),
                "--train_size",
                "2048",
                "--num_epochs",
                "20",
                "--batch_size",
                str(args.batch_size),
                "--seed",
                str(seed),
            ]
            log_path = output_root / f"{condition}_seed_{seed}.log"
            row = {
                "condition": condition,
                "seed": seed,
                "status": "running",
                "started_utc": utc_now(),
                "run_dir": str(run_dir),
                "log_path": str(log_path),
                "command": command,
            }
            state["runs"].append(row)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            row["returncode"] = completed.returncode
            row["completed_utc"] = utc_now()
            row["status"] = "complete" if completed.returncode == 0 else "failed"
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            if completed.returncode != 0:
                state["status"] = "failed"
                state["failed_condition"] = condition
                state["failed_seed"] = seed
                state["completed_utc"] = utc_now()
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                raise SystemExit(completed.returncode)

    summary_dir = output_root / "summary"
    summary_command = [
        sys.executable,
        "summarize_stage1_feature_ablation.py",
        "--dust3r_dirs",
        *run_dirs["dust3r"],
        "--rgb_dirs",
        *run_dirs["rgb"],
        "--output_dir",
        str(summary_dir),
    ]
    summary_log = output_root / "summary.log"
    with summary_log.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            summary_command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    state["summary_returncode"] = completed.returncode
    state["summary_dir"] = str(summary_dir)
    state["completed_utc"] = utc_now()
    state["status"] = "complete" if completed.returncode == 0 else "summary_failed"
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
