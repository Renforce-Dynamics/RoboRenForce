#!/usr/bin/env python3
"""Download Psi0 (USC-PSI-Lab) datasets from HuggingFace.

Source: https://github.com/physical-superintelligence-lab/Psi0
Hub:    https://huggingface.co/datasets/USC-PSI-Lab/psi-data

Examples:
    # Smallest sim task (~30MB) — good for smoke tests
    python scripts/data/download_psi0_dataset.py \\
        --task G1WholebodyTabletopGraspMP-v0 --split simple

    # Smallest real task (~274MB)
    python scripts/data/download_psi0_dataset.py \\
        --task Hold_lunch_bag_with_both_hands_and_squat_to_put_on_the_coffee_table \\
        --split real

    # All sim tasks
    python scripts/data/download_psi0_dataset.py --task all --split simple

    # List available tasks without downloading
    python scripts/data/download_psi0_dataset.py --list
"""

import argparse
import subprocess
import zipfile
from pathlib import Path

REPO_ID = "USC-PSI-Lab/psi-data"

REAL_TASKS = [
    "Hold_lunch_bag_with_both_hands_and_squat_to_put_on_the_coffee_table",
    "Pick_bottle_and_turn_and_pour_into_cup",
    "Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw",
    "Pull_the_tray_out_of_chips_can_and_throw_the_can_into_trash_bin",
    "Push_cart_grasp_and_place_grapes_on_plate",
    "Put_dumpling_into_blanket_and_turn_around_and_pass_to_human",
    "Remove_the_cap_turn_on_the_faucet_and_fill_the_bottle_with_water",
    "Rotate_to_pour_ham_into_plate_and_push_the_cart_forward",
    "Spray_the_bowl_and_wipe_it_and_stack_it_up",
]

SIMPLE_TASKS = [
    "G1WholebodyBendPickMP-v0",
    "G1WholebodyHandoverTeleop-v0",
    "G1WholebodyLocomotionPickBetweenTablesTeleop-v0",
    "G1WholebodyTabletopGraspMP-v0",
    "G1WholebodyXMoveBendPickTeleop-v0",
    "G1WholebodyXMovePickTeleop-v0",
]

SIMPLE_EVAL_TASKS = ["simple-eval"]


def list_tasks():
    print(f"Repo: {REPO_ID}")
    print(f"\nReal-world tasks ({len(REAL_TASKS)}):")
    for t in REAL_TASKS:
        print(f"  - {t}")
    print(f"\nSimulation tasks ({len(SIMPLE_TASKS)}):")
    for t in SIMPLE_TASKS:
        print(f"  - {t}")


def download_one(task: str, split: str, out_dir: Path, unzip: bool):
    rel = f"{split}/{task}.zip"
    print(f"[download] {REPO_ID}/{rel} -> {out_dir}")
    cmd = [
        "hf", "download", REPO_ID, rel,
        "--repo-type", "dataset",
        "--local-dir", str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    if unzip:
        zip_path = out_dir / rel
        target = out_dir / split
        target.mkdir(parents=True, exist_ok=True)
        print(f"[unzip] {zip_path} -> {target}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default=None,
                    help="Task name (use 'all' to download all tasks for the chosen split)")
    ap.add_argument("--split", choices=["real", "simple", "simple-eval"], default="simple",
                    help="real (G1 teleop) | simple (G1 sim) | simple-eval")
    ap.add_argument("--output-dir", default="data/psi0",
                    help="Local destination root (default: data/psi0)")
    ap.add_argument("--no-unzip", action="store_true", help="Skip unzipping")
    ap.add_argument("--list", action="store_true", help="List tasks and exit")
    args = ap.parse_args()

    if args.list or args.task is None:
        list_tasks()
        if args.task is None:
            print("\n(specify --task <name> or --task all to download)")
        return

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = {"real": REAL_TASKS, "simple": SIMPLE_TASKS, "simple-eval": SIMPLE_EVAL_TASKS}[args.split]
    tasks = pool if args.task == "all" else [args.task]

    for t in tasks:
        if t not in pool and args.split != "simple-eval":
            print(f"[warn] unknown task: {t} (continuing — will fail if not on hub)")
        download_one(t, args.split, out_dir, unzip=not args.no_unzip)

    print(f"\nDone. Data at: {out_dir}/{args.split}/")


if __name__ == "__main__":
    main()
