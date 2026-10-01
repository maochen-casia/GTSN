"""Launch training/evaluation in a NEW disposable container, without host installs.

    python3 scripts/docker_run.py train --run-dir /run/user/1016/experiments/baseline
    python3 scripts/docker_run.py evaluate --checkpoint /run/user/1016/experiments/baseline/best.pt
    python3 scripts/docker_run.py generate-recovery --output-dir /run/user/1016/experiments/recovery
    Add --print-command before the subcommand to review the command without execution.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="gtsn-baseline:tsn-1k")
    parser.add_argument("--cpu", action="store_true", help="Run with --device cpu and no GPU request")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("command", choices=("train", "evaluate", "generate-recovery"))
    args, extra = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    data = Path("/run/user/1016/tsn-1k")
    experiments = Path("/run/user/1016/experiments")
    if not data.is_dir() or not experiments.is_dir():
        raise FileNotFoundError("Dataset and experiment mount directories must already exist")
    command = [
        "docker", "run", "--rm", "--init", "--network", "none", "--read-only",
        "--user", f"{os.getuid()}:{os.getgid()}", "--shm-size", "4g",
        "--tmpfs", "/tmp:rw,exec,size=2g",
        "--env", "XDG_CACHE_HOME=/tmp/cache", "--env", "MPLCONFIGDIR=/tmp/matplotlib",
        "--mount", f"type=bind,source={root},target=/workspace,readonly",
        "--mount", f"type=bind,source={data},target={data},readonly",
        "--mount", f"type=bind,source={experiments},target={experiments}",
    ]
    if not args.cpu and args.command != "generate-recovery":
        command.extend(["--gpus", "all"])
    module = {"generate-recovery": "generate_recovery"}.get(args.command, args.command)
    command.extend([args.image, "python", "-m", f"tsn.cli.{module}", *extra])
    if args.cpu and args.command != "generate-recovery":
        command.extend(["--device", "cpu"])
    print(shlex.join(command), flush=True)
    if not args.print_command:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
