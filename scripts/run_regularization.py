"""Authorized 20-seed ablation runner with local logs and disjoint seed workers."""

import argparse
import fcntl
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "4"
os.umask(0o077)
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import torch  # noqa: E402

from stageworld.regularization_workflow import run_study  # noqa: E402
from stageworld.regularization_spec import arms  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--action", choices=("prepare", "run", "evaluate"), default="run")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--worker-seeds")
    parser.add_argument("--arm", choices=["all", *[arm["name"] for arm in arms()]],
                        default=os.environ.get("GENERATED651_ARM", "bs4_baseline"))
    args = parser.parse_args()
    os.environ["GENERATED651_ARM"] = args.arm
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("Authorized formal and real smoke require CUDA")
    torch.cuda.set_per_process_memory_fraction(0.18)
    root = PROJECT / "artifacts" / f"{args.arm}-{args.mode}"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    seeds = [int(s) for s in args.worker_seeds.split(",")] if args.worker_seeds else None
    lock_name = "run.lock" if seeds is None else "worker-" + "-".join(map(str, seeds)) + ".lock"
    with (root / lock_name).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_study(
            PROJECT,
            root,
            smoke=args.mode == "smoke",
            action=args.action,
            workers=args.workers,
            worker_seeds=seeds,
        )
    print(
        {k: result[k] for k in ("status", "seconds", "seeds", "configurations") if k in result},
        flush=True,
    )


if __name__ == "__main__":
    main()
