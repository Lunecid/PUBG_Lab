"""Wait for fixed jobs then run frozen evaluations and presentation checks once."""

# Support explicit direct-file execution as well as package entry points.
if not __package__:
    import sys as _pkg_sys
    from pathlib import Path as _PackagePath

    _workspace = _PackagePath(__file__).resolve().parents[3]
    _pkg_sys.path.insert(0, str(_workspace))
    _pkg_sys.path.insert(0, str(_workspace / "src"))

from pubg_lab.common.paths import REPO_ROOT
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = REPO_ROOT
OUT = ROOT / "reports/heterograph_diffusion_experiment_v4"
expected = (
    [
        f"{v}_{k}_{s}"
        for v in ["observer", "server"]
        for k in ["regression", "diffusion"]
        for s in [41, 42, 43]
    ]
    + [f"observer_independent_{s}" for s in [41, 42, 43]]
    + ["observer_regression_41_uniform"]
)
start = time.monotonic()
while not all((OUT / (name + ".pt")).exists() for name in expected):
    for p in OUT.glob("run_*.log"):
        if "Traceback (most recent call last)" in p.read_text():
            raise RuntimeError(f"Training failed: {p.name}")
    if time.monotonic() - start > 7200:
        raise TimeoutError("Training not complete")
    time.sleep(20)
env = os.environ.copy()
env.update(
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="2",
    MPLCONFIGDIR="/private/tmp/pubg-mpl",
    XDG_CACHE_HOME="/private/tmp/pubg-cache",
)
for file, args in [
    ("experiment_v4.py", ["evaluate"]),
    ("analyze_v4.py", []),
    ("visualize_v4.py", []),
    ("report_v4.py", []),
    ("finalize_v4.py", []),
]:
    print("START", file, flush=True)
    subprocess.run(
        [sys.executable, str(ROOT / "tools/movement_diffusion" / file), *args],
        cwd=ROOT,
        env=env,
        check=True,
    )
    print("DONE", file, flush=True)
print("V4_PIPELINE_COMPLETE", flush=True)
