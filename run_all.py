#!/usr/bin/env python3
"""
Reproduce the whole experiment battery: runs the model self-test, then every wf_*.py driver,
capturing each script's stdout to workflow-results/<name>.out.txt and (re)writing its
workflow-results/<name>.result.json. Compare your fresh outputs to the committed ones with
`git diff` to verify the published numbers.

Requires pulp + numpy:
    pip install -r requirements.txt && python run_all.py
or, with uv (no venv needed):
    uv run --with pulp --with numpy run_all.py

CBC is deterministic (fixed seeds), so a faithful re-run reproduces the committed results
bit-for-bit on the same solver build.
"""
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "workflow-results")
os.makedirs(OUT, exist_ok=True)


def run(script):
    print(f"\n{'=' * 72}\n{script}\n{'=' * 72}", flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, script)],
                       cwd=HERE, capture_output=True, text=True)
    body = r.stdout + (("\n[stderr]\n" + r.stderr) if r.stderr.strip() else "")
    print(body)
    if script.startswith("wf_"):
        with open(os.path.join(OUT, os.path.splitext(script)[0] + ".out.txt"), "w",
                  encoding="utf-8") as f:
            f.write(body)
    print(f"({time.time() - t:.0f}s, exit {r.returncode})", flush=True)
    return r.returncode


if __name__ == "__main__":
    print("### MODEL SELF-TEST (asserts the formulation controls) ###")
    if run("model.py") != 0:
        print("Self-test failed; aborting.")
        sys.exit(1)
    drivers = sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "wf_*.py")))
    print(f"\n### Running {len(drivers)} experiment drivers ###")
    failures = [d for d in drivers if run(d) != 0]
    print("\nDone. See workflow-results/ for raw outputs and FINDINGS-WORKFLOW.md / "
          "FINDINGS-ADDENDUM.md for the analysis.")
    if failures:
        print("Drivers that returned non-zero:", ", ".join(failures))
