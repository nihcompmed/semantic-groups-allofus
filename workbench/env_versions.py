#!/usr/bin/env python3
"""env_versions.py -- the versions of Python and of the libraries the analysis uses on the Workbench. JNO asks for the
software and its version in the Methods, and this folder ships only the scripts: statsmodels, numpy, pandas, scipy and
scikit-learn come from the Workbench's own Python environment.

versions() is imported by the Workbench scripts, which write it into their download's meta file ("versions"). Run on
its own, this script records the current environment in a download of its own.

OUTPUT  screen_out/env_versions/env_versions.json, and screen_out/env_versions_<CDR>.zip

Run:  python3 env_versions.py
"""
import importlib
import json
import os
import platform
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "screen_out")
CDR = os.environ.get("WORKSPACE_CDR", "")
LIBRARIES = ["statsmodels", "numpy", "pandas", "scipy", "sklearn", "google.cloud.bigquery"]


def versions():
    """Python and library versions of the running environment (a library that is not installed is omitted)."""
    v = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            m = importlib.import_module(name)
            v["scikit-learn" if name == "sklearn" else name] = getattr(m, "__version__", "unknown")
        except ImportError:
            pass
    return v


def main():
    out = os.path.join(OUT_DIR, "env_versions")
    os.makedirs(out, exist_ok=True)
    v = {"cdr": CDR, "executable": sys.executable, **versions()}
    json.dump(v, open(os.path.join(out, "env_versions.json"), "w"), indent=2)
    for k, x in v.items():
        print(f"  {k:24s} {x}")
    zpath = os.path.join(OUT_DIR, f"env_versions_{CDR or 'cdr'}.zip")
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(os.path.join(out, "env_versions.json"), arcname="env_versions/env_versions.json")
    print(f"\n[env_versions] download -> {zpath}")


if __name__ == "__main__":
    main()
