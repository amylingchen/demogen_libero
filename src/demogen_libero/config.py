"""Central data-location config.

Source demo HDF5s (the original LIBERO libero_object datasets) are looked up in
DATA_DIR, which defaults to <repo>/data/libero_object and can be overridden
with the DEMOGEN_LIBERO_DATA environment variable:

    set DEMOGEN_LIBERO_DATA=E:\\datasets\\libero_object      (Windows)
    export DEMOGEN_LIBERO_DATA=/data/libero_object          (Linux)

Download the source files there via huggingface_hub
(repo yifengzhu-hf/LIBERO-datasets, path libero_object/*_demo.hdf5) --
see README for a one-liner.
"""
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.environ.get(
    "DEMOGEN_LIBERO_DATA",
    os.path.join(_REPO_ROOT, "data", "libero_object"),
)


def source_hdf5(task_key: str) -> str:
    """Path of a task's source demo file, e.g.
    source_hdf5('pick_up_the_butter_and_place_it_in_the_basket')."""
    return os.path.join(DATA_DIR, f"{task_key}_demo.hdf5")
