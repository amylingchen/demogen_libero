"""Read one LIBERO/robomimic HDF5 demo into (state, action, init_state)."""
import os
from dataclasses import dataclass

import h5py
import numpy as np

from libero.libero import get_libero_path


def resolve_bddl_file(bddl_file_name: str) -> str:
    """The hdf5-stored bddl path is relative to a different repo layout than
    ours; re-resolve it against the installed LIBERO's bddl_files root using
    its task-category subdir + basename."""
    parts = bddl_file_name.replace("\\", "/").split("/")
    category, filename = parts[-2], parts[-1]
    resolved = os.path.join(get_libero_path("bddl_files"), category, filename)
    assert os.path.exists(resolved), f"bddl file not found: {resolved}"
    return resolved


@dataclass
class SourceDemo:
    demo_key: str
    state: np.ndarray       # (T, 3) absolute EE position
    action: np.ndarray      # (T, 7) OSC_POSE delta action [dx,dy,dz,dax,day,daz,grip]
    init_state: np.ndarray  # (110,) full sim state for env.sim.set_state_from_flattened
    bddl_file: str


def load_demo(hdf5_path: str, demo_key: str) -> SourceDemo:
    with h5py.File(hdf5_path, "r") as f:
        bddl_file = resolve_bddl_file(f["data"].attrs["bddl_file_name"])
        demo = f["data"][demo_key]
        state = np.array(demo["obs"]["ee_pos"], dtype=np.float64)
        action = np.array(demo["actions"], dtype=np.float64)
        init_state = np.array(demo["states"][0], dtype=np.float64)
    return SourceDemo(
        demo_key=demo_key,
        state=state,
        action=action,
        init_state=init_state,
        bddl_file=bddl_file,
    )


def list_demo_keys(hdf5_path: str) -> list[str]:
    with h5py.File(hdf5_path, "r") as f:
        keys = list(f["data"].keys())
    keys.sort(key=lambda k: int(k.split("_")[1]))
    return keys
