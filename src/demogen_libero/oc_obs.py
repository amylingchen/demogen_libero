"""OC-format observation collection: RGB + depth + instance segmentation +
per-frame bounding boxes, matching the layout of output/demo (LIBERO-OC style).

Conventions (reverse-engineered from output/demo):
    - images stored top-down (viewer orientation) -> flip the raw OpenGL obs
    - depth: real metric depth in cm, uint8, clipped to 255 (~2.55 m cap)
    - seg ids: robot = 50, then 60 + 10*i following OBJECT_ORDER
    - robot_states (8) = [ee_pos(3), ee_ori_axis_angle(3), gripper_qpos(2)]
"""
import json
from pathlib import Path

import h5py
import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import get_real_depth_map

from libero.libero.envs import OffScreenRenderEnv

# Only the end-effector counts as "robot" in the seg mask (matches the small
# robot bbox in the reference metainfo); the arm/mount stay background.
ROBOT_INSTANCE_KEYWORDS = ("gripper",)
ROBOT_SEG_ID = 50

# Per-task object order (target=60, basket=70, distractors by 10s), matching
# output/demo/metainfo.json. Loaded from tasks_config.json (all libero_object
# tasks); the salad entry is kept inline as a fallback.
_CONFIG_PATH = Path(__file__).with_name("tasks_config.json")
OBJECT_ORDER = {
    "pick_up_the_salad_dressing_and_place_it_in_the_basket": [
        "salad_dressing_1", "basket_1", "ketchup_1", "alphabet_soup_1",
        "cream_cheese_1", "milk_1", "tomato_sauce_1",
    ],
}
if _CONFIG_PATH.exists():
    _cfg = json.loads(_CONFIG_PATH.read_text())
    OBJECT_ORDER = {t: c["object_order"] for t, c in _cfg.items()}


def load_task_config() -> dict:
    return json.loads(_CONFIG_PATH.read_text())


def display_name(instance_name: str) -> str:
    return instance_name.rsplit("_", 1)[0].replace("_", " ") + " 1"


def make_oc_env(bddl_file: str, height: int = 256, width: int = 256):
    return OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=height,
        camera_widths=width,
        camera_depths=True,
        camera_segmentations="instance",
    )


def build_seg_lut(env, object_order: list[str]) -> np.ndarray:
    """LUT from robosuite instance-seg values (index into instances_to_ids + 1,
    0 = background) to OC seg ids."""
    instances = list(env.env.model.instances_to_ids.keys())
    lut = np.zeros(len(instances) + 2, dtype=np.uint8)
    for idx, inst in enumerate(instances):
        raw = idx + 1
        low = inst.lower()
        if any(k in low for k in ROBOT_INSTANCE_KEYWORDS):
            lut[raw] = ROBOT_SEG_ID
        elif inst in object_order:
            lut[raw] = 60 + 10 * object_order.index(inst)
    return lut


def extract_oc_frame(env, obs, lut: np.ndarray, object_order: list[str] = None,
                     depth_mm: bool = True) -> dict[str, np.ndarray]:
    """Convert one raw LIBERO obs dict into OC-format arrays (flipped upright).

    object_order: instance names (e.g. ["salad_dressing_1", "basket_1", ...]);
    when given, emit per-frame ground-truth poses obj_pos (n,3) and obj_quat
    (n,4, robosuite xyzw) straight from the sim obs -- no state-layout parsing
    needed downstream. Order matches the seg ids (60 + 10*i).

    depth_mm: additionally emit *_depth_mm (uint16, millimeters) -- the same
    rendered depth as the uint8 cm channels but at 1 mm quantization, which is
    effectively lossless for sim depth. The cm channels stay untouched for
    format parity with output/demo.
    """
    def flip(img):
        return np.ascontiguousarray(img[::-1])

    def real_depth_m(depth_norm):
        return get_real_depth_map(env.sim, depth_norm)[..., 0]

    def depth_cm(real_m):
        return np.clip(np.rint(real_m * 100.0), 0, 255).astype(np.uint8)

    def depth_to_mm(real_m):
        return np.clip(np.rint(real_m * 1000.0), 0, 65535).astype(np.uint16)

    def seg_ids(seg_raw):
        return lut[np.clip(seg_raw[..., 0], 0, len(lut) - 1)]

    agent_m = real_depth_m(obs["agentview_depth"])
    hand_m = real_depth_m(obs["robot0_eye_in_hand_depth"])
    ee_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    ee_aa = T.quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    row = {
        "agentview_rgb": flip(obs["agentview_image"]),
        "agentview_depth": flip(depth_cm(agent_m)),
        "agentview_seg": flip(seg_ids(obs["agentview_segmentation_instance"])),
        "eye_in_hand_rgb": flip(obs["robot0_eye_in_hand_image"]),
        "eye_in_hand_depth": flip(depth_cm(hand_m)),
        "eye_in_hand_seg": flip(seg_ids(obs["robot0_eye_in_hand_segmentation_instance"])),
        "ee_pos": ee_pos,
        "ee_ori": ee_aa,
        "ee_states": np.concatenate([ee_pos, ee_aa]),
        "gripper_states": grip,
        "joint_states": np.asarray(obs["robot0_joint_pos"], dtype=np.float64),
        "robot_states": np.concatenate([ee_pos, ee_aa, grip]),
        "states": env.sim.get_state().flatten(),
    }
    if depth_mm:
        row["agentview_depth_mm"] = flip(depth_to_mm(agent_m))
        row["eye_in_hand_depth_mm"] = flip(depth_to_mm(hand_m))
    if object_order is not None:
        row["obj_pos"] = np.array([obs[f"{n}_pos"] for n in object_order], dtype=np.float64)
        row["obj_quat"] = np.array([obs[f"{n}_quat"] for n in object_order], dtype=np.float64)
    return row


def boxes_for_seg(seg: np.ndarray, names: list[str]) -> list[dict]:
    """Per-frame normalized [x, y, w, h] boxes keyed by display name, exactly as
    export_oc_metainfo.py computes them (objects with no visible pixels are
    omitted from that frame's dict)."""
    frames = []
    h, w = int(seg.shape[1]), int(seg.shape[2])
    id_names = [("robot", ROBOT_SEG_ID), *[(name, 60 + 10 * i) for i, name in enumerate(names)]]
    for frame in seg:
        frame_boxes = {}
        for name, seg_id in id_names:
            ys, xs = np.where(frame == seg_id)
            if xs.size == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            frame_boxes[name] = [int(seg_id), [x0 / w, y0 / h, (x1 - x0 + 1) / w, (y1 - y0 + 1) / h]]
        frames.append(frame_boxes)
    return frames


OBS_KEYS = (
    "agentview_rgb", "agentview_depth", "agentview_seg",
    "eye_in_hand_rgb", "eye_in_hand_depth", "eye_in_hand_seg",
    "ee_pos", "ee_ori", "ee_states", "gripper_states", "joint_states",
)
# extra obs written when present in the rollout: lossless mm depth (uint16) and
# per-frame ground-truth object poses (world frame, quat xyzw)
EXTRA_OBS_KEYS = ("agentview_depth_mm", "eye_in_hand_depth_mm", "obj_pos", "obj_quat")

# per-frame phase_id labels (DemoGen two-stage structure; settle frames belong
# to the segment they precede). phase_id[t] in {0,1,2,3} indexes the per-demo
# `phases` attr, which names each stage's ACTION WORD and TARGET OBJECT:
#   0 move  (target = object to pick)   reach toward the object
#   1 grasp (target = object)           incl. pre-grasp correction + regrasps
#   2 move  (target = basket)           transport toward the basket
#   3 place (target = basket)           lower/release + tail hold
PHASE_MAP = {
    0: "motion_1_reach",
    1: "skill_1_grasp",
    2: "motion_2_transport",
    3: "skill_2_place",
}
PHASE_ACTIONS = {0: "move", 1: "grasp", 2: "move", 3: "place"}


def phase_stages(phase_id: np.ndarray, target_name: str, goal_name: str) -> list:
    """Fine-grained stage table for the per-frame phase_id: one entry per
    contiguous phase span with its action word + target object + instruction.
    phase targets: reach/grasp -> the object to pick; transport/place -> goal."""
    target_short = target_name.rsplit(" 1", 1)[0]
    goal_short = goal_name.rsplit(" 1", 1)[0]
    templates = {
        0: ("move", target_name, f"move the gripper to the {target_short}"),
        1: ("grasp", target_name, f"grasp the {target_short}"),
        2: ("move", goal_name, f"carry the {target_short} to the {goal_short}"),
        3: ("place", goal_name, f"place the {target_short} into the {goal_short}"),
    }
    phase_id = np.asarray(phase_id)
    stages = []
    start = 0
    for t in range(1, len(phase_id) + 1):
        if t == len(phase_id) or phase_id[t] != phase_id[start]:
            p = int(phase_id[start])
            action, target, instruction = templates[p]
            stages.append({"phase": p, "action": action, "target": target,
                           "instruction": instruction, "start": int(start), "end": int(t)})
            start = t
    return stages

# Closed-vocabulary subtask annotation (arXiv:2607.06403): 15 primitive actions
# + transit/idle/other. For pick-and-place the grasp/carry/release phases of one
# interaction form a single `move` subtask; boundaries fall at gripper-object
# contact and final release. Regrasp cycles of the same object stay inside the
# one `move` (boundary only on object change, action change, or sustained pause).
SUBTASK_VOCAB = [
    "move", "pour", "push", "pull", "rotate", "open", "close", "detach",
    "fold", "unfold", "wipe", "stir", "cut", "press", "attach",
    "transit", "idle", "other",
]
SUBTASK_IDS = {name: i for i, name in enumerate(SUBTASK_VOCAB)}


def annotate_subtasks(actions: np.ndarray, target_name: str, goal_name: str,
                      post_steps: int = 3):
    """Per-frame closed-vocabulary subtask labels for a pick-and-place episode.

    Returns (subtask_id (T,) int32 indices into SUBTASK_VOCAB, subtasks list).
    Each subtask: {action, object, destination?, instruction, start, end} with
    [start, end) frame spans.
        transit  [0, first gripper close)        object = target (approach goal)
        move     [first close, final open)       object = target, destination = goal
        transit  [final open, T - post_steps)    retreat, empty hand
        idle     [T - post_steps, T)             recorded hold frames
    """
    T = actions.shape[0]
    grip = actions[:, 6]
    closes = np.where((grip[:-1] <= 0) & (grip[1:] > 0))[0] + 1
    opens = np.where((grip[:-1] > 0) & (grip[1:] <= 0))[0] + 1
    assert closes.size > 0 and opens.size > 0, "no grasp cycle to annotate"
    contact = int(closes[0])
    release = int(opens[-1])
    idle_start = max(T - post_steps, release)

    target_short = target_name.rsplit(" 1", 1)[0]
    goal_short = goal_name.rsplit(" 1", 1)[0]
    subtasks = [
        {"action": "transit", "object": target_name, "destination": None,
         "instruction": f"move the gripper to the {target_short}",
         "start": 0, "end": contact},
        {"action": "move", "object": target_name, "destination": goal_name,
         "instruction": f"move the {target_short} into the {goal_short}",
         "start": contact, "end": release},
        {"action": "transit", "object": None, "destination": None,
         "instruction": "retreat the gripper",
         "start": release, "end": idle_start},
        {"action": "idle", "object": None, "destination": None,
         "instruction": "keep the arm stationary",
         "start": idle_start, "end": T},
    ]
    subtasks = [s for s in subtasks if s["end"] > s["start"]]
    sub_id = np.empty(T, dtype=np.int32)
    for s in subtasks:
        sub_id[s["start"]:s["end"]] = SUBTASK_IDS[s["action"]]
    return sub_id, subtasks


def write_oc_demo(hdf5_path: str, demo_name: str, rollout: dict, success: bool,
                  object_order: list[str] = None, subtasks: list = None):
    """Append one episode in output/demo layout: obs/* (rgb/depth/seg + optional
    mm-depth and obj_pos/obj_quat), actions, states, robot_states, rewards,
    dones; per-demo num_samples (+ object_names/instances) attrs and data-level
    num_demos/total attrs.

    subtasks: closed-vocabulary annotation from annotate_subtasks(); writes the
    per-frame `subtask_id` dataset + a `subtasks` JSON attr on the demo group.
    """
    n = int(rollout["actions"].shape[0])
    rewards = np.zeros((n,), dtype=np.uint8)
    dones = np.zeros((n,), dtype=np.uint8)
    if n:
        rewards[-1] = 1 if success else 0
        dones[-1] = 1
    with h5py.File(hdf5_path, "a") as f:
        data = f.require_group("data")
        if demo_name in data:
            del data[demo_name]
        grp = data.create_group(demo_name)
        grp.attrs["num_samples"] = n
        obs_grp = grp.create_group("obs")
        for key in OBS_KEYS:
            obs_grp.create_dataset(key, data=rollout[key], compression="gzip", compression_opts=4)
        for key in EXTRA_OBS_KEYS:
            if key in rollout:
                obs_grp.create_dataset(key, data=rollout[key], compression="gzip", compression_opts=4)
        grp.create_dataset("actions", data=rollout["actions"], compression="gzip", compression_opts=4)
        grp.create_dataset("states", data=rollout["states"], compression="gzip", compression_opts=4)
        grp.create_dataset("robot_states", data=rollout["robot_states"], compression="gzip", compression_opts=4)
        grp.create_dataset("rewards", data=rewards)
        grp.create_dataset("dones", data=dones)
        if "phase_id" in rollout:
            grp.create_dataset("phase_id", data=np.asarray(rollout["phase_id"], dtype=np.int32))
            if object_order is not None:
                grp.attrs["phases"] = json.dumps(phase_stages(
                    rollout["phase_id"], display_name(object_order[0]), display_name(object_order[1])))
        if "subtask_id" in rollout:
            grp.create_dataset("subtask_id", data=np.asarray(rollout["subtask_id"], dtype=np.int32))
        if subtasks is not None:
            grp.attrs["subtasks"] = json.dumps(subtasks)
            grp.attrs["subtask_vocab"] = json.dumps(SUBTASK_VOCAB)
        if object_order is not None:
            grp.attrs["object_names"] = json.dumps([display_name(x) for x in object_order])
            grp.attrs["object_instances"] = json.dumps(list(object_order))
        data.attrs["num_demos"] = len(data.keys())
        data.attrs["total"] = int(sum(data[k].attrs["num_samples"] for k in data.keys()))


def metainfo_entry(task_key: str, rollout: dict, object_order: list[str],
                   init_state: np.ndarray, subtasks: list = None) -> dict:
    names = [display_name(x) for x in object_order]
    entry_subtasks = {"subtasks": subtasks} if subtasks is not None else {}
    if "phase_id" in rollout:
        entry_subtasks["phases"] = phase_stages(rollout["phase_id"], names[0], names[1])
    return {
        **entry_subtasks,
        "success": True,
        "initial_state": np.asarray(init_state, dtype=float).tolist(),
        "task_nouns": ["robot", names[0], "basket 1"],
        "task_description": task_key.replace("_", " "),
        "target_object": names[0],                          # object to pick
        "goal_object": names[1] if len(names) > 1 else "basket 1",  # where to place
        "object_names": names,
        "exo_boxes": boxes_for_seg(rollout["agentview_seg"], names),
        "ego_boxes": boxes_for_seg(rollout["eye_in_hand_seg"], names),
    }


def write_metainfo(path: str, task_key: str, demo_name: str, rollout: dict,
                   object_order: list[str], init_state: np.ndarray):
    """Single-demo convenience writer (reads + rewrites the whole file); for
    batch generation accumulate metainfo_entry() results and dump once."""
    p = Path(path)
    meta = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    meta.setdefault(task_key, {})[demo_name] = metainfo_entry(task_key, rollout, object_order, init_state)
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
