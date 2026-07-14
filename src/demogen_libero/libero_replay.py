"""Build a LIBERO env, translate an object's free joint, open-loop replay an
action sequence, and report task success."""
import os

os.environ.setdefault("MUJOCO_GL", "wgl")

import numpy as np
from libero.libero.envs import OffScreenRenderEnv


def make_env(bddl_file: str, camera_height: int = 128, camera_width: int = 128):
    return OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=camera_height,
        camera_widths=camera_width,
    )


def reset_to_init_state(env, init_state: np.ndarray):
    env.reset()
    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()


def translate_body_joint(env, joint_name: str, offset_xyz: np.ndarray):
    """Add offset_xyz (3,) to a free joint's position qpos[:3], then forward()."""
    qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
    start = qpos_addr[0] if isinstance(qpos_addr, tuple) else qpos_addr
    env.sim.data.qpos[start : start + 3] += offset_xyz
    env.sim.forward()


def get_body_pos(env, joint_name: str) -> np.ndarray:
    qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
    start = qpos_addr[0] if isinstance(qpos_addr, tuple) else qpos_addr
    return np.array(env.sim.data.qpos[start : start + 3])


def swap_object_xy(env, joint_a: str, joint_b: str):
    """Swap the tabletop (x, y) positions of two free-jointed objects, keeping
    each object's own z (they may rest at different heights)."""
    addr_a = env.sim.model.get_joint_qpos_addr(joint_a)
    addr_b = env.sim.model.get_joint_qpos_addr(joint_b)
    sa = addr_a[0] if isinstance(addr_a, tuple) else addr_a
    sb = addr_b[0] if isinstance(addr_b, tuple) else addr_b
    xy_a = env.sim.data.qpos[sa : sa + 2].copy()
    env.sim.data.qpos[sa : sa + 2] = env.sim.data.qpos[sb : sb + 2]
    env.sim.data.qpos[sb : sb + 2] = xy_a
    env.sim.forward()


def replay_actions(env, actions: np.ndarray, render: bool = False, collect: bool = False):
    """Open-loop replay. Returns (success, last_obs, frames) or, with
    collect=True, (success, last_obs, rollout_dict).

    The rollout dict mirrors the source LIBERO HDF5 layout so a saved episode
    can be replayed / converted the same way as an original demo:
        actions (T,7), states (T,110), ee_pos (T,3), gripper_states (T,2),
        joint_states (T,7), agentview_rgb (T,H,W,3), eye_in_hand_rgb (T,H,W,3)
    """
    frames = []
    success = False
    obs = None
    rollout = {
        "actions": [], "states": [], "ee_pos": [], "gripper_states": [],
        "joint_states": [], "agentview_rgb": [], "eye_in_hand_rgb": [],
    } if collect else None
    for t in range(actions.shape[0]):
        if collect:
            rollout["states"].append(env.sim.get_state().flatten())
        obs, reward, done, info = env.step(actions[t])
        if render:
            frames.append(obs["agentview_image"])
        if collect:
            rollout["actions"].append(actions[t])
            rollout["ee_pos"].append(obs["robot0_eef_pos"])
            rollout["gripper_states"].append(obs["robot0_gripper_qpos"])
            rollout["joint_states"].append(obs["robot0_joint_pos"])
            rollout["agentview_rgb"].append(obs["agentview_image"])
            rollout["eye_in_hand_rgb"].append(obs["robot0_eye_in_hand_image"])
        if env.check_success():
            success = True
    if collect:
        rollout = {k: np.asarray(v) for k, v in rollout.items()}
        return success, obs, rollout
    return success, obs, frames


def replay_actions_closed_loop(env, actions: np.ndarray, ref_states: np.ndarray,
                               frames, gain: float = 1.0, max_corr: float = 0.4,
                               output_max_xyz: float = 0.05, collect: bool = False,
                               settle_tol: float = 0.005, max_settle: int = 40):
    """Replay the source actions while steering the EE onto `ref_states` (the
    intended absolute path from trajectory.intended_states).

    LIBERO's OSC controller re-anchors its goal to the *actual* EE pose every
    step and only converges partway within one control step, so purely open-loop
    offsets systematically undershoot -- position feedback in the motion
    segments closes that gap. Skill (contact) segments run verbatim open-loop so
    grasp/place dynamics match the source exactly.

    Before entering each skill segment, up to `max_settle` extra feedback-only
    frames (zero nominal motion, gripper held) are inserted until the EE is
    within `settle_tol` of the segment-start reference -- large offsets cannot
    be fully absorbed inside the motion segment's original frame budget. These
    settle frames are recorded in the rollout like ordinary frames, so the saved
    episode may be longer than the source.

    gain: feedback proportional gain; max_corr: per-step correction clamp in
    normalized action units (0.4 -> 2 cm at output_max 0.05 m).
    """
    f1, f2, f3 = frames.as_tuple()
    T = actions.shape[0]
    success = False
    obs = None
    rollout = {
        "actions": [], "states": [], "ee_pos": [], "gripper_states": [],
        "joint_states": [], "agentview_rgb": [], "eye_in_hand_rgb": [],
    } if collect else None

    def record_pre(act):
        if collect:
            rollout["states"].append(env.sim.get_state().flatten())

    def record_post(act, obs):
        if collect:
            rollout["actions"].append(act)
            rollout["ee_pos"].append(obs["robot0_eef_pos"])
            rollout["gripper_states"].append(obs["robot0_gripper_qpos"])
            rollout["joint_states"].append(obs["robot0_joint_pos"])
            rollout["agentview_rgb"].append(obs["agentview_image"])
            rollout["eye_in_hand_rgb"].append(obs["robot0_eye_in_hand_image"])

    ee_pos = None
    nonlocal_success = [False]

    def step(act):
        nonlocal obs, ee_pos
        record_pre(act)
        obs, reward, done, info = env.step(act)
        ee_pos = obs["robot0_eef_pos"]
        record_post(act, obs)
        if env.check_success():
            nonlocal_success[0] = True

    def settle_to(ref_pos, grip_cmd):
        for _ in range(max_settle):
            if ee_pos is None or np.linalg.norm(ref_pos - ee_pos) < settle_tol:
                break
            corr = np.clip(gain * (ref_pos - ee_pos) / output_max_xyz, -max_corr, max_corr)
            act = np.zeros(7)
            act[:3] = np.clip(corr, -1.0, 1.0)
            act[6] = grip_cmd
            step(act)

    for t in range(T):
        if t == f1:
            settle_to(ref_states[f1], actions[max(f1 - 1, 0)][6])
        elif t == f3:
            settle_to(ref_states[f3], actions[max(f3 - 1, 0)][6])
        act = actions[t].copy()
        in_motion = (t < f1) or (f2 <= t < f3)
        if in_motion and ee_pos is not None:
            corr = gain * (ref_states[t] - ee_pos) / output_max_xyz
            act[:3] = np.clip(act[:3] + np.clip(corr, -max_corr, max_corr), -1.0, 1.0)
        step(act)

    success = nonlocal_success[0]
    if collect:
        rollout = {k: np.asarray(v) for k, v in rollout.items()}
    return success, obs, rollout


def replay_uniform(env, base_actions: np.ndarray, ref_path: np.ndarray, frames,
                   gain: float = 1.0, ff_gain: float = 1.25, max_corr: float = 0.4,
                   output_max_xyz: float = 0.05, collect: bool = False,
                   settle_tol: float = 0.005, max_settle: int = 20,
                   extract=None, warmup_steps: int = 5, post_steps: int = 3):
    """Execute a uniform-speed synthesized trajectory (from
    trajectory.synthesize_uniform): motion segments run feedforward (the
    reference velocity, boosted by ff_gain to offset the OSC controller's
    per-step undershoot) plus position feedback onto ref_path; skill segments
    run the source actions verbatim. A short settle before each skill segment
    absorbs any residual tracking error.

    Recording semantics (BC-friendly): obs[t] is the observation BEFORE
    executing action[t], so states[0]/ee_pos[0] describe the initial scene and
    a policy trained on (obs[t] -> action[t]) never re-executes a motion it has
    already completed. `warmup_steps` zero-action frames run un-recorded first
    to let physics/controller settle; `post_steps` zero-action hold frames are
    appended (and recorded) after the trajectory ends.

    Phase labels: settle frames before the grasp belong to skill_1 (position
    correction is part of grasping); settle frames before the place belong to
    skill_2.

    extract: optional callable (env, obs) -> dict of per-frame arrays; when
    given (with collect=True) each rollout row comes from it (plus the executed
    action under "actions") instead of the default key set."""
    f1, f2, f3 = frames.as_tuple()
    T = base_actions.shape[0]
    rollout = {} if collect else None
    flag = [False]
    obs = None

    def record(act, pre_obs, phase):
        if extract is not None:
            row = extract(env, pre_obs)  # env.sim is pre-step at record time
        else:
            row = {
                "states": env.sim.get_state().flatten(),
                "ee_pos": pre_obs["robot0_eef_pos"],
                "gripper_states": pre_obs["robot0_gripper_qpos"],
                "joint_states": pre_obs["robot0_joint_pos"],
                "agentview_rgb": pre_obs["agentview_image"],
                "eye_in_hand_rgb": pre_obs["robot0_eye_in_hand_image"],
            }
        row["actions"] = act
        row["phase_id"] = np.int32(phase)
        for k, v in row.items():
            rollout.setdefault(k, []).append(v)

    def step(act, phase):
        nonlocal obs
        if collect:
            record(act, obs, phase)  # obs = pre-step observation
        obs, reward, done, info = env.step(act)
        if env.check_success():
            flag[0] = True

    def ee():
        return np.asarray(obs["robot0_eef_pos"])

    def settle_to(ref_pos, grip_cmd, phase):
        for _ in range(max_settle):
            if np.linalg.norm(ref_pos - ee()) < settle_tol:
                break
            act = np.zeros(7)
            act[:3] = np.clip(gain * (ref_pos - ee()) / output_max_xyz, -max_corr, max_corr)
            act[6] = grip_cmd
            step(act, phase)

    # warmup: not recorded; hold the source's initial gripper command
    warm_act = np.zeros(7)
    warm_act[6] = base_actions[0][6]
    for _ in range(max(warmup_steps, 1)):  # >=1 so obs is populated before recording
        obs, _, _, _ = env.step(warm_act)

    # phase ids: 0=motion_1 (reach), 1=skill_1 (grasp incl. pre-grasp settle),
    #            2=motion_2 (transport), 3=skill_2 (place incl. pre-place settle)
    for t in range(T):
        if t == f1:
            settle_to(ref_path[f1], base_actions[max(f1 - 1, 0)][6], 1)
        elif t == f3:
            settle_to(ref_path[f3], base_actions[max(f3 - 1, 0)][6], 3)
        act = base_actions[t].copy()
        in_motion = (t < f1) or (f2 <= t < f3)
        if in_motion:
            ff = ff_gain * (ref_path[min(t + 1, T - 1)] - ref_path[t]) / output_max_xyz
            fb = np.clip(gain * (ref_path[t] - ee()) / output_max_xyz, -max_corr, max_corr)
            act[:3] = np.clip(ff + fb, -1.0, 1.0)
        phase = 0 if t < f1 else (1 if t < f2 else (2 if t < f3 else 3))
        step(act, phase)

    # post-trajectory hold frames (recorded, phase 3)
    hold_act = np.zeros(7)
    hold_act[6] = base_actions[-1][6]
    for _ in range(post_steps):
        step(hold_act, 3)

    success = flag[0]
    if collect:
        rollout = {k: np.asarray(v) for k, v in rollout.items()}
    return success, obs, rollout


def save_video(video_path: str, rgb_frames: np.ndarray, fps: int = 20):
    """Write (T,H,W,3) uint8 RGB frames to an mp4. LIBERO offscreen frames use
    the OpenGL convention (bottom row first), so flip vertically for viewing."""
    import cv2

    T, H, W, _ = rgb_frames.shape
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    try:
        for t in range(T):
            frame = rgb_frames[t][::-1]  # opengl -> viewer orientation
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def save_episode(hdf5_path: str, episode_key: str, rollout: dict, attrs: dict = None):
    """Append one collected rollout to an output HDF5 file under data/<episode_key>,
    mirroring the LIBERO demo layout (obs/* subgroup for observations)."""
    import h5py

    obs_keys = ("ee_pos", "gripper_states", "joint_states", "agentview_rgb", "eye_in_hand_rgb")
    with h5py.File(hdf5_path, "a") as f:
        data = f.require_group("data")
        if episode_key in data:
            del data[episode_key]
        ep = data.create_group(episode_key)
        ep.create_dataset("actions", data=rollout["actions"])
        ep.create_dataset("states", data=rollout["states"])
        obs_grp = ep.create_group("obs")
        for k in obs_keys:
            obs_grp.create_dataset(k, data=rollout[k])
        for k, v in (attrs or {}).items():
            ep.attrs[k] = v
