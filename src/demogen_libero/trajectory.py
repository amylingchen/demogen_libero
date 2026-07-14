"""DemoGen-style two-stage trajectory retargeting ("linear branch"): re-interpolate
the two free-motion segments in a straight line to the (offset-shifted) skill-segment
endpoints, and copy the two skill (contact) segments' actions verbatim.

Segment convention (T = number of source (state, action) rows):
    motion-1: t in [0, f1)   free approach to the object
    skill-1:  t in [f1, f2)  grasp (contact with object) -- action kept verbatim
    motion-2: t in [f2, f3)  transport while holding the object
    skill-2:  t in [f3, T)   place (contact with target) -- action kept verbatim

`obj_t` (the object's translation offset) shifts where skill-1 happens; `tar_t`
(the target/placement translation offset) shifts where skill-2 happens. Both are
3-vectors, in practice with obj_t[2] == tar_t[2] == 0 (LIBERO objects only get
re-placed within the tabletop plane).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Frames:
    skill_1_frame: int
    motion_2_frame: int
    skill_2_frame: int

    def as_tuple(self):
        return (self.skill_1_frame, self.motion_2_frame, self.skill_2_frame)


DEFAULT_OUTPUT_MAX_XYZ = np.array([0.05, 0.05, 0.05])  # OSC_POSE controller's action->meters scale


def synthesize_two_stage(
    state: np.ndarray,
    action: np.ndarray,
    frames: Frames,
    obj_t: np.ndarray,
    tar_t: np.ndarray,
    output_max_xyz: np.ndarray = DEFAULT_OUTPUT_MAX_XYZ,
):
    """Return (new_action, new_state) with the same shape as the source.

    `action[:, :3]` is a *normalized* OSC_POSE command in [-1, 1]; the real
    per-step Cartesian delta is action[:,:3] * output_max_xyz (and even then only
    approximately, since the controller has PD tracking dynamics). So motion
    segments are NOT re-interpolated from scratch in position space -- instead we
    keep the original (known-good) action shape and add a small constant
    correction (in action units = physical_delta / output_max_xyz) per step, which
    linearly steers the OSC setpoint by the required extra displacement without
    touching the segment's underlying dynamics. Skill segments stay untouched
    (verbatim), matching the source's contact behavior exactly.

    new_state is a diagnostic-only reconstruction that integrates
    new_action[:,:3] * output_max_xyz; it approximates but will not exactly equal
    the real simulated trajectory because of that PD tracking lag -- ground truth
    always comes from replaying new_action in the simulator.
    """
    f1, f2, f3 = frames.as_tuple()
    T = action.shape[0]
    assert state.shape[0] == T
    assert 0 < f1 < f2 < f3 < T, f"frames must satisfy 0 < f1 < f2 < f3 < T={T}, got {frames}"

    obj_t = np.asarray(obj_t, dtype=np.float64)
    tar_t = np.asarray(tar_t, dtype=np.float64)
    output_max_xyz = np.asarray(output_max_xyz, dtype=np.float64)

    new_action = action.copy()

    n1 = f1
    correction_1 = (obj_t / output_max_xyz) / n1
    new_action[0:f1, :3] = action[0:f1, :3] + correction_1

    # skill-1: verbatim (already copied by action.copy())

    n2 = f3 - f2
    correction_2 = ((tar_t - obj_t) / output_max_xyz) / n2
    new_action[f2:f3, :3] = action[f2:f3, :3] + correction_2

    # skill-2: verbatim (already copied by action.copy())

    new_action[:, :3] = np.clip(new_action[:, :3], -1.0, 1.0)

    new_state = np.zeros_like(state)
    new_state[0] = state[0]
    for t in range(T - 1):
        new_state[t + 1] = new_state[t] + new_action[t, :3] * output_max_xyz

    return new_action, new_state


def intended_states(state: np.ndarray, frames: Frames, obj_t: np.ndarray,
                    tar_t: np.ndarray, ramp_frac: float = 0.7) -> np.ndarray:
    """The DemoGen-style intended absolute EE path: the source's recorded EE
    positions plus a per-frame cumulative offset -- motion-1 ramps 0 -> obj_t,
    skill-1 holds obj_t, motion-2 ramps obj_t -> tar_t, skill-2 holds tar_t.
    Used as the tracking reference for closed-loop replay.

    The lateral offset ramp completes within the first `ramp_frac` of each
    motion segment and then holds, so the shift happens while the EE is still
    cruising high -- the final descent toward the (re-placed) object/target is
    purely vertical like the source's, instead of a low sideways sweep that
    grazes neighboring objects or closes the gripper mid-slide."""
    f1, f2, f3 = frames.as_tuple()
    T = state.shape[0]
    obj_t = np.asarray(obj_t, dtype=np.float64)
    tar_t = np.asarray(tar_t, dtype=np.float64)

    def ramp(n):
        n_ramp = max(int(n * ramp_frac), 1)
        r = np.ones(n)
        r[:n_ramp] = np.linspace(0.0, 1.0, n_ramp, endpoint=False) + 1.0 / n_ramp
        return r[:, None]

    offset = np.zeros((T, 3))
    offset[0:f1] = ramp(f1) * obj_t
    offset[f1:f2] = obj_t
    offset[f2:f3] = obj_t + ramp(f3 - f2) * (tar_t - obj_t)
    offset[f3:T] = tar_t
    return state + offset


def synthesize_uniform(state: np.ndarray, action: np.ndarray, frames: Frames,
                       obj_t: np.ndarray, tar_t: np.ndarray,
                       ramp_frac: float = 0.7, speed: float = None):
    """DemoGen-faithful synthesis with time reparameterization: motion segments
    are the source EE path plus an early-completing lateral offset ramp,
    resampled at uniform arc-length so the EE moves at the source demo's own
    average motion speed regardless of how far the object was displaced -- the
    segment's frame count grows/shrinks with actual path length instead of
    being locked to the source's frame budget. Skill segments stay verbatim.

    Returns (ref_path, base_actions, new_frames):
        ref_path (T',3)   uniform-speed absolute EE reference
        base_actions (T',7) per-frame nominal action: orientation deltas
                           resampled from the source (scaled to preserve total
                           rotation), gripper held per segment; [:3] left zero
                           for the executor to fill with feedforward+feedback
        new_frames        segment boundaries in the resampled timeline
    """
    f1, f2, f3 = frames.as_tuple()
    T = state.shape[0]
    obj_t = np.asarray(obj_t, dtype=np.float64)
    tar_t = np.asarray(tar_t, dtype=np.float64)

    if speed is None:
        seg_speeds = np.linalg.norm(np.diff(state[:f1], axis=0), axis=1)
        speed = max(float(np.mean(seg_speeds)), 1e-4)

    def offset_path(seg_states, start_off, end_off):
        n = len(seg_states)
        n_ramp = max(int(n * ramp_frac), 1)
        w = np.ones(n)
        w[:n_ramp] = np.linspace(0.0, 1.0, n_ramp, endpoint=False) + 1.0 / n_ramp
        return seg_states + (start_off + w[:, None] * (np.asarray(end_off) - start_off))

    def resample_uniform(path):
        """Resample the (offset) path so its duration scales with its arc
        length, while keeping the source segment's *normalized speed profile*
        (slow-down before contact etc.) instead of forcing constant speed."""
        d = np.linalg.norm(np.diff(path, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(d)])
        total = arc[-1]
        n_src = len(path) - 1
        n_new = max(int(np.ceil(total / speed)), 2)
        # normalized time -> normalized arc progress, taken from the source
        u_src = arc / max(total, 1e-9)
        tau_src = np.linspace(0.0, 1.0, n_src + 1)
        tau_new = np.linspace(0.0, 1.0, n_new)
        s_new = np.interp(tau_new, tau_src, u_src) * total
        out = np.empty((n_new, 3))
        for k in range(3):
            out[:, k] = np.interp(s_new, arc, path[:, k])
        return out

    def resample_actions(seg_actions, n_new):
        n_src = len(seg_actions)
        out = np.zeros((n_new, 7))
        src_idx = np.minimum((np.arange(n_new) * n_src / n_new).astype(int), n_src - 1)
        # orientation deltas: resample then scale so the summed rotation matches
        out[:, 3:6] = seg_actions[src_idx, 3:6] * (n_src / n_new)
        out[:, 6] = seg_actions[src_idx, 6]
        return out

    zero3 = np.zeros(3)
    # motion-1: source path shifted onto the object offset
    m1_path = resample_uniform(offset_path(state[0 : f1 + 1], zero3, obj_t))
    m1_act = resample_actions(action[0:f1], len(m1_path) - 1)
    # skill-1 verbatim: path is source + obj_t (reference only), actions untouched
    s1_path = state[f1 : f2 + 1] + obj_t
    s1_act = action[f1:f2].copy()
    # motion-2: transport, offset ramps obj_t -> tar_t
    m2_path = resample_uniform(offset_path(state[f2 : f3 + 1], obj_t, tar_t))
    m2_act = resample_actions(action[f2:f3], len(m2_path) - 1)
    # skill-2 verbatim
    s2_path = state[f3:T] + tar_t
    s2_act = action[f3:T].copy()

    ref_path = np.concatenate([m1_path[:-1], s1_path[:-1], m2_path[:-1], s2_path])
    base_actions = np.concatenate([m1_act, s1_act, m2_act, s2_act])
    nf1 = len(m1_act)
    nf2 = nf1 + len(s1_act)
    nf3 = nf2 + len(m2_act)
    new_frames = Frames(nf1, nf2, nf3)
    return ref_path, base_actions, new_frames


def auto_segment(state: np.ndarray, action: np.ndarray,
                 speed_resume=0.009, post_grasp_margin=2, pre_release_margin=10) -> Frames:
    """Derive the two-stage segmentation frames from the demo itself:

    - skill_1_frame: first frame the gripper-close command (+1) is issued.
    - motion_2_frame: first frame after the grasp where EE speed resumes above
      `speed_resume` m/frame (the settle during finger closing is near-zero
      speed), plus a small margin so the object is firmly in hand.
    - skill_2_frame: `pre_release_margin` frames before the gripper-open command,
      so the final descent into the target is kept verbatim as part of skill-2.

    Tuned to reproduce the manually verified (42, 52, 120) on alphabet-soup demo_0.
    """
    grip = action[:, 6]
    close_idx = int(np.argmax(grip > 0))
    assert grip[close_idx] > 0, "demo never closes the gripper"
    open_after = np.where(grip[close_idx:] < 0)[0]
    assert open_after.size > 0, "demo never re-opens the gripper"
    open_idx = close_idx + int(open_after[0])

    speed = np.linalg.norm(np.diff(state, axis=0), axis=1)
    resumed = np.where(speed[close_idx:] > speed_resume)[0]
    assert resumed.size > 0, "EE never moves after the grasp"
    f2 = close_idx + int(resumed[0]) + post_grasp_margin

    f3 = max(open_idx - pre_release_margin, f2 + 1)
    return Frames(close_idx, f2, f3)


def segment_regrasp(state: np.ndarray, action: np.ndarray,
                    speed_resume=0.009, post_grasp_margin=2, pre_release_margin=10) -> Frames:
    """Two-stage segmentation for trajectories that may contain a regrasp
    (gripper close->open->close->... before transport). The entire grasp attempt
    sequence -- from the FIRST close through the transport start after the LAST
    close -- is absorbed into skill_1, so the returned frames still describe the
    canonical four phases (reach / grasp / transport / place). Reduces to
    auto_segment for a clean single grasp.

    - skill_1_frame  = first gripper close
    - motion_2_frame = EE speed resumes above `speed_resume` after the LAST close
    - skill_2_frame  = `pre_release_margin` frames before the final gripper open
    """
    grip = action[:, 6]
    closes = np.where((grip[:-1] <= 0) & (grip[1:] > 0))[0] + 1
    opens = np.where((grip[:-1] > 0) & (grip[1:] <= 0))[0] + 1
    assert closes.size > 0 and opens.size > 0, "trajectory has no grasp cycle"
    f1 = int(closes[0])
    last_close = int(closes[-1])
    final_open = int(opens[-1])

    speed = np.linalg.norm(np.diff(state, axis=0), axis=1)
    resumed = np.where(speed[last_close:] > speed_resume)[0]
    assert resumed.size > 0, "EE never moves after the final grasp"
    f2 = last_close + int(resumed[0]) + post_grasp_margin

    f3 = max(final_open - pre_release_margin, f2 + 1)
    return Frames(f1, f2, f3)


def sample_object_offset(rng: np.random.Generator, radius_xy=(0.03, 0.08)) -> np.ndarray:
    """Sample a random in-plane (z=0) offset, uniform in an annulus so the object
    actually moves a noticeable amount but stays within a plausible reach radius."""
    r_min, r_max = radius_xy
    r = rng.uniform(r_min, r_max)
    theta = rng.uniform(0, 2 * np.pi)
    return np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
