import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from demogen_libero.trajectory import Frames, synthesize_two_stage


def make_fake_demo(T=60, seed=0):
    rng = np.random.default_rng(seed)
    action = rng.normal(scale=0.01, size=(T, 7))
    action[:, 3:6] = rng.normal(scale=0.02, size=(T, 3))
    action[:20, 6] = -1
    action[20:45, 6] = 1
    action[45:, 6] = -1
    state = np.zeros((T, 3))
    state[0] = rng.normal(size=3)
    for t in range(T - 1):
        state[t + 1] = state[t] + action[t, :3]
    return state, action


ONES = np.ones(3)  # fake demo's action IS the raw delta, so use an identity output_max scale


def test_skill_actions_unchanged():
    state, action = make_fake_demo()
    frames = Frames(20, 25, 45)
    obj_t = np.array([0.05, -0.03, 0.0])
    tar_t = np.array([-0.02, 0.04, 0.0])
    new_action, new_state = synthesize_two_stage(state, action, frames, obj_t, tar_t, ONES)
    f1, f2, f3 = frames.as_tuple()
    assert np.allclose(new_action[f1:f2], action[f1:f2]), "skill-1 action must be verbatim"
    assert np.allclose(new_action[f3:], action[f3:]), "skill-2 action must be verbatim"
    assert np.allclose(new_action[:, 3:6], action[:, 3:6]), "orientation must be unchanged everywhere"
    assert np.allclose(new_action[:, 6], action[:, 6]), "gripper command must be unchanged everywhere"


def test_reaches_shifted_object_and_target():
    state, action = make_fake_demo()
    frames = Frames(20, 25, 45)
    obj_t = np.array([0.05, -0.03, 0.0])
    tar_t = np.array([-0.02, 0.04, 0.0])
    new_action, new_state = synthesize_two_stage(state, action, frames, obj_t, tar_t, ONES)
    f1, f2, f3 = frames.as_tuple()
    assert np.allclose(new_state[f1], state[f1] + obj_t, atol=1e-8)
    assert np.allclose(new_state[f2], state[f2] + obj_t, atol=1e-8)
    assert np.allclose(new_state[f3], state[f3] + tar_t, atol=1e-8)


def test_relative_skill_geometry_preserved():
    """Skill segments preserve the *shape* (relative displacement) of the source,
    just rigidly translated -- this is the DemoGen invariant that lets a fixed
    grasp motion still work on a moved object."""
    state, action = make_fake_demo()
    frames = Frames(20, 25, 45)
    obj_t = np.array([0.05, -0.03, 0.0])
    tar_t = np.zeros(3)
    new_action, new_state = synthesize_two_stage(state, action, frames, obj_t, tar_t, ONES)
    f1, f2 = frames.skill_1_frame, frames.motion_2_frame
    source_disp = state[f2 - 1] - state[f1]
    new_disp = new_state[f2 - 1] - new_state[f1]
    assert np.allclose(source_disp, new_disp, atol=1e-8)


def test_zero_offset_is_identity():
    """With zero offset the correction terms are zero, so the synthesized action
    must be byte-identical to the source (drift-based synthesis, not from-scratch
    re-interpolation -- this is what lets the untouched case exactly reproduce the
    already-known-to-succeed source demo)."""
    state, action = make_fake_demo()
    frames = Frames(20, 25, 45)
    zero = np.zeros(3)
    new_action, new_state = synthesize_two_stage(state, action, frames, zero, zero, ONES)
    assert np.allclose(new_action, action, atol=1e-8)
    assert np.allclose(new_state, state, atol=1e-8)


def test_clips_infeasible_correction():
    """A correction that would exceed the [-1, 1] action range must be clipped,
    not silently wrap or blow up."""
    state, action = make_fake_demo()
    frames = Frames(20, 25, 45)
    huge = np.array([5.0, 0.0, 0.0])
    new_action, _ = synthesize_two_stage(state, action, frames, huge, np.zeros(3), ONES)
    assert np.all(new_action[:, :3] >= -1.0) and np.all(new_action[:, :3] <= 1.0)


if __name__ == "__main__":
    test_skill_actions_unchanged()
    test_reaches_shifted_object_and_target()
    test_relative_skill_geometry_preserved()
    test_zero_offset_is_identity()
    test_clips_infeasible_correction()
    print("all trajectory tests passed")
