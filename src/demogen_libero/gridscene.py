"""Scene placement for LIBERO floor tasks, following the reference
randomize_object_positions_grid.py design:

- the target object is sampled CONTINUOUSLY inside an inset workspace (not on
  grid cells), with farthest-point scoring against previously used target
  positions so placements spread out across the whole dataset;
- distractors go onto jittered grid cells with pairwise spacing constraints;
- the basket stays at a fixed reference position with a small jitter;
- a scene is defined by ABSOLUTE object positions, so the same scene can be
  applied to different source demos' init states (demos-per-scene > 1).
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PlacementSpec:
    x_range: tuple = (-0.26, 0.18)
    y_range: tuple = (-0.24, 0.18)      # extends toward the basket at (0, 0.26)
    edge_margin: float = 0.03           # inset for the distractor grid
    primary_edge_margin: float = 0.05   # extra inset for the target's continuous sampling
    cell_size: float = 0.11             # distractor grid cell
    jitter_frac: float = 0.15           # distractor jitter, fraction of cell size
    min_spacing: float = 0.10           # min pairwise object center distance
    basket_clearance: float = 0.12      # min object distance to the basket
    basket_jitter: float = 0.01
    primary_mode: str = "score"         # "score" (farthest-biased) or "random"
    primary_tries: int = 300
    park_xy: tuple = (2.0, 2.0)         # unused distractors go here (off-camera)
    occlusion_lateral: float = 0.035     # min lateral offset from the camera ray
                                        # through any other object (anti-occlusion;
                                        # the rendered seg check is the hard gate)

    def cells(self):
        xmin = self.x_range[0] + self.edge_margin
        xmax = self.x_range[1] - self.edge_margin
        ymin = self.y_range[0] + self.edge_margin
        ymax = self.y_range[1] - self.edge_margin
        nx = int(np.floor((xmax - xmin) / self.cell_size))
        ny = int(np.floor((ymax - ymin) / self.cell_size))
        assert nx > 0 and ny > 0, "cell_size too large for bounds"
        x0 = xmin + ((xmax - xmin) - nx * self.cell_size) / 2.0
        y0 = ymin + ((ymax - ymin) - ny * self.cell_size) / 2.0
        return [np.array([x0 + (i + 0.5) * self.cell_size, y0 + (j + 0.5) * self.cell_size])
                for i in range(nx) for j in range(ny)]


def _occludes(a_xy, b_xy, cam_xy, lateral):
    """True if a and b are nearly collinear with the camera (one hides the
    other from the agentview): the farther point lies within `lateral` of the
    ray from the camera through the nearer point."""
    if cam_xy is None:
        return False
    a = np.asarray(a_xy) - cam_xy
    b = np.asarray(b_xy) - cam_xy
    near, far = (a, b) if np.linalg.norm(a) <= np.linalg.norm(b) else (b, a)
    u = near / max(np.linalg.norm(near), 1e-9)
    perp = far - np.dot(far, u) * u
    return float(np.linalg.norm(perp)) < lateral and np.dot(far, u) > 0


def sample_scene(rng: np.random.Generator, spec: PlacementSpec, n_distractors: int,
                 ref_basket_xy: np.ndarray, used_target_xy: list = None,
                 cam_xy: np.ndarray = None, distractor_pool: list = None) -> dict:
    """Sample one scene as absolute positions: target (continuous, scored),
    n_distractors on grid cells, basket = reference + jitter.

    cam_xy: agentview camera ground position; when given, placements that are
    nearly collinear with the camera ray through any other object are rejected
    (anti-occlusion). distractor_pool: joint names to draw the scene's
    distractor identities from (fixed per scene so demos-per-scene share the
    exact same object layout)."""
    used_target_xy = used_target_xy if used_target_xy is not None else []
    basket_xy = np.asarray(ref_basket_xy, dtype=np.float64) + \
        rng.uniform(-spec.basket_jitter, spec.basket_jitter, size=2)

    pxmin = spec.x_range[0] + spec.primary_edge_margin
    pxmax = spec.x_range[1] - spec.primary_edge_margin
    pymin = spec.y_range[0] + spec.primary_edge_margin
    pymax = spec.y_range[1] - spec.primary_edge_margin

    best, best_score = None, -np.inf
    for _ in range(spec.primary_tries):
        xy = np.array([rng.uniform(pxmin, pxmax), rng.uniform(pymin, pymax)])
        d_basket = float(np.linalg.norm(xy - basket_xy))
        if d_basket < spec.basket_clearance:
            continue
        if _occludes(xy, basket_xy, cam_xy, spec.occlusion_lateral):
            continue
        if spec.primary_mode == "random":
            best = xy
            break
        # farthest-from-used only: adding a basket-distance bonus here biases
        # placements away from the basket side of the workspace
        score = min((float(np.linalg.norm(xy - q)) for q in used_target_xy), default=1.0)
        if score > best_score:
            best, best_score = xy, score
    if best is None:
        raise RuntimeError("could not sample a target position satisfying basket clearance")
    target_xy = best

    cells = spec.cells()
    rng.shuffle(cells)
    jitter_amp = spec.cell_size * spec.jitter_frac
    placed = [target_xy]
    distractor_xys = []
    for cell in cells:
        if len(distractor_xys) >= n_distractors:
            break
        xy = cell + rng.uniform(-jitter_amp, jitter_amp, size=2)
        if np.linalg.norm(xy - basket_xy) < spec.basket_clearance:
            continue
        if min(np.linalg.norm(xy - q) for q in placed) < spec.min_spacing:
            continue
        if any(_occludes(xy, q, cam_xy, spec.occlusion_lateral) for q in placed) or \
                _occludes(xy, basket_xy, cam_xy, spec.occlusion_lateral):
            continue
        distractor_xys.append(xy)
        placed.append(xy)
    if len(distractor_xys) < n_distractors:
        raise RuntimeError(f"only placed {len(distractor_xys)}/{n_distractors} distractors; "
                           f"loosen min_spacing/cell_size or shrink n_distractors")

    scene = {
        "target_xy": target_xy.tolist(),
        "distractor_xys": [x.tolist() for x in distractor_xys],
        "basket_xy": basket_xy.tolist(),
        "n_distractors": n_distractors,
    }
    if distractor_pool is not None:
        scene["distractor_joints"] = list(rng.permutation(distractor_pool)[:n_distractors])
    return scene


def object_state_slices(env, joint_names: list) -> dict:
    """Flattened-sim-state start index of each free joint's position (the
    flattened state is [time(1), qpos, qvel], so it's qpos addr + 1)."""
    out = {}
    for jn in joint_names:
        addr = env.sim.model.get_joint_qpos_addr(jn)
        start = addr[0] if isinstance(addr, tuple) else addr
        out[jn] = start + 1
    return out


def apply_scene(scene: dict, init_state: np.ndarray, env, target_joint: str,
                distractor_joints: list, basket_joint: str,
                rng: np.random.Generator, spec: PlacementSpec = None):
    """Write a sampled scene's absolute positions into a source demo's init
    state. Which distractor objects fill the scene's distractor slots is chosen
    randomly. Returns (new_init_state, info)."""
    spec = spec or PlacementSpec()
    state = np.asarray(init_state, dtype=np.float64).copy()
    slices = object_state_slices(env, [target_joint, basket_joint, *distractor_joints])

    a = slices[target_joint]
    target_old = state[a:a + 2].copy()
    state[a:a + 2] = scene["target_xy"]

    b = slices[basket_joint]
    basket_old = state[b:b + 2].copy()
    state[b:b + 2] = scene["basket_xy"]

    # distractor identities: fixed by the scene when present (demos-per-scene
    # then share the exact same layout), otherwise chosen here
    chosen = scene.get("distractor_joints") or \
        list(rng.permutation(distractor_joints)[:len(scene["distractor_xys"])])
    for joint, xy in zip(chosen, scene["distractor_xys"]):
        j = slices[joint]
        state[j:j + 2] = xy
    for k, joint in enumerate(j for j in distractor_joints if j not in chosen):
        j = slices[joint]
        state[j:j + 2] = np.array(spec.park_xy) + np.array([0.4 * k, 0.0])

    for joint in [target_joint, basket_joint, *distractor_joints]:
        j = slices[joint]
        q = state[j + 3:j + 7]
        nq_ = np.linalg.norm(q)
        if np.isfinite(nq_) and nq_ > 1e-8:
            state[j + 3:j + 7] = q / nq_
    nq = env.sim.model.nq
    state[1 + nq:] = 0.0

    info = {
        "target_old_xy": target_old.tolist(),
        "target_new_xy": list(scene["target_xy"]),
        "basket_delta": (np.array(scene["basket_xy"]) - basket_old).tolist(),
        "distractor_joints": chosen,
        "n_distractors": scene["n_distractors"],
    }
    return state, info
