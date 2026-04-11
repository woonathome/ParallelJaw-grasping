from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .html_grasp_parser import TopGraspData, parse_top_grasp_from_html

try:
    import mujoco
except Exception as _mujoco_import_error:  # pragma: no cover
    mujoco = None
    _MUJOCO_IMPORT_ERROR = _mujoco_import_error
else:
    _MUJOCO_IMPORT_ERROR = None


@dataclass
class SimConfig:
    # Contact/grasp parameters
    friction: float = 0.8
    grasp_force: float = 90.0
    object_density: float = 850.0
    gravity_z: float = 0.0

    # Parallel-jaw pad dimensions (meter)
    pad_width: float = 0.034
    pad_height: float = 0.021
    pad_depth: float = 0.007

    # Initial opening and closing behavior (meter)
    pre_clearance: float = 0.004
    squeeze_extra: float = 0.0015
    close_ramp: bool = True
    require_dual_contact: bool = True
    tighten_step: float = 0.001
    max_tighten_iters: int = 80
    tighten_settle_time: float = 0.03
    recenter_step: float = 0.0008
    max_recenter_iters: int = 12
    recenter_settle_time: float = 0.03

    # 3-direction perturb speeds (m/s)
    depth_speed: float = 0.6
    width_speed: float = 0.6
    height_speed: float = 0.6
    max_move: float = 0.03

    # Timing
    timestep: float = 0.001
    close_time: float = 0.35
    settle_time: float = 0.2
    perturb_time: float = 0.08
    hold_time: float = 0.08

    # Escape criteria
    escape_distance: float = 0.02
    min_loss_contact_steps: int = 12

    # Parse scale: HTML(mm) -> MuJoCo(m)
    unit_scale: float = 1e-3

    # Mesh cache
    cache_dir: str = "Grasping_Mujoco_Sim/.cache"


def _require_mujoco() -> None:
    if mujoco is None:
        raise RuntimeError(
            "mujoco package is not available in this environment. "
            "Install it in sam6d with: `conda run -n sam6d pip install mujoco`"
        ) from _MUJOCO_IMPORT_ERROR


def _unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    out = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(out))
    if n < 1e-12:
        return np.asarray(fallback, dtype=float)
    return out / n


def _format_vec(v: np.ndarray) -> str:
    return f"{v[0]:.9g} {v[1]:.9g} {v[2]:.9g}"


def _quat_wxyz_from_rotmat(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    trace = float(r[0, 0] + r[1, 1] + r[2, 2])
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)


def _write_obj(mesh_path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    with mesh_path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for tri in faces:
            i, j, k = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
            f.write(f"f {i} {j} {k}\n")


def _build_gripper_basis(grasp: TopGraspData) -> np.ndarray:
    x_axis = _unit(grasp.depth_axis, np.array([1.0, 0.0, 0.0]))
    y_seed = grasp.width_axis - x_axis * float(np.dot(grasp.width_axis, x_axis))
    y_axis = _unit(y_seed, np.array([0.0, 1.0, 0.0]))
    z_axis = _unit(np.cross(x_axis, y_axis), np.array([0.0, 0.0, 1.0]))

    # Align z with parsed height direction if available.
    if float(np.dot(z_axis, grasp.height_axis)) < 0.0:
        y_axis = -y_axis
        z_axis = -z_axis

    return np.column_stack((x_axis, y_axis, z_axis))


def _build_mjcf(mesh_file: Path, grasp: TopGraspData, cfg: SimConfig) -> tuple[str, Dict[str, float]]:
    depth_dist = float(np.linalg.norm(grasp.contact_j - grasp.contact_i))
    initial_gap = max(depth_dist + 2.0 * cfg.pre_clearance, cfg.pad_depth + 0.002)
    # Give generous over-travel margin for objects where HTML pair distance
    # underestimates local thickness around the actual pad contact region.
    joint_max = max(0.002, initial_gap * 0.72)
    close_each = min(cfg.pre_clearance + 0.5 * cfg.squeeze_extra, 0.9 * joint_max)

    base_force = max(140.0, 3.0 * cfg.grasp_force)
    friction = max(0.05, float(cfg.friction))
    fric_str = f"{friction:.6g} {0.04*friction:.6g} {0.008*friction:.6g}"

    rot = _build_gripper_basis(grasp)
    quat = _quat_wxyz_from_rotmat(rot)

    half_gap = 0.5 * initial_gap
    finger_left_pos = np.array([-(half_gap + 0.5 * cfg.pad_depth), 0.0, 0.0])
    finger_right_pos = np.array([half_gap + 0.5 * cfg.pad_depth, 0.0, 0.0])
    pad_half = np.array([0.5 * cfg.pad_depth, 0.5 * cfg.pad_width, 0.5 * cfg.pad_height])

    mesh_file_xml = mesh_file.as_posix()
    xml = f"""
<mujoco model="html_top_grasp_parallel_jaw">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="{cfg.timestep:.9g}" gravity="0 0 {cfg.gravity_z:.9g}" integrator="implicitfast"/>

  <default>
    <joint damping="3"/>
    <geom friction="{fric_str}" condim="4" solref="0.002 1" solimp="0.97 0.995 0.001"/>
  </default>

  <asset>
    <mesh name="obj_mesh" file="{mesh_file_xml}"/>
  </asset>

  <worldbody>
    <body name="gripper_base" pos="{_format_vec(grasp.grasp_center)}" quat="{quat[0]:.9g} {quat[1]:.9g} {quat[2]:.9g} {quat[3]:.9g}">
      <inertial pos="0 0 0" mass="0.35" diaginertia="0.0012 0.0012 0.0012"/>
      <joint name="tx" type="slide" axis="1 0 0" range="-0.08 0.08"/>
      <joint name="ty" type="slide" axis="0 1 0" range="-0.08 0.08"/>
      <joint name="tz" type="slide" axis="0 0 1" range="-0.08 0.08"/>

      <body name="finger_left" pos="{_format_vec(finger_left_pos)}">
        <joint name="finger_left_slide" type="slide" axis="1 0 0" range="0 {joint_max:.9g}"/>
        <geom name="finger_left_geom" type="box" size="{_format_vec(pad_half)}" rgba="0.15 0.45 0.85 1" contype="1" conaffinity="2"/>
      </body>

      <body name="finger_right" pos="{_format_vec(finger_right_pos)}">
        <joint name="finger_right_slide" type="slide" axis="-1 0 0" range="0 {joint_max:.9g}"/>
        <geom name="finger_right_geom" type="box" size="{_format_vec(pad_half)}" rgba="0.15 0.45 0.85 1" contype="1" conaffinity="2"/>
      </body>
    </body>

    <body name="object">
      <freejoint/>
      <geom name="object_geom" type="mesh" mesh="obj_mesh" density="{cfg.object_density:.9g}" rgba="0.73 0.73 0.73 1" contype="2" conaffinity="1"/>
    </body>
  </worldbody>

  <actuator>
    <position name="a_finger_left" joint="finger_left_slide" kp="14000" forcerange="-{cfg.grasp_force:.9g} {cfg.grasp_force:.9g}"/>
    <position name="a_finger_right" joint="finger_right_slide" kp="14000" forcerange="-{cfg.grasp_force:.9g} {cfg.grasp_force:.9g}"/>

    <position name="a_tx" joint="tx" kp="5000" forcerange="-{base_force:.9g} {base_force:.9g}"/>
    <position name="a_ty" joint="ty" kp="5000" forcerange="-{base_force:.9g} {base_force:.9g}"/>
    <position name="a_tz" joint="tz" kp="5000" forcerange="-{base_force:.9g} {base_force:.9g}"/>
  </actuator>
</mujoco>
"""
    return xml, {
        "initial_gap": initial_gap,
        "close_each": close_each,
        "joint_max": joint_max,
    }


def _name_id(model: Any, obj_type: Any, name: str) -> int:
    idx = int(mujoco.mj_name2id(model, obj_type, name))
    if idx < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return idx


def _has_object_finger_contact(data: Any, obj_gid: int, finger_gids: set[int]) -> bool:
    for ci in range(int(data.ncon)):
        c = data.contact[ci]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        if (g1 == obj_gid and g2 in finger_gids) or (g2 == obj_gid and g1 in finger_gids):
            return True
    return False


def _object_contact_each_finger(
    data: Any,
    obj_gid: int,
    left_gid: int,
    right_gid: int,
) -> tuple[bool, bool]:
    left = False
    right = False
    for ci in range(int(data.ncon)):
        c = data.contact[ci]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        if g1 == obj_gid:
            if g2 == left_gid:
                left = True
            elif g2 == right_gid:
                right = True
        elif g2 == obj_gid:
            if g1 == left_gid:
                left = True
            elif g1 == right_gid:
                right = True
        if left and right:
            break
    return left, right


def _snapshot_state(data: Any) -> Dict[str, np.ndarray]:
    return {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "act": data.act.copy() if data.act.size else np.empty((0,), dtype=float),
        "ctrl": data.ctrl.copy(),
    }


def _restore_state(model: Any, data: Any, state: Dict[str, np.ndarray]) -> None:
    data.qpos[:] = state["qpos"]
    data.qvel[:] = state["qvel"]
    if data.act.size and state["act"].size:
        data.act[:] = state["act"]
    data.ctrl[:] = state["ctrl"]
    mujoco.mj_forward(model, data)


def _hash_mesh(vertices: np.ndarray, faces: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(vertices.tobytes())
    h.update(faces.tobytes())
    return h.hexdigest()[:16]


def _init_simulation(
    grasp: TopGraspData,
    cfg: SimConfig,
) -> tuple[Any, Any, Dict[str, float], Dict[str, int]]:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = f"obj_{_hash_mesh(grasp.mesh_vertices, grasp.mesh_faces)}.obj"
    mesh_path = cache_dir / mesh_name
    if not mesh_path.exists():
        _write_obj(mesh_path, grasp.mesh_vertices, grasp.mesh_faces)

    xml, inner = _build_mjcf(mesh_path, grasp, cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ids = {
        "aid_f_l": _name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_finger_left"),
        "aid_f_r": _name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_finger_right"),
        "aid_tx": _name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_tx"),
        "aid_ty": _name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_ty"),
        "aid_tz": _name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_tz"),
        "gid_obj": _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"),
        "gid_l": _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "finger_left_geom"),
        "gid_r": _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "finger_right_geom"),
        "bid_obj": _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "object"),
        "bid_gripper": _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base"),
    }
    return model, data, inner, ids


def _run_simulation_core(
    *,
    model: Any,
    data: Any,
    grasp: TopGraspData,
    cfg: SimConfig,
    inner: Dict[str, float],
    ids: Dict[str, int],
    on_step: Optional[Any] = None,
    keep_running: Optional[Any] = None,
) -> Dict[str, Any]:
    finger_gids = {ids["gid_l"], ids["gid_r"]}
    ctrl_zero = np.zeros_like(data.ctrl)
    data.ctrl[:] = ctrl_zero

    steps_close = max(1, int(round(cfg.close_time / cfg.timestep)))
    steps_settle = max(1, int(round(cfg.settle_time / cfg.timestep)))
    steps_perturb = max(1, int(round(cfg.perturb_time / cfg.timestep)))
    steps_hold = max(1, int(round(cfg.hold_time / cfg.timestep)))
    aborted = False

    def _step_once() -> bool:
        nonlocal aborted
        if keep_running is not None and (not bool(keep_running())):
            aborted = True
            return False
        mujoco.mj_step(model, data)
        if on_step is not None:
            on_step(model, data)
        return True

    def _step_repeat(n: int) -> bool:
        for _ in range(n):
            if not _step_once():
                return False
        return True

    rel0 = data.xpos[ids["bid_obj"]].copy() - data.xpos[ids["bid_gripper"]].copy()

    # 1) Close grasp with ramp + auto tighten (until dual pad contact if requested).
    final_close_target = float(inner["close_each"])
    joint_cap = float(inner.get("joint_max", final_close_target * 2.0)) * 0.985

    def _set_finger_target(val: float) -> None:
        data.ctrl[ids["aid_f_l"]] = val
        data.ctrl[ids["aid_f_r"]] = val

    tx_bias = 0.0
    if cfg.close_ramp:
        for si in range(steps_close):
            alpha = float(si + 1) / float(steps_close)
            _set_finger_target(final_close_target * alpha)
            if not _step_once():
                break
    else:
        _set_finger_target(final_close_target)
        _step_repeat(steps_close)
    _step_repeat(steps_settle)

    left_contact, right_contact = _object_contact_each_finger(
        data, ids["gid_obj"], ids["gid_l"], ids["gid_r"]
    )
    recenter_iters_done = 0
    recenter_settle_steps = max(1, int(round(cfg.recenter_settle_time / cfg.timestep)))
    while (
        (not aborted)
        and cfg.require_dual_contact
        and (left_contact ^ right_contact)
        and (recenter_iters_done < int(cfg.max_recenter_iters))
    ):
        if left_contact and (not right_contact):
            tx_bias -= float(cfg.recenter_step)
        elif right_contact and (not left_contact):
            tx_bias += float(cfg.recenter_step)
        tx_bias = float(np.clip(tx_bias, -0.03, 0.03))
        data.ctrl[ids["aid_tx"]] = tx_bias
        _set_finger_target(final_close_target)
        _step_repeat(recenter_settle_steps)
        left_contact, right_contact = _object_contact_each_finger(
            data, ids["gid_obj"], ids["gid_l"], ids["gid_r"]
        )
        recenter_iters_done += 1

    tighten_iters_done = 0
    tighten_settle_steps = max(1, int(round(cfg.tighten_settle_time / cfg.timestep)))
    while (
        (not aborted)
        and cfg.require_dual_contact
        and (not (left_contact and right_contact))
        and (tighten_iters_done < int(cfg.max_tighten_iters))
        and (final_close_target < joint_cap)
    ):
        final_close_target = min(final_close_target + float(cfg.tighten_step), joint_cap)
        _set_finger_target(final_close_target)
        _step_repeat(tighten_settle_steps)
        left_contact, right_contact = _object_contact_each_finger(
            data, ids["gid_obj"], ids["gid_l"], ids["gid_r"]
        )
        tighten_iters_done += 1

    contact_after_grasp = _has_object_finger_contact(data, ids["gid_obj"], finger_gids)
    dual_contact_after_grasp = bool(left_contact and right_contact)
    rel_after_grasp = data.xpos[ids["bid_obj"]].copy() - data.xpos[ids["bid_gripper"]].copy()
    grasp_rel_shift = float(np.linalg.norm(rel_after_grasp - rel0))
    contact_ok = dual_contact_after_grasp if cfg.require_dual_contact else contact_after_grasp
    stable_after_grasp = bool(contact_ok and grasp_rel_shift <= cfg.escape_distance)

    grasped_state = _snapshot_state(data)
    rel_ref = rel_after_grasp.copy()

    axis_settings = {
        "depth": (ids["aid_tx"], float(cfg.depth_speed)),
        "width": (ids["aid_ty"], float(cfg.width_speed)),
        "height": (ids["aid_tz"], float(cfg.height_speed)),
    }

    perturb_results: Dict[str, Dict[str, Any]] = {}
    for axis_name, (act_id, speed) in axis_settings.items():
        if aborted:
            break

        _restore_state(model, data, grasped_state)
        data.ctrl[ids["aid_tx"]] = tx_bias
        data.ctrl[ids["aid_ty"]] = 0.0
        data.ctrl[ids["aid_tz"]] = 0.0
        _set_finger_target(final_close_target)

        move_target = float(np.clip(speed * cfg.perturb_time, -cfg.max_move, cfg.max_move))
        lost_contact_steps = 0
        lost_contact_steps_max = 0
        max_rel_shift = 0.0
        had_contact = False
        phase_stats: Dict[str, Dict[str, Any]] = {
            "positive": {"max_relative_shift_m": 0.0, "lost_contact_steps_max": 0, "had_contact": False},
            "negative": {"max_relative_shift_m": 0.0, "lost_contact_steps_max": 0, "had_contact": False},
            "return": {"max_relative_shift_m": 0.0, "lost_contact_steps_max": 0, "had_contact": False},
        }

        def _set_axis_offset(offset: float) -> None:
            data.ctrl[ids["aid_tx"]] = tx_bias
            data.ctrl[ids["aid_ty"]] = 0.0
            data.ctrl[ids["aid_tz"]] = 0.0
            if act_id == ids["aid_tx"]:
                data.ctrl[ids["aid_tx"]] = tx_bias + offset
            elif act_id == ids["aid_ty"]:
                data.ctrl[ids["aid_ty"]] = offset
            else:
                data.ctrl[ids["aid_tz"]] = offset

        def _update_contact_and_shift(phase_key: str) -> None:
            nonlocal lost_contact_steps, lost_contact_steps_max, max_rel_shift, had_contact
            in_contact = _has_object_finger_contact(data, ids["gid_obj"], finger_gids)
            had_contact = had_contact or in_contact
            phase_stats[phase_key]["had_contact"] = bool(phase_stats[phase_key]["had_contact"] or in_contact)

            lost_contact_steps = 0 if in_contact else (lost_contact_steps + 1)
            lost_contact_steps_max = max(lost_contact_steps_max, lost_contact_steps)
            phase_stats[phase_key]["lost_contact_steps_max"] = max(
                int(phase_stats[phase_key]["lost_contact_steps_max"]), lost_contact_steps
            )

            rel_now = data.xpos[ids["bid_obj"]] - data.xpos[ids["bid_gripper"]]
            rel_shift = float(np.linalg.norm(rel_now - rel_ref))
            max_rel_shift = max(max_rel_shift, rel_shift)
            phase_stats[phase_key]["max_relative_shift_m"] = max(
                float(phase_stats[phase_key]["max_relative_shift_m"]), rel_shift
            )

        def _run_phase(
            phase_key: str,
            start_offset: float,
            end_offset: float,
            ramp_steps: int,
            hold_steps: int,
        ) -> bool:
            ramp_steps = max(1, int(ramp_steps))
            for si in range(ramp_steps):
                if aborted:
                    return False
                alpha = float(si + 1) / float(ramp_steps)
                offset = start_offset + (end_offset - start_offset) * alpha
                _set_axis_offset(offset)
                if not _step_once():
                    return False
                _update_contact_and_shift(phase_key)

            for _ in range(max(0, int(hold_steps))):
                if aborted:
                    return False
                _set_axis_offset(end_offset)
                if not _step_once():
                    return False
                _update_contact_and_shift(phase_key)
            return True

        # + direction -> - direction -> return to 0
        did_return = False
        if _run_phase("positive", 0.0, move_target, steps_perturb, steps_hold):
            if _run_phase("negative", move_target, -move_target, steps_perturb * 2, steps_hold):
                did_return = _run_phase("return", -move_target, 0.0, steps_perturb, steps_hold)

        escaped_positive = bool(
            (int(phase_stats["positive"]["lost_contact_steps_max"]) >= cfg.min_loss_contact_steps)
            or (float(phase_stats["positive"]["max_relative_shift_m"]) > cfg.escape_distance)
        )
        escaped_negative = bool(
            (int(phase_stats["negative"]["lost_contact_steps_max"]) >= cfg.min_loss_contact_steps)
            or (float(phase_stats["negative"]["max_relative_shift_m"]) > cfg.escape_distance)
        )
        escaped_return = bool(
            (int(phase_stats["return"]["lost_contact_steps_max"]) >= cfg.min_loss_contact_steps)
            or (float(phase_stats["return"]["max_relative_shift_m"]) > cfg.escape_distance)
        )
        escaped = bool(escaped_positive or escaped_negative or escaped_return)

        perturb_results[axis_name] = {
            "speed_mps": speed,
            "move_target_m": move_target,
            "max_relative_shift_m": max_rel_shift,
            "lost_contact_steps": int(lost_contact_steps),
            "lost_contact_steps_max": int(lost_contact_steps_max),
            "had_contact": bool(had_contact),
            "escaped": escaped,
            "escaped_positive": escaped_positive,
            "escaped_negative": escaped_negative,
            "escaped_return": escaped_return,
            "phase": phase_stats,
            "returned_to_origin": bool(did_return and (not aborted)),
        }

    any_escape = any(r["escaped"] for r in perturb_results.values())
    return {
        "pair_name": grasp.pair_name,
        "legend_group": grasp.legend_group,
        "source_html": grasp.source_html,
        "config": asdict(cfg),
        "initial_gap_m": inner["initial_gap"],
        "close_each_m": inner["close_each"],
        "final_close_target_m": final_close_target,
        "tx_bias_m": tx_bias,
        "recenter_iters_done": int(recenter_iters_done),
        "tighten_iters_done": int(tighten_iters_done),
        "stable_after_grasp": stable_after_grasp,
        "contact_after_grasp": contact_after_grasp,
        "dual_contact_after_grasp": dual_contact_after_grasp,
        "grasp_relative_shift_m": grasp_rel_shift,
        "perturbation": perturb_results,
        "any_escape": any_escape,
        "aborted": bool(aborted),
    }


def simulate_from_parsed_grasp(grasp: TopGraspData, cfg: Optional[SimConfig] = None) -> Dict[str, Any]:
    _require_mujoco()
    cfg = cfg or SimConfig()
    model, data, inner, ids = _init_simulation(grasp, cfg)
    return _run_simulation_core(
        model=model,
        data=data,
        grasp=grasp,
        cfg=cfg,
        inner=inner,
        ids=ids,
    )


def play_in_viewer_from_parsed_grasp(
    grasp: TopGraspData,
    cfg: Optional[SimConfig] = None,
    *,
    realtime: bool = True,
    show_contact_points: bool = False,
    show_contact_forces: bool = False,
) -> Dict[str, Any]:
    """
    Open native MuJoCo viewer window and play grasp + perturbation episode.
    """
    _require_mujoco()
    cfg = cfg or SimConfig()
    try:
        import mujoco.viewer as mj_viewer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "mujoco.viewer is unavailable. Ensure GUI-capable MuJoCo install."
        ) from exc

    model, data, inner, ids = _init_simulation(grasp, cfg)
    dt = float(cfg.timestep)
    last_wall = time.perf_counter()

    with mj_viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1 if show_contact_points else 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1 if show_contact_forces else 0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0
        viewer.cam.distance = max(0.35, 3.0 * float(model.stat.extent))
        viewer.cam.lookat[:] = data.xpos[ids["bid_obj"]]
        viewer.sync()

        def _keep_running() -> bool:
            return bool(viewer.is_running())

        def _on_step(_: Any, __: Any) -> None:
            nonlocal last_wall
            viewer.sync()
            if realtime:
                now = time.perf_counter()
                elapsed = now - last_wall
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                    now = time.perf_counter()
                last_wall = now

        result = _run_simulation_core(
            model=model,
            data=data,
            grasp=grasp,
            cfg=cfg,
            inner=inner,
            ids=ids,
            on_step=_on_step,
            keep_running=_keep_running,
        )

        for _ in range(max(1, int(round(0.35 / dt)))):
            if not viewer.is_running():
                break
            viewer.sync()
            if realtime:
                time.sleep(dt)

    return result


def simulate_from_html(html_path: str, cfg: Optional[SimConfig] = None) -> Dict[str, Any]:
    cfg = cfg or SimConfig()
    parsed = parse_top_grasp_from_html(html_path, unit_scale=cfg.unit_scale)
    return simulate_from_parsed_grasp(parsed, cfg=cfg)


def play_in_viewer_from_html(
    html_path: str,
    cfg: Optional[SimConfig] = None,
    *,
    realtime: bool = True,
    show_contact_points: bool = False,
    show_contact_forces: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or SimConfig()
    parsed = parse_top_grasp_from_html(html_path, unit_scale=cfg.unit_scale)
    return play_in_viewer_from_parsed_grasp(
        parsed,
        cfg=cfg,
        realtime=realtime,
        show_contact_points=show_contact_points,
        show_contact_forces=show_contact_forces,
    )


def run_from_html(
    html_path: str,
    *,
    depth_speed: float = 0.6,
    width_speed: float = 0.6,
    height_speed: float = 0.6,
    friction: float = 0.8,
    grasp_force: float = 90.0,
    gravity_z: float = 0.0,
    pre_clearance: float = 0.004,
    squeeze_extra: float = 0.0015,
    require_dual_contact: bool = True,
    tighten_step: float = 0.001,
    max_tighten_iters: int = 80,
    perturb_time: float = 0.08,
    hold_time: float = 0.08,
    open_viewer: bool = False,
    realtime_viewer: bool = True,
    show_contact_points: bool = False,
    show_contact_forces: bool = False,
) -> Dict[str, Any]:
    cfg = SimConfig(
        depth_speed=depth_speed,
        width_speed=width_speed,
        height_speed=height_speed,
        friction=friction,
        grasp_force=grasp_force,
        gravity_z=gravity_z,
        pre_clearance=pre_clearance,
        squeeze_extra=squeeze_extra,
        require_dual_contact=require_dual_contact,
        tighten_step=tighten_step,
        max_tighten_iters=max_tighten_iters,
        perturb_time=perturb_time,
        hold_time=hold_time,
    )
    if open_viewer:
        return play_in_viewer_from_html(
            html_path,
            cfg=cfg,
            realtime=realtime_viewer,
            show_contact_points=show_contact_points,
            show_contact_forces=show_contact_forces,
        )
    return simulate_from_html(html_path, cfg=cfg)


def launch_notebook_ui(default_html_path: str = "") -> Any:
    """
    Notebook UI:
    - 3-direction move speeds (depth/width/height)
    - friction coefficient
    - grasp force
    """
    import ipywidgets as widgets
    from IPython.display import display

    html_path = widgets.Text(
        value=default_html_path,
        description="HTML",
        placeholder="Grasping_Results_SurfaceContact/.../1-grasping_pairs_feasible_*.html",
        layout=widgets.Layout(width="95%"),
    )
    friction = widgets.FloatSlider(value=0.8, min=0.1, max=2.0, step=0.05, description="mu")
    grasp_force = widgets.FloatSlider(value=90.0, min=10.0, max=300.0, step=5.0, description="force(N)")
    depth_speed = widgets.FloatSlider(value=0.6, min=0.05, max=2.5, step=0.05, description="depth(m/s)")
    width_speed = widgets.FloatSlider(value=0.6, min=0.05, max=2.5, step=0.05, description="width(m/s)")
    height_speed = widgets.FloatSlider(value=0.6, min=0.05, max=2.5, step=0.05, description="height(m/s)")
    perturb_time = widgets.FloatSlider(value=0.08, min=0.02, max=0.3, step=0.01, description="move_t(s)")
    hold_time = widgets.FloatSlider(value=0.08, min=0.01, max=0.3, step=0.01, description="hold_t(s)")

    open_viewer = widgets.Checkbox(value=True, description="Open MuJoCo Viewer")
    realtime_viewer = widgets.Checkbox(value=True, description="Realtime Viewer")
    run_btn = widgets.Button(description="Run MuJoCo Grasp Test", button_style="primary")
    out = widgets.Output(layout=widgets.Layout(border="1px solid #ccc", padding="8px"))

    def _on_run(_: Any) -> None:
        with out:
            out.clear_output()
            if not html_path.value.strip():
                print("HTML 경로를 입력하세요.")
                return

            try:
                result = run_from_html(
                    html_path=html_path.value.strip(),
                    depth_speed=depth_speed.value,
                    width_speed=width_speed.value,
                    height_speed=height_speed.value,
                    friction=friction.value,
                    grasp_force=grasp_force.value,
                    perturb_time=perturb_time.value,
                    hold_time=hold_time.value,
                    open_viewer=open_viewer.value,
                    realtime_viewer=realtime_viewer.value,
                )
            except Exception as exc:
                print(f"[ERROR] {exc}")
                return

            print(f"Pair: {result['pair_name']}")
            print(f"Stable after grasp: {result['stable_after_grasp']}")
            print(f"Any escape: {result['any_escape']}")
            print(f"Aborted: {result.get('aborted', False)}")
            print("-" * 60)
            for axis in ("depth", "width", "height"):
                axis_res = result["perturbation"][axis]
                print(
                    f"[{axis}] escaped={axis_res['escaped']} | "
                    f"speed={axis_res['speed_mps']:.3f} m/s | "
                    f"move={axis_res['move_target_m']:.4f} m | "
                    f"max_shift={axis_res['max_relative_shift_m']:.4f} m | "
                    f"lost_contact_steps={axis_res['lost_contact_steps']}"
                )

    run_btn.on_click(_on_run)

    ui = widgets.VBox(
        [
            widgets.HTML("<h3>MuJoCo Parallel Jaw Grasp Test</h3>"),
            html_path,
            widgets.HBox([friction, grasp_force]),
            widgets.HBox([depth_speed, width_speed, height_speed]),
            widgets.HBox([perturb_time, hold_time]),
            widgets.HBox([open_viewer, realtime_viewer]),
            run_btn,
            out,
        ]
    )
    display(ui)
    return ui
