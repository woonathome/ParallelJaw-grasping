from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .html_grasp_parser import (
    TopGraspData,
    parse_ranked_grasp_candidates_from_html,
    parse_top_grasp_from_html,
)

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
    object_density: float = 7870.0
    gravity_z: float = 0.0
    contact_margin: float = 0.0
    pad_collision_scale: float = 0.94
    object_collision_scale: float = 0.96
    contact_penetration_threshold: float = 5e-5
    object_collision_mode: str = "auto"  # mesh | surface_spheres | auto
    object_collision_auto_hull_ratio_threshold: float = 1.15
    object_collision_auto_use_spheres_when_hull_unknown: bool = True
    object_collision_sphere_count: int = 180
    object_collision_sphere_radius_scale: float = 0.55
    object_collision_sphere_min_radius: float = 4e-4
    object_collision_sphere_max_radius: float = 6e-3
    lock_object_until_dual_contact: bool = True
    force_unlock_before_perturb: bool = True

    # Parallel-jaw pad dimensions (meter)
    pad_width: float = 0.034
    pad_height: float = 0.021
    pad_depth: float = 0.007

    # Initial opening and closing behavior (meter)
    pre_clearance: float = 0.002
    approach_start_extra_gap: float = 0.015
    squeeze_extra: float = 0.0015
    close_ramp: bool = True
    require_dual_contact: bool = True
    tighten_step: float = 0.0002
    max_tighten_iters: int = 500
    max_tighten_overtravel: float = 0.0012
    tighten_settle_time: float = 0.03
    grasp_validation_time: float = 0.03
    min_any_contact_steps_for_grasp: int = 20
    min_dual_contact_steps_for_grasp: int = 20
    min_any_contact_steps_for_thin_support: int = 8
    require_grasp_before_perturb: bool = False
    relax_dual_contact_for_thin_support: bool = True
    recenter_step: float = 0.0008
    max_recenter_iters: int = 12
    recenter_settle_time: float = 0.03
    maintain_dual_contact_during_perturb: bool = True
    maintain_contact_grace_steps: int = 2
    maintain_tighten_step: float = 0.0001
    auto_expand_span_for_parallel_jaw: bool = True
    min_span_ratio_to_object_extent: float = 0.35
    min_span_abs: float = 0.003
    local_support_min_vertices: int = 20
    opening_retry_step: float = 0.004
    max_initial_opening_extra: float = 0.05
    max_opening_retries: int = 8
    enable_pair_fallback: bool = True
    pair_fallback_max_candidates: int = 64
    auto_pair_rescue_on_no_contact: bool = False
    auto_pair_rescue_max_candidates: int = 64
    auto_pair_rescue_require_thin_support: bool = True

    # 3-direction perturb speeds (m/s)
    depth_speed: float = 0.6
    width_speed: float = 0.6
    height_speed: float = 0.6
    max_move: float = 0.03

    # Timing
    timestep: float = 0.0002
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


def _mesh_key(vertices: np.ndarray, faces: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(np.asarray(vertices, dtype=np.float64).tobytes())
    h.update(np.asarray(faces, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


_HULL_RATIO_CACHE: Dict[str, Optional[float]] = {}


def _estimate_convex_hull_ratio(vertices: np.ndarray, faces: np.ndarray) -> Optional[float]:
    key = _mesh_key(vertices, faces)
    if key in _HULL_RATIO_CACHE:
        return _HULL_RATIO_CACHE[key]
    try:
        import trimesh  # type: ignore

        m = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=float), faces=np.asarray(faces, dtype=np.int64), process=False)
        if (not bool(m.is_volume)) or (abs(float(m.volume)) < 1e-12):
            _HULL_RATIO_CACHE[key] = None
            return None
        h = m.convex_hull
        ratio = abs(float(h.volume)) / (abs(float(m.volume)) + 1e-12)
        _HULL_RATIO_CACHE[key] = float(ratio)
        return float(ratio)
    except Exception:
        _HULL_RATIO_CACHE[key] = None
        return None


def _farthest_point_sample(points: np.ndarray, count: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    n = int(pts.shape[0])
    k = max(1, min(int(count), n))
    if k >= n:
        return pts.copy()
    centroid = np.mean(pts, axis=0)
    first = int(np.argmax(np.sum((pts - centroid[None, :]) ** 2, axis=1)))
    selected = np.empty((k,), dtype=np.int64)
    selected[0] = first
    min_d2 = np.full((n,), np.inf, dtype=float)
    cur = pts[first]
    for i in range(1, k):
        d2 = np.sum((pts - cur[None, :]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
        selected[i] = int(np.argmax(min_d2))
        cur = pts[int(selected[i])]
    return pts[selected]


def _build_surface_sphere_collision(
    vertices: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, float]:
    verts = np.asarray(vertices, dtype=float)
    sampled = _farthest_point_sample(verts, int(cfg.object_collision_sphere_count))
    center = np.mean(sampled, axis=0)
    scale = float(np.clip(float(cfg.object_collision_scale), 0.85, 1.0))
    sampled = center[None, :] + scale * (sampled - center[None, :])

    if sampled.shape[0] >= 2:
        dif = sampled[:, None, :] - sampled[None, :, :]
        d2 = np.sum(dif * dif, axis=2)
        np.fill_diagonal(d2, np.inf)
        nn = np.sqrt(np.min(d2, axis=1))
        base = float(np.median(nn[np.isfinite(nn)]))
    else:
        bbox = np.max(sampled, axis=0) - np.min(sampled, axis=0)
        base = 0.05 * float(np.linalg.norm(bbox))
    radius = float(cfg.object_collision_sphere_radius_scale) * max(1e-7, base)
    radius = float(np.clip(radius, float(cfg.object_collision_sphere_min_radius), float(cfg.object_collision_sphere_max_radius)))
    return sampled, radius


def _prepare_grasp_for_parallel_jaw(
    grasp: TopGraspData, cfg: SimConfig
) -> tuple[TopGraspData, Dict[str, Any]]:
    depth_axis = _unit(grasp.depth_axis, np.array([1.0, 0.0, 0.0]))
    width_axis = _unit(grasp.width_axis, np.array([0.0, 1.0, 0.0]))
    height_axis = _unit(grasp.height_axis, np.array([0.0, 0.0, 1.0]))

    contact_i = np.asarray(grasp.contact_i, dtype=float).copy()
    contact_j = np.asarray(grasp.contact_j, dtype=float).copy()
    grasp_center = np.asarray(grasp.grasp_center, dtype=float).copy()

    mesh_vertices = np.asarray(grasp.mesh_vertices, dtype=float)
    orig_span = float(np.linalg.norm(contact_j - contact_i))

    global_proj = mesh_vertices @ depth_axis
    global_lo = float(np.min(global_proj))
    global_hi = float(np.max(global_proj))
    obj_extent_along_depth = float(global_hi - global_lo)

    # Estimate local depth support around the intended grasp center using pad footprint.
    rel = mesh_vertices - grasp_center[None, :]
    rel_depth = rel @ depth_axis
    rel_width = rel @ width_axis
    rel_height = rel @ height_axis
    support_lo = float(np.min(rel_depth))
    support_hi = float(np.max(rel_depth))
    support_kind = "global"
    support_vertex_count = int(rel_depth.shape[0])
    min_vertices = max(4, int(cfg.local_support_min_vertices))
    half_w = 0.5 * float(cfg.pad_width)
    half_h = 0.5 * float(cfg.pad_height)
    for scale in (0.7, 1.0, 1.4, 2.0, 2.8):
        mask = (np.abs(rel_width) <= scale * half_w) & (np.abs(rel_height) <= scale * half_h)
        cnt = int(np.count_nonzero(mask))
        if cnt < min_vertices:
            continue
        support_lo = float(np.min(rel_depth[mask]))
        support_hi = float(np.max(rel_depth[mask]))
        support_kind = f"local_x{scale:.1f}"
        support_vertex_count = cnt
        break

    support_extent = float(support_hi - support_lo)
    ci_proj = float(np.dot(contact_i - grasp_center, depth_axis))
    cj_proj = float(np.dot(contact_j - grasp_center, depth_axis))
    outside_tol = max(1e-5, 0.5 * float(cfg.pre_clearance))
    endpoint_outside = bool(
        (ci_proj < support_lo - outside_tol)
        or (ci_proj > support_hi + outside_tol)
        or (cj_proj < support_lo - outside_tol)
        or (cj_proj > support_hi + outside_tol)
    )
    span_adjusted = False

    if (
        cfg.auto_expand_span_for_parallel_jaw
        and support_extent > 1e-7
        and (
            orig_span
            < max(float(cfg.min_span_abs), float(cfg.min_span_ratio_to_object_extent) * support_extent)
            or endpoint_outside
        )
    ):
        contact_i = grasp_center + depth_axis * support_lo
        contact_j = grasp_center + depth_axis * support_hi
        grasp_center = 0.5 * (contact_i + contact_j)
        span_adjusted = True

    eff_span = float(np.linalg.norm(contact_j - contact_i))
    prepared = TopGraspData(
        pair_name=grasp.pair_name,
        legend_group=grasp.legend_group,
        source_html=grasp.source_html,
        mesh_vertices=grasp.mesh_vertices,
        mesh_faces=grasp.mesh_faces,
        contact_i=contact_i,
        contact_j=contact_j,
        grasp_center=grasp_center,
        depth_axis=depth_axis,
        width_axis=width_axis,
        height_axis=height_axis,
    )
    prep_info: Dict[str, Any] = {
        "original_span_m": orig_span,
        "effective_span_m": eff_span,
        "object_extent_along_depth_m": obj_extent_along_depth,
        "support_extent_m": support_extent,
        "support_kind": support_kind,
        "support_vertex_count": support_vertex_count,
        "contact_endpoint_outside_projection": endpoint_outside,
        "span_adjusted": bool(span_adjusted),
    }
    return prepared, prep_info


def _build_mjcf(
    mesh_file: Path,
    grasp: TopGraspData,
    cfg: SimConfig,
    *,
    opening_extra: float = 0.0,
) -> tuple[str, Dict[str, Any]]:
    grasp, prep_info = _prepare_grasp_for_parallel_jaw(grasp, cfg)
    depth_dist = float(np.linalg.norm(grasp.contact_j - grasp.contact_i))
    initial_gap = max(
        depth_dist + 2.0 * cfg.pre_clearance + float(cfg.approach_start_extra_gap) + float(opening_extra),
        cfg.pad_depth + 0.002,
    )
    # Give generous over-travel margin for objects where HTML pair distance
    # underestimates local thickness around the actual pad contact region.
    joint_max = max(0.002, initial_gap * 0.72)
    close_each = min(cfg.pre_clearance + 0.5 * cfg.squeeze_extra, 0.9 * joint_max)
    support_extent = float(prep_info.get("support_extent_m", depth_dist))
    thin_support_threshold = 2.0 * float(cfg.pad_depth)
    is_thin_support = bool((support_extent > 0.0) and (support_extent <= thin_support_threshold))
    target_gap = max(0.0, depth_dist - float(cfg.squeeze_extra))
    target_close_each = 0.5 * max(0.0, initial_gap - target_gap)
    tighten_extra = min(float(cfg.max_tighten_overtravel), 0.3 * max(depth_dist, 1e-4))
    if is_thin_support:
        tighten_cap = min(0.985 * joint_max, max(close_each, target_close_each + tighten_extra))
    else:
        tighten_cap = 0.985 * joint_max

    base_force = max(140.0, 3.0 * cfg.grasp_force)
    friction = max(0.05, float(cfg.friction))
    fric_str = f"{friction:.6g} {0.04*friction:.6g} {0.008*friction:.6g}"

    rot = _build_gripper_basis(grasp)
    quat = _quat_wxyz_from_rotmat(rot)

    half_gap = 0.5 * initial_gap
    finger_left_pos = np.array([-(half_gap + 0.5 * cfg.pad_depth), 0.0, 0.0])
    finger_right_pos = np.array([half_gap + 0.5 * cfg.pad_depth, 0.0, 0.0])
    pad_half = np.array([0.5 * cfg.pad_depth, 0.5 * cfg.pad_width, 0.5 * cfg.pad_height])
    pad_collision_scale = float(np.clip(float(cfg.pad_collision_scale), 0.6, 1.0))
    pad_half_collision = pad_half * pad_collision_scale
    obj_collision_scale = float(np.clip(float(cfg.object_collision_scale), 0.85, 1.0))
    obj_scale_xyz = f"{obj_collision_scale:.9g} {obj_collision_scale:.9g} {obj_collision_scale:.9g}"
    contact_margin = max(0.0, float(cfg.contact_margin))

    requested_mode = str(cfg.object_collision_mode).strip().lower()
    if requested_mode not in {"mesh", "surface_spheres", "auto"}:
        requested_mode = "auto"
    hull_ratio = _estimate_convex_hull_ratio(grasp.mesh_vertices, grasp.mesh_faces)
    auto_reason = "forced"
    if requested_mode == "surface_spheres":
        use_surface_spheres = True
    elif requested_mode == "mesh":
        use_surface_spheres = False
    else:
        if hull_ratio is None:
            use_surface_spheres = bool(cfg.object_collision_auto_use_spheres_when_hull_unknown)
            auto_reason = "hull_ratio_unknown"
        else:
            use_surface_spheres = bool(
                hull_ratio >= float(cfg.object_collision_auto_hull_ratio_threshold)
            )
            auto_reason = "hull_ratio_threshold"

    sphere_count = 0
    sphere_radius = 0.0
    if use_surface_spheres:
        sph_pts, sphere_radius = _build_surface_sphere_collision(grasp.mesh_vertices, cfg)
        sphere_count = int(sph_pts.shape[0])
        object_collision_xml = "\n".join(
            f'      <geom name="object_col_{i}" type="sphere" pos="{_format_vec(p)}" size="{sphere_radius:.9g}" density="0" rgba="0.73 0.73 0.73 0" contype="2" conaffinity="1"/>'
            for i, p in enumerate(sph_pts)
        )
        effective_mode = "surface_spheres"
    else:
        object_collision_xml = (
            f'      <geom name="object_geom" type="mesh" mesh="obj_mesh" size="{obj_scale_xyz}" '
            f'density="0" rgba="0.73 0.73 0.73 0" contype="2" conaffinity="1"/>'
        )
        effective_mode = "mesh"

    mesh_file_xml = mesh_file.as_posix()
    xml = f"""
<mujoco model="html_top_grasp_parallel_jaw">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="{cfg.timestep:.9g}" gravity="0 0 {cfg.gravity_z:.9g}" integrator="implicitfast"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <rgba contactpoint="1 0.25 0.25 0.6"/>
  </visual>

  <default>
    <joint damping="3"/>
    <geom friction="{fric_str}" condim="4" solref="0.002 1" solimp="0.97 0.995 0.001" margin="{contact_margin:.9g}" gap="0"/>
  </default>

  <asset>
    <texture name="bg_sky" type="skybox" builtin="flat" rgb1="0.86 0.86 0.86" width="512" height="512"/>
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
        <geom name="finger_left_vis" type="box" size="{_format_vec(pad_half)}" rgba="0.15 0.45 0.85 0.6" contype="0" conaffinity="0"/>
        <geom name="finger_left_geom" type="box" size="{_format_vec(pad_half_collision)}" rgba="0.15 0.45 0.85 0" contype="1" conaffinity="2"/>
      </body>

      <body name="finger_right" pos="{_format_vec(finger_right_pos)}">
        <joint name="finger_right_slide" type="slide" axis="-1 0 0" range="0 {joint_max:.9g}"/>
        <geom name="finger_right_vis" type="box" size="{_format_vec(pad_half)}" rgba="0.15 0.45 0.85 0.6" contype="0" conaffinity="0"/>
        <geom name="finger_right_geom" type="box" size="{_format_vec(pad_half_collision)}" rgba="0.15 0.45 0.85 0" contype="1" conaffinity="2"/>
      </body>
    </body>

    <body name="object">
      <freejoint/>
      <geom name="object_vis" type="mesh" mesh="obj_mesh" density="0" rgba="0.73 0.73 0.73 0.6" contype="0" conaffinity="0"/>
      <geom name="object_mass" type="mesh" mesh="obj_mesh" density="{cfg.object_density:.9g}" rgba="0.73 0.73 0.73 0" contype="0" conaffinity="0"/>
{object_collision_xml}
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
    inner: Dict[str, Any] = {
        "initial_gap": initial_gap,
        "close_each": close_each,
        "joint_max": joint_max,
        "target_close_each": target_close_each,
        "tighten_cap": tighten_cap,
        "is_thin_support": bool(is_thin_support),
        "opening_extra_m": float(opening_extra),
        "approach_start_extra_gap_m": float(cfg.approach_start_extra_gap),
        "pad_collision_scale": float(pad_collision_scale),
        "object_collision_scale": float(obj_collision_scale),
        "contact_margin": float(contact_margin),
        "object_collision_mode": str(effective_mode),
        "object_collision_auto_reason": str(auto_reason),
        "object_collision_auto_threshold": float(cfg.object_collision_auto_hull_ratio_threshold),
        "object_convex_hull_ratio": float(hull_ratio) if hull_ratio is not None else None,
        "object_collision_proxy_count": int(sphere_count),
        "object_collision_proxy_radius_m": float(sphere_radius),
    }
    inner.update(prep_info)
    return xml, inner


def _name_id(model: Any, obj_type: Any, name: str) -> int:
    idx = int(mujoco.mj_name2id(model, obj_type, name))
    if idx < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return idx


def _collect_geom_ids_with_prefix(model: Any, prefix: str) -> set[int]:
    out: set[int] = set()
    for gid in range(int(model.ngeom)):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if nm and str(nm).startswith(prefix):
            out.add(int(gid))
    return out


def _has_object_finger_contact(
    data: Any,
    obj_gids: set[int],
    finger_gids: set[int],
    min_penetration: float,
) -> bool:
    dist_max = -abs(float(min_penetration))
    for ci in range(int(data.ncon)):
        c = data.contact[ci]
        if float(c.dist) > dist_max:
            continue
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        if (g1 in obj_gids and g2 in finger_gids) or (g2 in obj_gids and g1 in finger_gids):
            return True
    return False


def _object_contact_each_finger(
    data: Any,
    obj_gids: set[int],
    left_gid: int,
    right_gid: int,
    min_penetration: float,
) -> tuple[bool, bool]:
    left = False
    right = False
    dist_max = -abs(float(min_penetration))
    for ci in range(int(data.ncon)):
        c = data.contact[ci]
        if float(c.dist) > dist_max:
            continue
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        if g1 in obj_gids:
            if g2 == left_gid:
                left = True
            elif g2 == right_gid:
                right = True
        elif g2 in obj_gids:
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
) -> tuple[Any, Any, Dict[str, Any], Dict[str, Any]]:
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mesh_name = f"obj_{_hash_mesh(grasp.mesh_vertices, grasp.mesh_faces)}.obj"
    mesh_path = cache_dir / mesh_name
    if not mesh_path.exists():
        _write_obj(mesh_path, grasp.mesh_vertices, grasp.mesh_faces)

    opening_extra = 0.0
    step_extra = max(1e-4, float(cfg.opening_retry_step))
    max_extra = max(0.0, float(cfg.max_initial_opening_extra))
    max_retries = max(0, int(cfg.max_opening_retries))

    model = None
    data = None
    ids: Dict[str, Any] = {}
    inner: Dict[str, Any] = {}
    retry_count = 0
    had_start_contact = False

    for retry in range(max_retries + 1):
        xml, inner = _build_mjcf(mesh_path, grasp, cfg, opening_extra=opening_extra)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        object_col_gids = _collect_geom_ids_with_prefix(model, "object_col_")
        if object_col_gids:
            obj_gid_set = object_col_gids
        else:
            obj_gid_set = {_name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")}

        ids = {
            "aid_f_l": _name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_finger_left"),
            "aid_f_r": _name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_finger_right"),
            "aid_tx": _name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_tx"),
            "aid_ty": _name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_ty"),
            "aid_tz": _name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "a_tz"),
            "gid_obj_set": obj_gid_set,
            "gid_l": _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "finger_left_geom"),
            "gid_r": _name_id(model, mujoco.mjtObj.mjOBJ_GEOM, "finger_right_geom"),
            "bid_obj": _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "object"),
            "bid_gripper": _name_id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base"),
        }
        start_contact = _has_object_finger_contact(
            data,
            ids["gid_obj_set"],
            {ids["gid_l"], ids["gid_r"]},
            cfg.contact_penetration_threshold,
        )
        had_start_contact = bool(had_start_contact or start_contact)
        retry_count = retry
        if (not start_contact) or (opening_extra >= max_extra) or (retry >= max_retries):
            break
        opening_extra = min(max_extra, opening_extra + step_extra * (1.0 + 0.5 * retry))

    inner["opening_extra_m"] = float(opening_extra)
    inner["init_overlap_retry_count"] = int(retry_count)
    inner["had_start_contact_overlap"] = bool(had_start_contact)
    return model, data, inner, ids


def _run_simulation_core(
    *,
    model: Any,
    data: Any,
    grasp: TopGraspData,
    cfg: SimConfig,
    inner: Dict[str, Any],
    ids: Dict[str, Any],
    on_step: Optional[Any] = None,
    keep_running: Optional[Any] = None,
) -> Dict[str, Any]:
    finger_gids = {ids["gid_l"], ids["gid_r"]}
    ctrl_zero = np.zeros_like(data.ctrl)
    data.ctrl[:] = ctrl_zero

    # Optional approach-stage object lock:
    # keep object fixed until both pads contact it to suppress pre-grasp push-out.
    lock_active = bool(cfg.lock_object_until_dual_contact)
    lock_released_by_dual_contact = False
    lock_forced_release = False
    lock_release_step = -1
    lock_release_time_s = -1.0
    obj_jnt = int(model.body_jntadr[ids["bid_obj"]]) if int(model.body_jntnum[ids["bid_obj"]]) > 0 else -1
    obj_qpos_adr = int(model.jnt_qposadr[obj_jnt]) if obj_jnt >= 0 else -1
    obj_qvel_adr = int(model.jnt_dofadr[obj_jnt]) if obj_jnt >= 0 else -1
    obj_lock_qpos = (
        data.qpos[obj_qpos_adr : obj_qpos_adr + 7].copy()
        if (lock_active and obj_qpos_adr >= 0 and obj_qvel_adr >= 0)
        else np.empty((0,), dtype=float)
    )
    step_counter = 0

    steps_close = max(1, int(round(cfg.close_time / cfg.timestep)))
    steps_settle = max(1, int(round(cfg.settle_time / cfg.timestep)))
    steps_perturb = max(1, int(round(cfg.perturb_time / cfg.timestep)))
    steps_hold = max(1, int(round(cfg.hold_time / cfg.timestep)))
    aborted = False

    def _step_once() -> bool:
        nonlocal aborted, lock_active, lock_released_by_dual_contact, lock_release_step, lock_release_time_s, step_counter
        if keep_running is not None and (not bool(keep_running())):
            aborted = True
            return False
        mujoco.mj_step(model, data)
        step_counter += 1
        if lock_active and obj_qpos_adr >= 0 and obj_qvel_adr >= 0:
            l_now, r_now = _object_contact_each_finger(
                data, ids["gid_obj_set"], ids["gid_l"], ids["gid_r"], cfg.contact_penetration_threshold
            )
            if bool(l_now and r_now):
                lock_active = False
                lock_released_by_dual_contact = True
                lock_release_step = int(step_counter)
                lock_release_time_s = float(data.time)
                data.qvel[obj_qvel_adr : obj_qvel_adr + 6] = 0.0
                mujoco.mj_forward(model, data)
            else:
                data.qpos[obj_qpos_adr : obj_qpos_adr + 7] = obj_lock_qpos
                data.qvel[obj_qvel_adr : obj_qvel_adr + 6] = 0.0
                mujoco.mj_forward(model, data)
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
    tighten_cap = float(np.clip(float(inner.get("tighten_cap", joint_cap)), final_close_target, joint_cap))
    if bool(inner.get("is_thin_support", False)):
        maintain_cap = float(
            np.clip(tighten_cap + max(0.0, float(cfg.max_tighten_overtravel)), final_close_target, joint_cap)
        )
    else:
        maintain_cap = float(joint_cap)

    def _set_finger_target(val: float) -> None:
        data.ctrl[ids["aid_f_l"]] = val
        data.ctrl[ids["aid_f_r"]] = val

    tx_bias = 0.0
    # Pre-close recenter: if one pad already contacts at the fully-open pose,
    # shift base along grasp depth before squeezing to avoid immediate push-out.
    pre_recenter_iters_done = 0
    recenter_settle_steps = max(1, int(round(cfg.recenter_settle_time / cfg.timestep)))
    _set_finger_target(0.0)
    left_contact, right_contact = _object_contact_each_finger(
        data, ids["gid_obj_set"], ids["gid_l"], ids["gid_r"], cfg.contact_penetration_threshold
    )
    while (
        (not aborted)
        and cfg.require_dual_contact
        and (left_contact ^ right_contact)
        and (pre_recenter_iters_done < int(cfg.max_recenter_iters))
    ):
        if left_contact and (not right_contact):
            tx_bias -= float(cfg.recenter_step)
        elif right_contact and (not left_contact):
            tx_bias += float(cfg.recenter_step)
        tx_bias = float(np.clip(tx_bias, -0.03, 0.03))
        data.ctrl[ids["aid_tx"]] = tx_bias
        _set_finger_target(0.0)
        _step_repeat(recenter_settle_steps)
        left_contact, right_contact = _object_contact_each_finger(
            data, ids["gid_obj_set"], ids["gid_l"], ids["gid_r"], cfg.contact_penetration_threshold
        )
        pre_recenter_iters_done += 1

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
        data, ids["gid_obj_set"], ids["gid_l"], ids["gid_r"], cfg.contact_penetration_threshold
    )
    recenter_iters_done = 0
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
            data, ids["gid_obj_set"], ids["gid_l"], ids["gid_r"], cfg.contact_penetration_threshold
        )
        recenter_iters_done += 1

    tighten_iters_done = 0
    tighten_settle_steps = max(1, int(round(cfg.tighten_settle_time / cfg.timestep)))
    while (
        (not aborted)
        and cfg.require_dual_contact
        and (not (left_contact and right_contact))
        and (tighten_iters_done < int(cfg.max_tighten_iters))
        and (final_close_target < tighten_cap)
    ):
        final_close_target = min(final_close_target + float(cfg.tighten_step), tighten_cap)
        _set_finger_target(final_close_target)
        _step_repeat(tighten_settle_steps)
        left_contact, right_contact = _object_contact_each_finger(
            data, ids["gid_obj_set"], ids["gid_l"], ids["gid_r"], cfg.contact_penetration_threshold
        )
        if left_contact ^ right_contact:
            if left_contact and (not right_contact):
                tx_bias -= float(cfg.recenter_step)
            elif right_contact and (not left_contact):
                tx_bias += float(cfg.recenter_step)
            tx_bias = float(np.clip(tx_bias, -0.03, 0.03))
            data.ctrl[ids["aid_tx"]] = tx_bias
            _set_finger_target(final_close_target)
            _step_repeat(recenter_settle_steps)
            left_contact, right_contact = _object_contact_each_finger(
                data, ids["gid_obj_set"], ids["gid_l"], ids["gid_r"], cfg.contact_penetration_threshold
            )
        tighten_iters_done += 1

    if lock_active and bool(cfg.force_unlock_before_perturb):
        lock_active = False
        lock_forced_release = True
        lock_release_step = int(step_counter)
        lock_release_time_s = float(data.time)
        if obj_qvel_adr >= 0:
            data.qvel[obj_qvel_adr : obj_qvel_adr + 6] = 0.0
        mujoco.mj_forward(model, data)

    validation_steps = max(1, int(round(float(cfg.grasp_validation_time) / float(cfg.timestep))))
    any_contact_streak = 0
    dual_contact_streak = 0
    max_any_contact_streak = 0
    max_dual_contact_streak = 0
    for _ in range(validation_steps):
        _set_finger_target(final_close_target)
        if not _step_once():
            break
        left_contact, right_contact = _object_contact_each_finger(
            data, ids["gid_obj_set"], ids["gid_l"], ids["gid_r"], cfg.contact_penetration_threshold
        )
        in_any_contact = bool(left_contact or right_contact)
        in_dual_contact = bool(left_contact and right_contact)
        any_contact_streak = (any_contact_streak + 1) if in_any_contact else 0
        dual_contact_streak = (dual_contact_streak + 1) if in_dual_contact else 0
        max_any_contact_streak = max(int(max_any_contact_streak), int(any_contact_streak))
        max_dual_contact_streak = max(int(max_dual_contact_streak), int(dual_contact_streak))

    min_any_cfg = int(cfg.min_any_contact_steps_for_grasp)
    if bool(inner.get("is_thin_support", False)):
        min_any_cfg = min(min_any_cfg, int(cfg.min_any_contact_steps_for_thin_support))
    min_any_contact_steps = max(1, min(min_any_cfg, int(validation_steps)))
    min_dual_contact_steps = max(1, min(int(cfg.min_dual_contact_steps_for_grasp), int(validation_steps)))
    contact_after_grasp = bool(max_any_contact_streak >= min_any_contact_steps)
    dual_contact_after_grasp = bool(max_dual_contact_streak >= min_dual_contact_steps)
    rel_after_grasp = data.xpos[ids["bid_obj"]].copy() - data.xpos[ids["bid_gripper"]].copy()
    grasp_rel_shift = float(np.linalg.norm(rel_after_grasp - rel0))
    use_dual_contact_criterion = bool(cfg.require_dual_contact)
    if bool(cfg.relax_dual_contact_for_thin_support) and bool(inner.get("is_thin_support", False)):
        use_dual_contact_criterion = False
    contact_ok = dual_contact_after_grasp if use_dual_contact_criterion else contact_after_grasp
    stable_after_grasp = bool(contact_ok and grasp_rel_shift <= cfg.escape_distance)

    grasped_state = _snapshot_state(data)
    rel_ref = rel_after_grasp.copy()

    axis_settings = {
        "depth": (ids["aid_tx"], float(cfg.depth_speed)),
        "width": (ids["aid_ty"], float(cfg.width_speed)),
        "height": (ids["aid_tz"], float(cfg.height_speed)),
    }

    perturb_results: Dict[str, Dict[str, Any]] = {}
    run_perturbation = bool((not cfg.require_grasp_before_perturb) or stable_after_grasp)
    for axis_name, (act_id, speed) in axis_settings.items():
        if aborted:
            break

        move_target = float(np.clip(speed * cfg.perturb_time, -cfg.max_move, cfg.max_move))
        if not run_perturbation:
            zero_leg = {
                "outward": {"max_relative_shift_m": 0.0, "lost_contact_steps_max": 0, "had_contact": False},
                "return": {"max_relative_shift_m": 0.0, "lost_contact_steps_max": 0, "had_contact": False},
            }
            perturb_results[axis_name] = {
                "speed_mps": speed,
                "move_target_m": move_target,
                "max_relative_shift_m": float(grasp_rel_shift),
                "lost_contact_steps": int(cfg.min_loss_contact_steps),
                "lost_contact_steps_max": int(cfg.min_loss_contact_steps),
                "had_contact": bool(contact_after_grasp),
                "escaped": True,
                "escaped_positive": True,
                "escaped_negative": True,
                "escaped_return": True,
                "phase": {"positive": zero_leg, "negative": zero_leg},
                "returned_to_origin": True,
                "skipped_not_grasped": True,
            }
            continue

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

        def _make_empty_excursion() -> Dict[str, Any]:
            return {
                "max_relative_shift_m": 0.0,
                "lost_contact_steps_max": 0,
                "had_contact": False,
                "escaped": False,
                "escaped_outward": False,
                "escaped_return": False,
                "returned_to_origin": False,
                "phase": {
                    "outward": {"max_relative_shift_m": 0.0, "lost_contact_steps_max": 0, "had_contact": False},
                    "return": {"max_relative_shift_m": 0.0, "lost_contact_steps_max": 0, "had_contact": False},
                },
            }

        def _run_excursion(signed_target: float) -> Dict[str, Any]:
            nonlocal aborted
            rec = _make_empty_excursion()
            if aborted:
                return rec

            _restore_state(model, data, grasped_state)
            data.ctrl[ids["aid_tx"]] = tx_bias
            data.ctrl[ids["aid_ty"]] = 0.0
            data.ctrl[ids["aid_tz"]] = 0.0
            _set_finger_target(final_close_target)

            lost_contact_steps = 0
            dual_contact_loss_steps = 0

            def _update_contact_and_shift(phase_key: str, current_offset: float) -> None:
                nonlocal lost_contact_steps, dual_contact_loss_steps, tx_bias, final_close_target
                left_contact_now, right_contact_now = _object_contact_each_finger(
                    data, ids["gid_obj_set"], ids["gid_l"], ids["gid_r"], cfg.contact_penetration_threshold
                )
                in_contact = bool(left_contact_now or right_contact_now)
                rec["had_contact"] = bool(rec["had_contact"] or in_contact)
                rec["phase"][phase_key]["had_contact"] = bool(rec["phase"][phase_key]["had_contact"] or in_contact)

                lost_contact_steps = 0 if in_contact else (lost_contact_steps + 1)
                rec["lost_contact_steps_max"] = max(int(rec["lost_contact_steps_max"]), lost_contact_steps)
                rec["phase"][phase_key]["lost_contact_steps_max"] = max(
                    int(rec["phase"][phase_key]["lost_contact_steps_max"]), lost_contact_steps
                )

                rel_now = data.xpos[ids["bid_obj"]] - data.xpos[ids["bid_gripper"]]
                rel_shift = float(np.linalg.norm(rel_now - rel_ref))
                rec["max_relative_shift_m"] = max(float(rec["max_relative_shift_m"]), rel_shift)
                rec["phase"][phase_key]["max_relative_shift_m"] = max(
                    float(rec["phase"][phase_key]["max_relative_shift_m"]), rel_shift
                )

                if cfg.require_dual_contact and cfg.maintain_dual_contact_during_perturb:
                    has_dual = bool(left_contact_now and right_contact_now)
                    dual_contact_loss_steps = 0 if has_dual else (dual_contact_loss_steps + 1)
                    if dual_contact_loss_steps >= int(cfg.maintain_contact_grace_steps):
                        if left_contact_now and (not right_contact_now):
                            tx_bias -= float(cfg.recenter_step)
                        elif right_contact_now and (not left_contact_now):
                            tx_bias += float(cfg.recenter_step)
                        tx_bias = float(np.clip(tx_bias, -0.03, 0.03))

                        if final_close_target < maintain_cap:
                            final_close_target = min(
                                final_close_target + float(cfg.maintain_tighten_step),
                                maintain_cap,
                            )
                        _set_finger_target(final_close_target)
                        _set_axis_offset(current_offset)

            def _run_leg(
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
                    _update_contact_and_shift(phase_key, offset)

                for _ in range(max(0, int(hold_steps))):
                    if aborted:
                        return False
                    _set_axis_offset(end_offset)
                    if not _step_once():
                        return False
                    _update_contact_and_shift(phase_key, end_offset)
                return True

            ok = _run_leg("outward", 0.0, signed_target, steps_perturb, steps_hold)
            if ok:
                ok = _run_leg("return", signed_target, 0.0, steps_perturb, steps_hold)
            rec["returned_to_origin"] = bool(ok and (not aborted))

            rec["escaped_outward"] = bool(
                (int(rec["phase"]["outward"]["lost_contact_steps_max"]) >= cfg.min_loss_contact_steps)
                or (float(rec["phase"]["outward"]["max_relative_shift_m"]) > cfg.escape_distance)
            )
            rec["escaped_return"] = bool(
                (int(rec["phase"]["return"]["lost_contact_steps_max"]) >= cfg.min_loss_contact_steps)
                or (float(rec["phase"]["return"]["max_relative_shift_m"]) > cfg.escape_distance)
            )
            rec["escaped"] = bool(rec["escaped_outward"] or rec["escaped_return"])
            return rec

        positive_rec = _run_excursion(+move_target)
        negative_rec = _run_excursion(-move_target)

        max_rel_shift = max(
            float(positive_rec["max_relative_shift_m"]),
            float(negative_rec["max_relative_shift_m"]),
        )
        lost_contact_steps_max = max(
            int(positive_rec["lost_contact_steps_max"]),
            int(negative_rec["lost_contact_steps_max"]),
        )
        had_contact = bool(positive_rec["had_contact"] or negative_rec["had_contact"])
        escaped_positive = bool(positive_rec["escaped"])
        escaped_negative = bool(negative_rec["escaped"])
        escaped_return = bool(positive_rec["escaped_return"] or negative_rec["escaped_return"])
        escaped = bool(escaped_positive or escaped_negative)
        phase_stats: Dict[str, Dict[str, Any]] = {
            "positive": positive_rec["phase"],
            "negative": negative_rec["phase"],
        }
        did_return = bool(positive_rec["returned_to_origin"] and negative_rec["returned_to_origin"])

        perturb_results[axis_name] = {
            "speed_mps": speed,
            "move_target_m": move_target,
            "max_relative_shift_m": max_rel_shift,
            "lost_contact_steps": int(lost_contact_steps_max),
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
        "pad_collision_scale": inner.get("pad_collision_scale"),
        "object_collision_scale": inner.get("object_collision_scale"),
        "object_collision_mode": inner.get("object_collision_mode"),
        "object_collision_auto_reason": inner.get("object_collision_auto_reason"),
        "object_collision_auto_threshold": inner.get("object_collision_auto_threshold"),
        "object_convex_hull_ratio": inner.get("object_convex_hull_ratio"),
        "object_collision_proxy_count": int(inner.get("object_collision_proxy_count", 0)),
        "object_collision_proxy_radius_m": inner.get("object_collision_proxy_radius_m"),
        "contact_margin": inner.get("contact_margin"),
        "contact_penetration_threshold_m": float(cfg.contact_penetration_threshold),
        "initial_gap_m": inner["initial_gap"],
        "close_each_m": inner["close_each"],
        "target_close_each_m": inner.get("target_close_each"),
        "tighten_cap_m": inner.get("tighten_cap"),
        "maintain_cap_m": maintain_cap,
        "original_span_m": inner.get("original_span_m"),
        "effective_span_m": inner.get("effective_span_m"),
        "object_extent_along_depth_m": inner.get("object_extent_along_depth_m"),
        "support_extent_m": inner.get("support_extent_m"),
        "support_kind": inner.get("support_kind"),
        "support_vertex_count": inner.get("support_vertex_count"),
        "is_thin_support": bool(inner.get("is_thin_support", False)),
        "contact_endpoint_outside_projection": bool(inner.get("contact_endpoint_outside_projection", False)),
        "span_adjusted": bool(inner.get("span_adjusted", False)),
        "opening_extra_m": inner.get("opening_extra_m"),
        "init_overlap_retry_count": int(inner.get("init_overlap_retry_count", 0)),
        "had_start_contact_overlap": bool(inner.get("had_start_contact_overlap", False)),
        "final_close_target_m": final_close_target,
        "tx_bias_m": tx_bias,
        "pre_recenter_iters_done": int(pre_recenter_iters_done),
        "recenter_iters_done": int(recenter_iters_done),
        "tighten_iters_done": int(tighten_iters_done),
        "object_lock_until_dual_contact": bool(cfg.lock_object_until_dual_contact),
        "object_lock_released_by_dual_contact": bool(lock_released_by_dual_contact),
        "object_lock_forced_release": bool(lock_forced_release),
        "object_lock_release_step": int(lock_release_step),
        "object_lock_release_time_s": float(lock_release_time_s),
        "grasp_validation_steps": int(validation_steps),
        "max_any_contact_streak": int(max_any_contact_streak),
        "max_dual_contact_streak": int(max_dual_contact_streak),
        "min_any_contact_steps_used": int(min_any_contact_steps),
        "min_dual_contact_steps_used": int(min_dual_contact_steps),
        "use_dual_contact_criterion": bool(use_dual_contact_criterion),
        "stable_after_grasp": stable_after_grasp,
        "contact_after_grasp": contact_after_grasp,
        "dual_contact_after_grasp": dual_contact_after_grasp,
        "grasp_relative_shift_m": grasp_rel_shift,
        "perturbation_skipped_not_grasped": bool(not run_perturbation),
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


def _result_quality_key(result: Dict[str, Any]) -> tuple[int, int, int, int, float]:
    return (
        1 if (not bool(result.get("any_escape", True))) else 0,
        1 if bool(result.get("stable_after_grasp", False)) else 0,
        1 if bool(result.get("dual_contact_after_grasp", False)) else 0,
        1 if bool(result.get("contact_after_grasp", False)) else 0,
        -float(result.get("grasp_relative_shift_m", 1e9)),
    )


def _should_auto_pair_rescue(best_result: Dict[str, Any], cfg: SimConfig) -> bool:
    if not bool(cfg.auto_pair_rescue_on_no_contact):
        return False
    if bool(cfg.auto_pair_rescue_require_thin_support) and (not bool(best_result.get("is_thin_support", False))):
        return False
    if bool(best_result.get("contact_after_grasp", False)):
        return False
    return int(best_result.get("max_any_contact_streak", 0)) <= 0


def _select_best_grasp_candidate_from_html(
    html_path: str,
    cfg: SimConfig,
) -> tuple[TopGraspData, Dict[str, Any]]:
    candidates = parse_ranked_grasp_candidates_from_html(
        html_path=html_path,
        unit_scale=cfg.unit_scale,
        max_candidates=1,
    )
    best_rank = 0
    tested = 1
    best_grasp = candidates[0]
    best_result = simulate_from_parsed_grasp(best_grasp, cfg=cfg)
    best_key = _result_quality_key(best_result)

    need_fallback = bool(cfg.enable_pair_fallback) and (
        bool(best_result.get("any_escape", True)) or (not bool(best_result.get("stable_after_grasp", False)))
    )
    need_rescue = (not bool(cfg.enable_pair_fallback)) and _should_auto_pair_rescue(best_result, cfg)
    used_limit = 1
    if need_fallback or need_rescue:
        scan_cap = int(cfg.pair_fallback_max_candidates) if need_fallback else int(cfg.auto_pair_rescue_max_candidates)
        scan_cap = max(2, int(scan_cap))
        scan_candidates = parse_ranked_grasp_candidates_from_html(
            html_path=html_path,
            unit_scale=cfg.unit_scale,
            max_candidates=scan_cap,
        )
        used_limit = int(scan_cap)
        for rank, cand in enumerate(scan_candidates[1:], start=1):
            tested += 1
            cand_res = simulate_from_parsed_grasp(cand, cfg=cfg)
            cand_key = _result_quality_key(cand_res)
            if cand_key > best_key:
                best_grasp = cand
                best_result = cand_res
                best_key = cand_key
                best_rank = rank
            if (best_key[0] == 1) and (best_key[1] == 1):
                break

    best_result = dict(best_result)
    best_result["pair_fallback_used"] = bool(best_rank != 0)
    best_result["pair_fallback_selected_rank"] = int(best_rank)
    best_result["pair_fallback_candidates_tested"] = int(tested)
    best_result["pair_rescue_triggered"] = bool(need_rescue)
    best_result["pair_rescue_max_candidates"] = int(used_limit)
    return best_grasp, best_result


def simulate_from_html(html_path: str, cfg: Optional[SimConfig] = None) -> Dict[str, Any]:
    cfg = cfg or SimConfig()
    _, best = _select_best_grasp_candidate_from_html(html_path, cfg)
    return best


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


def record_offscreen_from_parsed_grasp(
    grasp: TopGraspData,
    video_path: str,
    cfg: Optional[SimConfig] = None,
    *,
    video_fps: int = 30,
    video_width: int = 960,
    video_height: int = 540,
    camera_azimuth: float = 135.0,
    camera_elevation: float = -25.0,
    camera_distance_scale: float = 3.0,
    split_dual_view: bool = True,
    opposite_azimuth_offset: float = 180.0,
    camera_track_object: bool = False,
    show_contact_points: bool = True,
    show_contact_forces: bool = False,
) -> Dict[str, Any]:
    """
    Record MuJoCo simulation directly from an offscreen renderer (no viewer window).
    """
    _require_mujoco()
    import cv2

    cfg = cfg or SimConfig()
    model, data, inner, ids = _init_simulation(grasp, cfg)

    out_path = Path(video_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dual_view = bool(split_dual_view)
    out_w = int(video_width)
    out_h = int(video_height)
    if dual_view:
        panel_w = max(1, out_w // 2)
        out_w = panel_w * 2
    else:
        panel_w = out_w

    renderer_a = mujoco.Renderer(model, height=out_h, width=panel_w)
    renderer_b = mujoco.Renderer(model, height=out_h, width=panel_w) if dual_view else None
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = float(camera_azimuth)
    cam.elevation = float(camera_elevation)
    cam.distance = max(0.35, float(camera_distance_scale) * float(model.stat.extent))
    cam.lookat[:] = data.xpos[ids["bid_obj"]]
    cam_b = None
    if dual_view:
        cam_b = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam_b)
        cam_b.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam_b.azimuth = float(camera_azimuth + opposite_azimuth_offset)
        cam_b.elevation = float(camera_elevation)
        cam_b.distance = float(cam.distance)
        cam_b.lookat[:] = data.xpos[ids["bid_obj"]]

    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1 if show_contact_points else 0
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1 if show_contact_forces else 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(video_fps), (out_w, out_h))
    if not writer.isOpened():
        renderer_a.close()
        if renderer_b is not None:
            renderer_b.close()
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

    frame_step = max(1, int(round(1.0 / (max(1, int(video_fps)) * float(cfg.timestep)))))
    step_idx = 0

    def _render_combined_rgb() -> np.ndarray:
        if camera_track_object:
            lookat = data.xpos[ids["bid_obj"]]
            cam.lookat[:] = lookat
            if cam_b is not None:
                cam_b.lookat[:] = lookat
        renderer_a.update_scene(data, camera=cam, scene_option=opt)
        rgb_a = renderer_a.render()
        if renderer_b is None or cam_b is None:
            return rgb_a
        renderer_b.update_scene(data, camera=cam_b, scene_option=opt)
        rgb_b = renderer_b.render()
        return np.concatenate((rgb_a, rgb_b), axis=1)

    def _capture_frame(_: Any, __: Any) -> None:
        nonlocal step_idx
        step_idx += 1
        if step_idx % frame_step != 0:
            return
        rgb = _render_combined_rgb()
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    try:
        # initial frame
        rgb0 = _render_combined_rgb()
        writer.write(cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR))

        result = _run_simulation_core(
            model=model,
            data=data,
            grasp=grasp,
            cfg=cfg,
            inner=inner,
            ids=ids,
            on_step=_capture_frame,
        )

        # final settle frames
        rgbf = _render_combined_rgb()
        for _ in range(max(1, int(video_fps * 0.2))):
            writer.write(cv2.cvtColor(rgbf, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        renderer_a.close()
        if renderer_b is not None:
            renderer_b.close()

    result["video_path"] = str(out_path)
    result["video_dual_view"] = bool(dual_view)
    result["video_resolution"] = [int(out_w), int(out_h)]
    return result


def record_offscreen_from_html(
    html_path: str,
    video_path: str,
    cfg: Optional[SimConfig] = None,
    *,
    video_fps: int = 30,
    video_width: int = 960,
    video_height: int = 540,
    camera_azimuth: float = 135.0,
    camera_elevation: float = -25.0,
    camera_distance_scale: float = 3.0,
    split_dual_view: bool = True,
    opposite_azimuth_offset: float = 180.0,
    camera_track_object: bool = False,
    show_contact_points: bool = True,
    show_contact_forces: bool = False,
) -> Dict[str, Any]:
    cfg = cfg or SimConfig()
    best_grasp, best_eval = _select_best_grasp_candidate_from_html(html_path, cfg)
    recorded = record_offscreen_from_parsed_grasp(
        best_grasp,
        video_path=video_path,
        cfg=cfg,
        video_fps=video_fps,
        video_width=video_width,
        video_height=video_height,
        camera_azimuth=camera_azimuth,
        camera_elevation=camera_elevation,
        camera_distance_scale=camera_distance_scale,
        split_dual_view=split_dual_view,
        opposite_azimuth_offset=opposite_azimuth_offset,
        camera_track_object=camera_track_object,
        show_contact_points=show_contact_points,
        show_contact_forces=show_contact_forces,
    )
    recorded["pair_fallback_used"] = bool(best_eval.get("pair_fallback_used", False))
    recorded["pair_fallback_selected_rank"] = int(best_eval.get("pair_fallback_selected_rank", 0))
    recorded["pair_fallback_candidates_tested"] = int(best_eval.get("pair_fallback_candidates_tested", 1))
    recorded["pair_rescue_triggered"] = bool(best_eval.get("pair_rescue_triggered", False))
    recorded["pair_rescue_max_candidates"] = int(best_eval.get("pair_rescue_max_candidates", 1))
    return recorded


def run_from_html(
    html_path: str,
    *,
    depth_speed: float = 0.6,
    width_speed: float = 0.6,
    height_speed: float = 0.6,
    friction: float = 0.8,
    grasp_force: float = 90.0,
    gravity_z: float = 0.0,
    contact_margin: float = 0.0,
    pad_collision_scale: float = 0.94,
    object_collision_scale: float = 0.96,
    contact_penetration_threshold: float = 5e-5,
    object_collision_mode: str = "auto",
    object_collision_auto_hull_ratio_threshold: float = 1.15,
    object_collision_auto_use_spheres_when_hull_unknown: bool = True,
    object_collision_sphere_count: int = 180,
    object_collision_sphere_radius_scale: float = 0.55,
    object_collision_sphere_min_radius: float = 4e-4,
    object_collision_sphere_max_radius: float = 6e-3,
    lock_object_until_dual_contact: bool = True,
    force_unlock_before_perturb: bool = True,
    pre_clearance: float = 0.002,
    approach_start_extra_gap: float = 0.008,
    squeeze_extra: float = 0.0015,
    require_dual_contact: bool = True,
    tighten_step: float = 0.001,
    max_tighten_iters: int = 80,
    max_tighten_overtravel: float = 0.0012,
    grasp_validation_time: float = 0.03,
    min_any_contact_steps_for_grasp: int = 20,
    min_dual_contact_steps_for_grasp: int = 20,
    min_any_contact_steps_for_thin_support: int = 8,
    require_grasp_before_perturb: bool = False,
    relax_dual_contact_for_thin_support: bool = True,
    maintain_dual_contact_during_perturb: bool = True,
    maintain_contact_grace_steps: int = 2,
    maintain_tighten_step: float = 0.0001,
    auto_expand_span_for_parallel_jaw: bool = True,
    min_span_ratio_to_object_extent: float = 0.35,
    min_span_abs: float = 0.003,
    local_support_min_vertices: int = 20,
    opening_retry_step: float = 0.004,
    max_initial_opening_extra: float = 0.05,
    max_opening_retries: int = 8,
    enable_pair_fallback: bool = False,
    pair_fallback_max_candidates: int = 64,
    auto_pair_rescue_on_no_contact: bool = False,
    auto_pair_rescue_max_candidates: int = 64,
    auto_pair_rescue_require_thin_support: bool = True,
    perturb_time: float = 0.08,
    hold_time: float = 0.08,
    open_viewer: bool = False,
    realtime_viewer: bool = True,
    show_contact_points: bool = True,
    show_contact_forces: bool = False,
    record_video_path: Optional[str] = None,
    record_video_fps: int = 30,
    record_video_width: int = 960,
    record_video_height: int = 540,
    record_camera_azimuth: float = 135.0,
    record_camera_elevation: float = -25.0,
    record_camera_distance_scale: float = 3.0,
    record_split_dual_view: bool = True,
    record_opposite_azimuth_offset: float = 180.0,
    record_camera_track_object: bool = False,
) -> Dict[str, Any]:
    cfg = SimConfig(
        depth_speed=depth_speed,
        width_speed=width_speed,
        height_speed=height_speed,
        friction=friction,
        grasp_force=grasp_force,
        gravity_z=gravity_z,
        contact_margin=contact_margin,
        pad_collision_scale=pad_collision_scale,
        object_collision_scale=object_collision_scale,
        contact_penetration_threshold=contact_penetration_threshold,
        object_collision_mode=object_collision_mode,
        object_collision_auto_hull_ratio_threshold=object_collision_auto_hull_ratio_threshold,
        object_collision_auto_use_spheres_when_hull_unknown=object_collision_auto_use_spheres_when_hull_unknown,
        object_collision_sphere_count=object_collision_sphere_count,
        object_collision_sphere_radius_scale=object_collision_sphere_radius_scale,
        object_collision_sphere_min_radius=object_collision_sphere_min_radius,
        object_collision_sphere_max_radius=object_collision_sphere_max_radius,
        lock_object_until_dual_contact=lock_object_until_dual_contact,
        force_unlock_before_perturb=force_unlock_before_perturb,
        pre_clearance=pre_clearance,
        approach_start_extra_gap=approach_start_extra_gap,
        squeeze_extra=squeeze_extra,
        require_dual_contact=require_dual_contact,
        tighten_step=tighten_step,
        max_tighten_iters=max_tighten_iters,
        max_tighten_overtravel=max_tighten_overtravel,
        grasp_validation_time=grasp_validation_time,
        min_any_contact_steps_for_grasp=min_any_contact_steps_for_grasp,
        min_dual_contact_steps_for_grasp=min_dual_contact_steps_for_grasp,
        min_any_contact_steps_for_thin_support=min_any_contact_steps_for_thin_support,
        require_grasp_before_perturb=require_grasp_before_perturb,
        relax_dual_contact_for_thin_support=relax_dual_contact_for_thin_support,
        maintain_dual_contact_during_perturb=maintain_dual_contact_during_perturb,
        maintain_contact_grace_steps=maintain_contact_grace_steps,
        maintain_tighten_step=maintain_tighten_step,
        auto_expand_span_for_parallel_jaw=auto_expand_span_for_parallel_jaw,
        min_span_ratio_to_object_extent=min_span_ratio_to_object_extent,
        min_span_abs=min_span_abs,
        local_support_min_vertices=local_support_min_vertices,
        opening_retry_step=opening_retry_step,
        max_initial_opening_extra=max_initial_opening_extra,
        max_opening_retries=max_opening_retries,
        enable_pair_fallback=enable_pair_fallback,
        pair_fallback_max_candidates=pair_fallback_max_candidates,
        auto_pair_rescue_on_no_contact=auto_pair_rescue_on_no_contact,
        auto_pair_rescue_max_candidates=auto_pair_rescue_max_candidates,
        auto_pair_rescue_require_thin_support=auto_pair_rescue_require_thin_support,
        perturb_time=perturb_time,
        hold_time=hold_time,
    )
    if record_video_path:
        return record_offscreen_from_html(
            html_path=html_path,
            video_path=record_video_path,
            cfg=cfg,
            video_fps=record_video_fps,
            video_width=record_video_width,
            video_height=record_video_height,
            camera_azimuth=record_camera_azimuth,
            camera_elevation=record_camera_elevation,
            camera_distance_scale=record_camera_distance_scale,
            split_dual_view=record_split_dual_view,
            opposite_azimuth_offset=record_opposite_azimuth_offset,
            camera_track_object=record_camera_track_object,
            show_contact_points=show_contact_points,
            show_contact_forces=show_contact_forces,
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
