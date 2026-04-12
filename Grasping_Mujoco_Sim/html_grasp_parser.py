from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np


_PLOTLY_DTYPE_MAP: Dict[str, np.dtype] = {
    "f4": np.dtype("<f4"),
    "f8": np.dtype("<f8"),
    "i1": np.dtype("<i1"),
    "i2": np.dtype("<i2"),
    "i4": np.dtype("<i4"),
    "i8": np.dtype("<i8"),
    "u1": np.dtype("<u1"),
    "u2": np.dtype("<u2"),
    "u4": np.dtype("<u4"),
    "u8": np.dtype("<u8"),
}


@dataclass
class TopGraspData:
    pair_name: str
    legend_group: str
    source_html: str
    mesh_vertices: np.ndarray
    mesh_faces: np.ndarray
    contact_i: np.ndarray
    contact_j: np.ndarray
    grasp_center: np.ndarray
    depth_axis: np.ndarray
    width_axis: np.ndarray
    height_axis: np.ndarray


def _unit(v: np.ndarray, fallback: Sequence[float]) -> np.ndarray:
    out = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(out))
    if n < 1e-12:
        return np.asarray(fallback, dtype=float)
    return out / n


def _infer_default_mesh_root(html_path: Path) -> Path:
    p = html_path.resolve()
    for parent in [p.parent] + list(p.parents):
        if parent.name == "Grasping_Results_SurfaceContact":
            cand = parent.parent / "models_target" / "models_cad"
            if cand.exists():
                return cand
    return (Path.cwd() / "models_target" / "models_cad").resolve()


def _extract_object_name_from_html_filename(path: Path) -> str:
    stem = path.stem
    m = re.match(r"1-grasping_pairs_feasible_(.+?)_\d+sol$", stem)
    if m:
        return m.group(1)
    m2 = re.match(r"1-grasping_pairs_feasible_(.+)$", stem)
    if m2:
        return m2.group(1)
    raise ValueError(f"Could not parse object key from HTML filename: {path.name}")


def _resolve_external_mesh_path(
    *,
    html_path: Path,
    mesh_root: str | Path | None,
) -> Path:
    root = _infer_default_mesh_root(html_path) if mesh_root is None else Path(mesh_root).resolve()
    category = html_path.parent.name
    obj_key = _extract_object_name_from_html_filename(html_path)
    cat_dir = root / category
    if not cat_dir.exists():
        raise ValueError(f"Mesh category folder not found: {cat_dir}")

    for ext in (".ply", ".obj"):
        cand = cat_dir / f"{obj_key}{ext}"
        if cand.exists():
            return cand

    fuzzy = sorted(
        p for p in cat_dir.glob(f"{obj_key}.*") if p.suffix.lower() in {".ply", ".obj"}
    )
    if fuzzy:
        return fuzzy[0]

    raise ValueError(
        f"Could not find mesh file for '{obj_key}' in '{cat_dir}' (.ply/.obj expected)."
    )


def _load_external_mesh(
    *,
    html_path: Path,
    unit_scale: float,
    mesh_root: str | Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    mesh_path = _resolve_external_mesh_path(html_path=html_path, mesh_root=mesh_root)
    try:
        import trimesh  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "trimesh is required to load external obj/ply meshes."
        ) from exc

    loaded = trimesh.load_mesh(str(mesh_path), process=False)
    if isinstance(loaded, trimesh.Scene):
        geom = [g for g in loaded.geometry.values() if hasattr(g, "vertices") and hasattr(g, "faces")]
        if not geom:
            raise ValueError(f"Scene mesh has no valid geometry: {mesh_path}")
        mesh = trimesh.util.concatenate(geom)
    else:
        mesh = loaded
    vertices = np.asarray(mesh.vertices, dtype=float) * float(unit_scale)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"Invalid mesh vertices from: {mesh_path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"Invalid mesh faces from: {mesh_path}")
    return vertices, faces


def _decode_plotly_array(arr: Any) -> np.ndarray:
    if isinstance(arr, dict) and "bdata" in arr and "dtype" in arr:
        dtype_code = str(arr["dtype"]).strip()
        dtype = _PLOTLY_DTYPE_MAP.get(dtype_code)
        if dtype is None:
            raise ValueError(f"Unsupported Plotly typed-array dtype: {dtype_code}")
        raw = base64.b64decode(arr["bdata"])
        return np.frombuffer(raw, dtype=dtype)

    return np.asarray(arr)


def _extract_plotly_data_array_text(html_text: str) -> str:
    anchor = html_text.find("Plotly.newPlot")
    if anchor < 0:
        raise ValueError("Could not find Plotly.newPlot(...) block in HTML.")

    i = html_text.find("[", anchor)
    if i < 0:
        raise ValueError("Could not find Plotly data array in HTML.")

    depth = 0
    in_string = False
    escaped = False
    quote_char = ""

    for j in range(i, len(html_text)):
        ch = html_text[j]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote_char:
                in_string = False
            continue

        if ch == '"' or ch == "'":
            in_string = True
            quote_char = ch
            continue

        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return html_text[i : j + 1]

    raise ValueError("Failed to parse Plotly data array: unmatched brackets.")


def _load_plotly_traces(html_text: str) -> List[Dict[str, Any]]:
    data_text = _extract_plotly_data_array_text(html_text)
    try:
        loaded = json.loads(data_text)
    except json.JSONDecodeError:
        normalized = (
            data_text.replace("NaN", "null")
            .replace("Infinity", "1e308")
            .replace("-Infinity", "-1e308")
        )
        loaded = json.loads(normalized)

    if not isinstance(loaded, list):
        raise ValueError("Plotly data block is not a list.")
    return loaded


def _to_points_xyz(trace: Dict[str, Any], unit_scale: float) -> np.ndarray:
    x = _decode_plotly_array(trace.get("x", []))
    y = _decode_plotly_array(trace.get("y", []))
    z = _decode_plotly_array(trace.get("z", []))
    n = min(len(x), len(y), len(z))
    if n == 0:
        return np.empty((0, 3), dtype=float)
    pts = np.column_stack((x[:n], y[:n], z[:n])).astype(float)
    pts *= unit_scale
    finite = np.isfinite(pts).all(axis=1)
    return pts[finite]


def _principal_axes_from_pad(pad_points: np.ndarray, depth_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(pad_points) < 4:
        fallback_w = _unit(np.cross(depth_axis, np.array([0.0, 0.0, 1.0])), [0.0, 1.0, 0.0])
        fallback_h = _unit(np.cross(depth_axis, fallback_w), [0.0, 0.0, 1.0])
        return fallback_w, fallback_h

    centered = pad_points - pad_points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh.copy()  # rows: principal directions
    projs = centered @ axes.T
    extents = np.ptp(projs, axis=0)

    depth_idx = int(np.argmax(np.abs(axes @ depth_axis)))
    remaining = [idx for idx in (0, 1, 2) if idx != depth_idx]
    if extents[remaining[0]] >= extents[remaining[1]]:
        width_idx, height_idx = remaining[0], remaining[1]
    else:
        width_idx, height_idx = remaining[1], remaining[0]

    width_raw = axes[width_idx]
    width_raw = width_raw - depth_axis * float(np.dot(width_raw, depth_axis))
    width_axis = _unit(width_raw, [0.0, 1.0, 0.0])

    height_axis = np.cross(depth_axis, width_axis)
    if np.linalg.norm(height_axis) < 1e-9:
        height_raw = axes[height_idx]
        height_raw = height_raw - depth_axis * float(np.dot(height_raw, depth_axis))
        height_axis = _unit(height_raw, [0.0, 0.0, 1.0])
        width_axis = _unit(np.cross(height_axis, depth_axis), [0.0, 1.0, 0.0])
    else:
        height_axis = _unit(height_axis, [0.0, 0.0, 1.0])

    return width_axis, height_axis


def _is_pad_mesh(tr: Dict[str, Any]) -> bool:
    tr_name = str(tr.get("name", "")).lower()
    return ("pad_i" in tr_name) or ("pad_j" in tr_name)


def _select_object_mesh_trace(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    mesh_candidates: List[Dict[str, Any]] = []
    for tr in traces:
        if tr.get("type") != "mesh3d":
            continue
        if not all(k in tr for k in ("x", "y", "z", "i", "j", "k")):
            continue
        mesh_candidates.append(tr)
    if not mesh_candidates:
        raise ValueError("Could not find object mesh trace in HTML.")

    non_pad = [tr for tr in mesh_candidates if not _is_pad_mesh(tr)]
    root_mesh = [tr for tr in non_pad if not str(tr.get("legendgroup", "")).strip()]
    if root_mesh:
        return max(root_mesh, key=lambda t: len(_decode_plotly_array(t["x"])))
    if non_pad:
        return max(non_pad, key=lambda t: len(_decode_plotly_array(t["x"])))
    return max(mesh_candidates, key=lambda t: len(_decode_plotly_array(t["x"])))


def _decode_mesh_trace(object_mesh: Dict[str, Any], unit_scale: float) -> tuple[np.ndarray, np.ndarray]:
    vx = _decode_plotly_array(object_mesh["x"]).astype(float) * unit_scale
    vy = _decode_plotly_array(object_mesh["y"]).astype(float) * unit_scale
    vz = _decode_plotly_array(object_mesh["z"]).astype(float) * unit_scale
    mesh_vertices = np.column_stack((vx, vy, vz))

    fi = _decode_plotly_array(object_mesh["i"]).astype(np.int64)
    fj = _decode_plotly_array(object_mesh["j"]).astype(np.int64)
    fk = _decode_plotly_array(object_mesh["k"]).astype(np.int64)
    mesh_faces = np.column_stack((fi, fj, fk))
    return mesh_vertices, mesh_faces


def _collect_pair_marker_traces(traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    markers: List[tuple[int, int, Dict[str, Any]]] = []
    for order, tr in enumerate(traces):
        name = str(tr.get("name", ""))
        mode = str(tr.get("mode", ""))
        if tr.get("type") != "scatter3d" or "markers" not in mode or not name.startswith("pair "):
            continue
        m = re.match(r"pair\s+(\d+)\b", name)
        pidx = int(m.group(1)) if m else 10**9
        markers.append((pidx, order, tr))
    markers.sort(key=lambda x: (x[0], x[1]))
    return [m[2] for m in markers]


def _collect_legend_aux_traces(traces: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    aux: Dict[str, Dict[str, Any]] = {}
    for tr in traces:
        lg = str(tr.get("legendgroup", ""))
        if not lg:
            continue
        ent = aux.setdefault(lg, {"pad_i": None, "pad_j": None, "centerline": None})
        if tr.get("type") == "mesh3d":
            tr_name = str(tr.get("name", "")).lower()
            if "pad_i" in tr_name:
                ent["pad_i"] = tr
            elif "pad_j" in tr_name:
                ent["pad_j"] = tr
        elif tr.get("type") == "scatter3d":
            mode = str(tr.get("mode", ""))
            if ("lines" in mode) and ("centerline" in str(tr.get("name", "")).lower()):
                ent["centerline"] = tr
    return aux


def _build_grasp_from_marker_trace(
    *,
    path: Path,
    marker_trace: Dict[str, Any],
    legend_aux: Dict[str, Dict[str, Any]],
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    unit_scale: float,
) -> TopGraspData:
    pair_name = str(marker_trace.get("name", "pair 0"))
    legend_group = str(marker_trace.get("legendgroup", pair_name))
    marker_pts = _to_points_xyz(marker_trace, unit_scale)
    if len(marker_pts) == 0:
        raise ValueError("Top grasp marker trace has no valid point.")
    marker_contact = marker_pts[0]
    contact_i = marker_contact.copy()
    contact_j = marker_contact.copy()

    ent = legend_aux.get(legend_group, {})
    pad_i_trace = ent.get("pad_i")
    pad_j_trace = ent.get("pad_j")
    centerline_trace = ent.get("centerline")
    pad_i_pts = _to_points_xyz(pad_i_trace, unit_scale) if pad_i_trace is not None else np.empty((0, 3), dtype=float)
    pad_j_pts = _to_points_xyz(pad_j_trace, unit_scale) if pad_j_trace is not None else np.empty((0, 3), dtype=float)
    pad_ci = np.mean(pad_i_pts, axis=0) if len(pad_i_pts) > 0 else None
    pad_cj = np.mean(pad_j_pts, axis=0) if len(pad_j_pts) > 0 else None

    # Prefer explicit centerline endpoints (closest endpoint to marker becomes contact_i).
    used_centerline = False
    if centerline_trace is not None:
        line_pts = _to_points_xyz(centerline_trace, unit_scale)
        if len(line_pts) >= 2:
            p0 = line_pts[0]
            p1 = line_pts[-1]
            use_line = True
            if (pad_ci is not None) and (pad_cj is not None):
                d_pad = float(np.linalg.norm(pad_cj - pad_ci))
                d_line = float(np.linalg.norm(p1 - p0))
                # Some HTMLs encode centerline along pad thickness, not pad-to-pad axis.
                # In that case, centerline is too short and/or orientation mismatches pad centroids.
                if (d_pad > 1e-7) and (d_line > 1e-7):
                    ratio = d_line / d_pad
                    dir_line = _unit(p1 - p0, [1.0, 0.0, 0.0])
                    dir_pad = _unit(pad_cj - pad_ci, [1.0, 0.0, 0.0])
                    align = abs(float(np.dot(dir_line, dir_pad)))
                    if (ratio < 0.45) or (ratio > 2.2) or (align < 0.65):
                        use_line = False
            if float(np.linalg.norm(p0 - marker_contact)) <= float(np.linalg.norm(p1 - marker_contact)):
                if use_line:
                    contact_i, contact_j = p0, p1
                    used_centerline = True
            else:
                if use_line:
                    contact_i, contact_j = p1, p0
                    used_centerline = True

    # Fallback: use pad centroids if centerline is unavailable/invalid.
    if (not used_centerline) and (pad_ci is not None) and (pad_cj is not None):
        if float(np.linalg.norm(pad_cj - pad_ci)) > 1e-7:
            if float(np.linalg.norm(pad_ci - marker_contact)) <= float(np.linalg.norm(pad_cj - marker_contact)):
                contact_i, contact_j = pad_ci, pad_cj
            else:
                contact_i, contact_j = pad_cj, pad_ci

    if float(np.linalg.norm(contact_j - contact_i)) < 1e-7:
        raise ValueError(f"Could not recover valid grasp line for legend group '{legend_group}'.")

    depth_axis = _unit(contact_j - contact_i, [1.0, 0.0, 0.0])
    grasp_center = 0.5 * (contact_i + contact_j)

    if len(pad_i_pts) > 0:
        width_axis, height_axis = _principal_axes_from_pad(pad_i_pts, depth_axis)
    else:
        width_axis = _unit(np.cross(depth_axis, np.array([0.0, 0.0, 1.0])), [0.0, 1.0, 0.0])
        height_axis = _unit(np.cross(depth_axis, width_axis), [0.0, 0.0, 1.0])

    return TopGraspData(
        pair_name=pair_name,
        legend_group=legend_group,
        source_html=str(path),
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        contact_i=contact_i,
        contact_j=contact_j,
        grasp_center=grasp_center,
        depth_axis=depth_axis,
        width_axis=width_axis,
        height_axis=height_axis,
    )


def parse_ranked_grasp_candidates_from_html(
    html_path: str | Path,
    unit_scale: float = 1e-3,
    max_candidates: int | None = None,
    mesh_root: str | Path | None = None,
) -> List[TopGraspData]:
    path = Path(html_path)
    html_text = path.read_text(encoding="utf-8", errors="ignore")
    traces = _load_plotly_traces(html_text)
    mesh_vertices, mesh_faces = _load_external_mesh(
        html_path=path,
        unit_scale=unit_scale,
        mesh_root=mesh_root,
    )
    markers = _collect_pair_marker_traces(traces)
    legend_aux = _collect_legend_aux_traces(traces)

    out: List[TopGraspData] = []
    limit = int(max_candidates) if max_candidates is not None else None
    for marker in markers:
        if limit is not None and len(out) >= limit:
            break
        try:
            out.append(
                _build_grasp_from_marker_trace(
                    path=path,
                    marker_trace=marker,
                    legend_aux=legend_aux,
                    mesh_vertices=mesh_vertices,
                    mesh_faces=mesh_faces,
                    unit_scale=unit_scale,
                )
            )
        except Exception:
            # Skip malformed individual traces and keep the parser robust.
            continue
    if not out:
        raise ValueError("Could not recover any valid grasp candidates from HTML.")
    return out


def parse_top_grasp_from_html(
    html_path: str | Path,
    unit_scale: float = 1e-3,
    mesh_root: str | Path | None = None,
) -> TopGraspData:
    """
    Parse a feasible-grasp Plotly HTML and extract the highest-ranked marker grasp.

    `unit_scale=1e-3` converts mm (grasp output) -> m (MuJoCo).
    """
    return parse_ranked_grasp_candidates_from_html(
        html_path=html_path,
        unit_scale=unit_scale,
        max_candidates=1,
        mesh_root=mesh_root,
    )[0]
