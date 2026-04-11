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


def parse_top_grasp_from_html(html_path: str | Path, unit_scale: float = 1e-3) -> TopGraspData:
    """
    Parse a feasible-grasp Plotly HTML and extract:
    - object mesh (first large mesh trace)
    - top-ranked grasp pair (first `pair ...` marker trace)
    - 3 local grasp axes (depth/width/height)

    `unit_scale=1e-3` converts mm (grasp output) -> m (MuJoCo).
    """
    path = Path(html_path)
    html_text = path.read_text(encoding="utf-8", errors="ignore")
    traces = _load_plotly_traces(html_text)

    mesh_candidates: List[Dict[str, Any]] = []
    for tr in traces:
        if tr.get("type") != "mesh3d":
            continue
        if not all(k in tr for k in ("x", "y", "z", "i", "j", "k")):
            continue
        vx = _decode_plotly_array(tr["x"])
        if len(vx) >= 30:
            mesh_candidates.append(tr)
    if not mesh_candidates:
        raise ValueError("Could not find object mesh trace in HTML.")

    object_mesh = max(mesh_candidates, key=lambda t: len(_decode_plotly_array(t["x"])))

    vx = _decode_plotly_array(object_mesh["x"]).astype(float) * unit_scale
    vy = _decode_plotly_array(object_mesh["y"]).astype(float) * unit_scale
    vz = _decode_plotly_array(object_mesh["z"]).astype(float) * unit_scale
    mesh_vertices = np.column_stack((vx, vy, vz))

    fi = _decode_plotly_array(object_mesh["i"]).astype(np.int64)
    fj = _decode_plotly_array(object_mesh["j"]).astype(np.int64)
    fk = _decode_plotly_array(object_mesh["k"]).astype(np.int64)
    mesh_faces = np.column_stack((fi, fj, fk))

    top_marker = None
    best_pair_idx = None
    for tr in traces:
        name = str(tr.get("name", ""))
        mode = str(tr.get("mode", ""))
        if tr.get("type") == "scatter3d" and "markers" in mode and name.startswith("pair "):
            m = re.match(r"pair\s+(\d+)\b", name)
            pidx = int(m.group(1)) if m else 10**9
            if top_marker is None or pidx < best_pair_idx:
                top_marker = tr
                best_pair_idx = pidx
    if top_marker is None:
        raise ValueError("Could not find top grasp pair marker trace in HTML.")

    pair_name = str(top_marker.get("name", "pair 0"))
    legend_group = str(top_marker.get("legendgroup", pair_name))
    marker_pts = _to_points_xyz(top_marker, unit_scale)
    if len(marker_pts) == 0:
        raise ValueError("Top grasp marker trace has no valid point.")
    contact_i = marker_pts[0]

    pad_i_trace = None
    pad_j_trace = None
    for tr in traces:
        if tr.get("type") != "mesh3d":
            continue
        if str(tr.get("legendgroup", "")) != legend_group:
            continue
        tr_name = str(tr.get("name", "")).lower()
        if "pad_i" in tr_name:
            pad_i_trace = tr
        elif "pad_j" in tr_name:
            pad_j_trace = tr

    if pad_i_trace is not None and pad_j_trace is not None:
        pad_i_pts = _to_points_xyz(pad_i_trace, unit_scale)
        pad_j_pts = _to_points_xyz(pad_j_trace, unit_scale)
        if len(pad_i_pts) > 0 and len(pad_j_pts) > 0:
            contact_i = np.mean(pad_i_pts, axis=0)
            contact_j = np.mean(pad_j_pts, axis=0)
        else:
            contact_j = np.copy(contact_i)
    else:
        contact_j = np.copy(contact_i)

    # Fallback to centerline if pad_i/pad_j-based contacts are not valid enough.
    if float(np.linalg.norm(contact_j - contact_i)) < 1e-7:
        centerline_trace = None
        for tr in traces:
            if tr.get("type") != "scatter3d":
                continue
            if str(tr.get("legendgroup", "")) != legend_group:
                continue
            mode = str(tr.get("mode", ""))
            if "lines" not in mode:
                continue
            if "centerline" in str(tr.get("name", "")).lower():
                centerline_trace = tr
                break
        if centerline_trace is None:
            raise ValueError(f"Could not find centerline trace for legend group '{legend_group}'.")

        line_pts = _to_points_xyz(centerline_trace, unit_scale)
        if len(line_pts) < 2:
            raise ValueError("Centerline trace does not have enough valid points.")

        d0 = float(np.linalg.norm(line_pts[0] - contact_i))
        d1 = float(np.linalg.norm(line_pts[-1] - contact_i))
        contact_j = line_pts[-1] if d1 >= d0 else line_pts[0]

    depth_axis = _unit(contact_j - contact_i, [1.0, 0.0, 0.0])
    grasp_center = 0.5 * (contact_i + contact_j)

    if pad_i_trace is not None:
        pad_points = _to_points_xyz(pad_i_trace, unit_scale)
        width_axis, height_axis = _principal_axes_from_pad(pad_points, depth_axis)
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
