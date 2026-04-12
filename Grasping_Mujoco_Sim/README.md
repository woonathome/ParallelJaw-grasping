# Grasping_Mujoco_Sim

MuJoCo-based parallel-jaw grasp simulation from feasible grasp HTML files
(`Grasping_Results_SurfaceContact/.../1-grasping_pairs_feasible_*.html`).

## What This Module Does

- Parses grasp pair pose and pad direction from Plotly HTML traces.
- Builds a lightweight MuJoCo scene (pads + object only, no robot arm model).
- Runs grasp close + 3-axis disturbance test (`depth`, `width`, `height`).
- Reports dynamic stability / escape status as a Python dict.
- Supports interactive viewer and offscreen video recording.

## Object Mesh Policy (Current)

Object mesh is selected in this order:

1. Decode object `mesh3d` from HTML.
2. Check whether that HTML mesh is watertight.
3. If watertight: use HTML mesh directly.
4. If not watertight: load original mesh from `models_target/models_cad/...`
   and align it to the HTML frame (uniform scale + centroid shift).

Notes:
- Grasp pair / pad pose always comes from HTML traces.
- `trimesh` is required for watertight check and original mesh loading.

## Main Files

- `html_grasp_parser.py`
  - Parses Plotly HTML traces.
  - Resolves object mesh (HTML watertight-first, external fallback).
- `mujoco_parallel_jaw_sim.py`
  - MuJoCo model construction, grasp simulation, perturbation, escape checks.
  - Viewer/offscreen recording and notebook UI.
- `__init__.py`
  - Public API exports.

## Environment

Recommended environment:

- Python: `3.9.x` (`sam6d`)
- Required runtime packages:
  - `mujoco`
  - `numpy`
  - `trimesh`
  - `opencv-python` (for MP4 recording)
  - `ipywidgets` (for notebook UI)

Install minimal dependencies in `sam6d`:

```bash
conda run -n sam6d pip install mujoco trimesh opencv-python ipywidgets
```

## Quick Start

```python
from Grasping_Mujoco_Sim import run_from_html

html_path = "Grasping_Results_SurfaceContact/BOP_ITODD/1-grasping_pairs_feasible_obj_000005_60sol.html"

result = run_from_html(
    html_path=html_path,
    depth_speed=0.8,
    width_speed=0.6,
    height_speed=0.6,
    friction=0.8,
    grasp_force=100.0,
)

print(result["pair_name"])
print(result["stable_after_grasp"], result["any_escape"])
```

## Viewer Run

```python
result = run_from_html(
    html_path=html_path,
    open_viewer=True,
    realtime_viewer=True,
    show_contact_points=True,
    show_contact_forces=False,
)
```

## Offscreen Video Recording

```python
result = run_from_html(
    html_path=html_path,
    record_video_path="Grasping_Mujoco_Result/sample.mp4",
    record_video_fps=30,
    record_video_width=960,
    record_video_height=540,
    record_split_dual_view=True,   # left/right split views
    show_contact_points=True,
    show_contact_forces=False,
)
```

## Candidate Pair Selection

Default behavior:

- Start from first ranked HTML pair (`max_candidates=1` initial pass).
- No fallback scan unless explicitly enabled.

Optional controls in `run_from_html(...)`:

- `enable_pair_fallback`
- `pair_fallback_max_candidates`
- `auto_pair_rescue_on_no_contact`
- `auto_pair_rescue_max_candidates`
- `auto_pair_rescue_require_thin_support`

## Notebook UI

```python
from Grasping_Mujoco_Sim import launch_notebook_ui

launch_notebook_ui(
    "Grasping_Results_SurfaceContact/BOP_ITODD/1-grasping_pairs_feasible_obj_000005_60sol.html"
)
```

UI sliders:

- disturbance speeds (`depth/width/height`)
- friction (`mu`)
- grasp force
- perturbation / hold timing
- viewer options

