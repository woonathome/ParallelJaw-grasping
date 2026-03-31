# Grasping_Face

`Grasping_Face` computes grasp candidates for a parallel-jaw gripper (AG145-style pad params) from CAD meshes (`.obj` / `.ply`) using geometric analysis.

Main files:
- [grasping.py](./Grasping_Face/grasping.py)
- [visualize_grasping.py](./Grasping_Face/visualize_grasping.py)

## 1. Pipeline (Current)

The core flow is:

1. Mesh preprocessing and simplification.
2. Patch extraction (planar region growing).
3. Patch pair filtering/scoring/ranking.
4. Gripper pad collision check on face-level contact candidates.

This order is preserved in the current optimized implementation.

## 2. Dependencies

```bash
pip install numpy scipy trimesh open3d plotly
```

`check_gripper_feasibility_*` uses `trimesh.collision.CollisionManager`.  
Depending on your system, an extra collision backend (for `python-fcl`) may be required.

## 3. Quick Start

```python
import Grasping_Face.grasping as gf

mesh_path = "./models_target/models_cad/custom_ind/bracket_1.obj"

result = gf.compute_best_patch_pairs(
    mesh_path=mesh_path,
    mesh_max_triangles=1500,
    angle_deg=10.0,
    min_opening=1.0,
    max_opening=140.0,
    angle_tolerance_deg=10.0,
    top_k=500,
    pair_sort_key="score",      # "score" or "epsilon"
    max_faces_after_split=25000,
)

reports = gf.check_gripper_feasibility_faces_with_yaw(
    result,
    max_face_trials_per_pair=40,
    max_feasible_per_pair=4,
    sort_key="dist_moment",     # "dist_moment" or "epsilon"
)

print(result["num_patches"], result["num_candidates"], len(reports))
```

## 4. Data Structures

### `PlanarPatch`

- `id`: patch index
- `face_indices`: face indices in this patch
- `normal`, `b`: plane equation `n dot x = b`
- `area`: patch area
- `centroid`: patch AABB center
- `centroid_check`: mean vertex center

### `PatchPairCandidate`

- `patch_i`, `patch_j`: patch IDs
- `normal`: normal of `patch_i`
- `width`: centroid distance (opening candidate)
- `score`: weighted geometric score
- `terms`: `distance`, `inertia`, `area`, `overlap` (+ optional `epsilon`)

### `PadParams` (default)

- `pad_w = 34.0`
- `pad_h = 21.0`
- `pad_d = 7.0`
- `mu = 0.8`

Units are assumed to be millimeters.

## 5. Main Functions (Updated)

### `split_long_edges(mesh, max_iter=100, max_faces_after_split=20000, max_len=None)`

- Splits long edges iteratively.
- Includes a face-count guard (`max_faces_after_split`) to prevent mesh explosion.
- If this cap is too low, planar objects may produce fewer patch/face combinations.

### `compute_best_patch_pairs(...)`

Current arguments include:

- geometry: `mesh_path`, `mesh_max_triangles`, `angle_deg`, `min_opening`, `max_opening`, `angle_tolerance_deg`, `top_k`
- speed/coverage controls:
  - `prefilter_multiplier=8`
  - `min_prefilter_pool=256`
  - `min_area_ratio_prefilter=0.03`
  - `max_faces_after_split=20000`
- ranking:
  - `pair_sort_key="score"` or `"epsilon"`
  - `epsilon_mu`, `epsilon_k` (used when `pair_sort_key="epsilon"`)

Processing:

1. `load_uniform_mesh_with_open3d` (cleanup + decimation).
2. `split_long_edges` (with face cap).
3. `extract_planar_patches` and `orient_patch_normals`.
4. Vectorized pair prefilter (`opposite`, `facing`, opening, area-ratio, quick score).
5. Prefilter pool truncation (`prefilter_multiplier`, `min_prefilter_pool`).
6. Detailed scoring via `score_patch_pair` + overlap check.
7. Final sort:
   - `score`: descending
   - `epsilon`: descending epsilon, then score

### `score_patch_pair(...)`

Computes:

- `distance`: centroid-line alignment term
- `inertia`: COM-line distance term
- `area`: patch-area balance
- `overlap`: bidirectional projection overlap

Final `score` uses the weighted sum of `distance + inertia + area`.  
`overlap` is still computed and used as a validity filter (`overlap > 0`).

### `check_gripper_feasibility_faces_with_yaw(...)`

Face-level feasibility with yaw-aware pad OBB collision:

- Finds face matches via `_best_face_matches` (best face-j per face-i, then top-N).
- `max_face_trials_per_pair` limits tested face pairs (default `40`).
- `max_feasible_per_pair` limits accepted reports per patch pair (default `4`).
- Uses optional broad-phase reject before exact collision (`use_broadphase=True`).
- Sort options:
  - `sort_key="dist_moment"`: ascending `(dist, moment)` (smaller is better)
  - `sort_key="epsilon"`: descending `epsilon` (larger is better), then `(dist, moment)`

### `check_gripper_feasibility_faces_with_rotation(...)`

Alternative feasibility path with cylindrical pads:

- Same face-pair preselection and limits as yaw version.
- Uses `make_pad_cylinder_at_patch`.
- Supports the same sort behavior (`dist_moment` / `epsilon`).

### Pose / Robot Utilities

- `build_gripper_pose_obj(...)`
- `build_gripper_pose_obj_OPE(...)`
- `ee_delta_pose_des(...)`

These convert feasible contact pairs into object-frame gripper pose and EE delta transforms.

### Quality Metrics

- `calculate_epsilon_quality(...)`
- `calculate_squeeze_epsilon_quality(...)`

`epsilon` is now available as a sorting option in patch-pair ranking and feasibility report ranking.

## 6. Output Schema

### `result = compute_best_patch_pairs(...)`

Key fields:

- `mesh_quad`, `mesh_patches`
- `num_patches`, `num_candidates`
- `params`, `patches`
- `best`, `top_k`

### `reports = check_gripper_feasibility_* (...)`

Feasible item fields:

- common: `pair_index`, `patch_i`, `patch_j`, `face_i`, `face_j`, `dist`, `moment`, `feasible`
- yaw version adds: `feasible_yaw`
- epsilon sort mode adds: `epsilon`

## 7. Visualization (`visualize_grasping.py`)

Main entry points:

- `visualize_merged_patches_plotly(result)`
- `visualize_pairs_centroid_lines(result, max_pairs_show=200, ...)`
- `visualize_feasible_pairs_pads(..., max_reports_show=300, ...)`
- `visualize_feasible_pairs_with_yaw(..., max_reports_show=300, ...)`
- `visualize_feasible_pairs_with_cylinder(..., max_reports_show=300, ...)`
- `visualize_feasible_pairs(..., max_reports_show=300, ...)`
- `visualize_feasible_pairs_pads_gripper(..., max_reports_show=300, ...)`

Visualization functions may display only a prefix of candidates/reports by default.

## 8. Coverage vs Speed Tuning

If planar-heavy objects seem to lose too many pairs, increase coverage by tuning:

- `max_faces_after_split` (more remesh subdivision)
- `min_area_ratio_prefilter` (smaller to keep more asymmetric pairs)
- `prefilter_multiplier` / `min_prefilter_pool` (larger candidate pool)
- `max_face_trials_per_pair` (more face pairs tested per patch pair)
- `max_feasible_per_pair` (more feasible contacts kept per patch pair)
- visualization caps: `max_pairs_show`, `max_reports_show`
