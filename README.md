# Grasping_Face

`Grasping_Face` computes parallel-jaw grasp candidates from CAD meshes (`.obj`, `.ply`) using geometric analysis.

Main code:
- `Grasping_Face/grasping.py`
- `Grasping_Face/visualize_grasping.py`

## 1. Pipeline

Current pipeline order:

1. Mesh preprocessing/simplification
2. Planar patch extraction
3. Patch-pair generation/ranking
4. Face-level pad collision feasibility

## 2. Dependencies

```bash
pip install numpy scipy trimesh open3d plotly
```

`check_gripper_feasibility_*` uses `trimesh.collision.CollisionManager`.  
Depending on environment, a collision backend (for `python-fcl`) may be required.

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
    pair_sort_key="score",   # "score" or "epsilon"
    max_faces_after_split=25000,
)

reports = gf.check_gripper_feasibility_faces_with_yaw(
    result,
    max_face_trials_per_pair=100,
    max_feasible_per_pair=4,
    sort_key="epsilon",      # "dist_moment" or "epsilon"
)

print(result["num_patches"], result["num_candidates"], len(reports))
```

## 4. Core Structures

### `PlanarPatch`

- `id`
- `face_indices`
- `normal`, `b` (plane `n dot x = b`)
- `area`
- `centroid`
- `centroid_check`

### `PatchPairCandidate`

- `patch_i`, `patch_j`
- `normal`
- `width`
- `score`
- `terms`: `distance`, `inertia`, `area`, `overlap` (+ optional `epsilon`)

### `PadParams` (default)

- `pad_w = 34.0`
- `pad_h = 21.0`
- `pad_d = 7.0`
- `mu = 0.8`

Units are assumed to be millimeters.

## 5. Main Functions (Current Behavior)

### `compute_best_patch_pairs(...)`

Main arguments:
- geometry: `mesh_path`, `mesh_max_triangles`, `angle_deg`, `min_opening`, `max_opening`, `angle_tolerance_deg`, `top_k`
- speed/coverage: `prefilter_multiplier`, `min_prefilter_pool`, `min_area_ratio_prefilter`, `max_faces_after_split`
- ranking: `pair_sort_key`, `epsilon_mu`, `epsilon_k`

Notes:
- Uses vectorized prefiltering for opposite/facing/opening/area checks.
- Uses overlap cache for faster overlap evaluation.
- If `pair_sort_key="epsilon"`, epsilon uses `calculate_epsilon_quality` (point-contact).

### `check_gripper_feasibility_faces_with_yaw(...)`

Yaw-aware OBB pad feasibility check.

Important point:
- For `sort_key="epsilon"`, epsilon is computed with `calculate_squeeze_epsilon_quality` (surface-contact model).
- The old point-contact epsilon call is kept as a commented line in code.

### `check_gripper_feasibility_faces_with_rotation(...)`

Cylindrical pad feasibility check.

Important point:
- For `sort_key="epsilon"`, this path currently uses `calculate_epsilon_quality` (point-contact model).

### `calculate_epsilon_quality(...)`

Point-contact epsilon metric from two contact points and friction cones.

### `calculate_squeeze_epsilon_quality(...)`

Surface-contact epsilon metric from pad-squeezed contact regions.

Current implementation details:
- Inputs `f_i_idx`, `f_j_idx` are treated as patch face-index sets.
- Patch-level submeshes are built via `mesh.submesh([...], append=True)`.
- Contact candidates are sampled by `get_points_in_squeezed_pad(...)`.
- Wrenches are built with `add_wrenches(...)`.

### `get_points_in_squeezed_pad(...)` and `_build_patch_sample_points(...)`

- `_build_patch_sample_points` builds deterministic sample points on patch submesh:
  - vertices
  - face centroids
  - edge midpoints
- `get_points_in_squeezed_pad` filters those samples by pad OBB containment.

This was added to better cover patch contact regions compared to vertices-only sampling.

## 6. Sorting Behavior

Current code behavior:

- `compute_best_patch_pairs(pair_sort_key="epsilon")`
  - sorts by epsilon/score with `reverse=True` (larger first)
- `check_gripper_feasibility_faces_* (sort_key="dist_moment")`
  - sorts ascending by `(dist, moment)` (smaller first)
- `check_gripper_feasibility_faces_* (sort_key="epsilon")`
  - currently sorts ascending by `(epsilon, dist, moment)` in code

If you want larger epsilon first in feasibility reports, adjust sort order in those functions.

## 7. Visualization

Main functions in `visualize_grasping.py`:

- Patch/pair/feasible visualization:
  - `visualize_merged_patches_plotly`
  - `visualize_pairs_centroid_lines`
  - `visualize_feasible_pairs_pads`
  - `visualize_feasible_pairs_with_yaw`
  - `visualize_feasible_pairs_with_cylinder`
  - `visualize_feasible_pairs`
  - `visualize_feasible_pairs_pads_gripper`
- Wrench-space visualization:
  - `visualize_wrench_space` (legacy point-contact)
  - `visualize_squeeze_wrench_space` (surface-contact)

For current surface-contact epsilon analysis, use `visualize_squeeze_wrench_space`.

## 8. Notebook Notes

`GraspingTest.ipynb` includes:
- single-object pipeline run
- GWS/eQuality cell for surface-contact model (`calculate_squeeze_epsilon_quality` + `visualize_squeeze_wrench_space`)
- batch CAD evaluation/export workflow

## 9. Coverage vs Speed Tuning

When candidate coverage is too small on planar-heavy objects, tune:

- `max_faces_after_split`
- `min_area_ratio_prefilter`
- `prefilter_multiplier`
- `min_prefilter_pool`
- `max_face_trials_per_pair`
- `max_feasible_per_pair`
- visualization caps (`max_pairs_show`, `max_reports_show`)
