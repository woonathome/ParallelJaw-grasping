# Grasping_Face

`Grasping_Face` computes feasible grasp pairs for a parallel-jaw gripper (AG145-style parameters) from a CAD mesh (`.obj` / `.ply`) using geometric analysis.

Main implementation:
- [grasping.py](./Grasping_Face/grasping.py)
- [visualize_grasping.py](./Grasping_Face/visualize_grasping.py)

## 1. What This Module Does

Pipeline overview:

1. Load and simplify the input mesh.
2. Extract planar patches by grouping adjacent faces.
3. Build and score opposite-facing patch-pair candidates.
4. Run pad-volume collision checks to keep physically feasible face pairs.
5. Optionally compute gripper pose and EE delta pose for robot execution.

## 2. Dependencies

```bash
pip install numpy scipy trimesh open3d plotly
```

`check_gripper_feasibility_*` uses `trimesh.collision.CollisionManager`; depending on your environment, an additional collision backend may be required.

## 3. Quick Start

```python
import Grasping_Face.grasping as gf

mesh_path = "./models_target/models_cad/custom_ind/bracket_1.obj"  # .ply is also supported

result = gf.compute_best_patch_pairs(
    mesh_path=mesh_path,
    mesh_max_triangles=1500,
    angle_deg=10,
    min_opening=1.0,
    max_opening=140.0,
    angle_tolerance_deg=10,
    top_k=500,
)

# Option A: Yaw-aware OBB pad collision check (default yaw grid: [0, 90])
reports = gf.check_gripper_feasibility_faces_with_yaw(result)

# Option B: Cylindrical swept-volume collision check (coarser but simple)
# reports = gf.check_gripper_feasibility_faces_with_rotation(result)

print(result["num_patches"], result["num_candidates"], len(reports))
```

## 4. Core Data Structures

### `PlanarPatch`

- `id`: Patch index
- `face_indices`: Face indices belonging to the patch
- `normal`, `b`: Plane equation `n·x = b`
- `area`: Patch area
- `centroid`: Patch bbox center
- `centroid_check`: Mean vertex center (used for checks)

### `PatchPairCandidate`

- `patch_i`, `patch_j`: Patch IDs
- `normal`: Normal of `patch_i`
- `width`: Distance between patch centroids (opening candidate)
- `score`: Final weighted score
- `terms`: `distance`, `inertia`, `area`, `overlap`

### `PadParams`

Default values(Robotiq AG145-105 Gripper):
- `pad_w = 34.0`
- `pad_h = 21.0`
- `pad_d = 7.0`
- `mu = 0.8`

Units are assumed to be **millimeters**. Input mesh scale should match this assumption.

## 5. Main Function Details

### `compute_best_patch_pairs(...)`

Main entry point of the grasp candidate pipeline.

Inputs:
- `mesh_path`
- mesh simplification target (`mesh_max_triangles`)
- patch merge angle (`angle_deg`)
- opening bounds (`min_opening`, `max_opening`)
- opposite-normal tolerance (`angle_tolerance_deg`)
- number of returned candidates (`top_k`)

Processing steps:
1. `load_uniform_mesh_with_open3d`: mesh cleanup + quadric decimation.
2. `split_long_edges`: subdivide long triangles.
3. `extract_planar_patches`: region-grow planar patches.
4. `orient_patch_normals`: flip normals to outward directions.
5. Pair filtering:
   - Nearly opposite normals
   - Facing each other
   - Opening range check
   - Positive overlap and score
6. Candidate scoring via `score_patch_pair`.

Returns a `result` dict containing meshes, patches, and ranked candidates.

### `extract_planar_patches(mesh, angle_deg=15.0)`

Extracts planar regions from adjacent triangle faces.

- Uses face adjacency graph and BFS-like region growing.
- Neighbor faces are merged when normal-angle condition passes.
- Re-fits patch plane by SVD (`plane_from_points`).
- Outputs `PlanarPatch` list with area/plane/centroid metadata.

### `orient_patch_normals(mesh_quad, mesh_patches, patches)`

Makes patch normals consistently outward.

- Tests points shifted along `+n` and `-n` using `mesh_quad.contains`.
- If the current direction points inward, flips the normal.
- Stabilizes downstream "opposite-facing pair" checks.

### `score_patch_pair(...)`

Computes geometric quality terms for a patch pair:

- `overlap`: bidirectional projection overlap ratio between patches
- `distance`: alignment quality between patch normals and opposite centroids
- `inertia`: distance from centroid-connection line to mesh COM
- `area`: area balance ratio between two patches

Current final score uses weighted sum of:
- `distance`, `inertia`, `area`

`overlap` is still computed and used as a filtering condition.

### `check_gripper_feasibility_faces_with_yaw(result, ...)`

Face-level feasibility check with yaw-aware pad OBB collision testing.

- For each patch pair candidate, finds best aligned face pair.
- Rejects badly misaligned face-to-face geometry.
- Builds pad OBBs with `make_rot_pad_box_at_patch`.
- Tests collision against check mesh.
- Keeps feasible pairs and records:
  - `dist`: midpoint-to-COM distance
  - `moment`: simple torque metric around COM
  - `feasible_yaw`

### `check_gripper_feasibility_faces_with_rotation(result, ...)`

Alternative feasibility check using cylindrical swept-volume approximation.

- Uses `make_pad_cylinder_at_patch`.
- Similar filtering flow with simpler shape model.
- Typically faster/coarser than yaw-aware OBB checks.

### `build_gripper_pose_obj(...)` / `build_gripper_pose_obj_OPE(...)`

Build object-frame gripper pose `H_OG` from two contact patches.

- Defines closing axis from patch normal difference.
- Applies yaw around closing frame.
- Applies stroke-dependent origin offset via `origin_offset_from_stroke`.
- `build_gripper_pose_obj_OPE` additionally uses `H_OC` to enforce camera-frame directional constraints.

### `ee_delta_pose_des(H_OC, H_OG)`

Computes desired EE relative motion transform for robot execution.

Outputs:
- `H_EdEn`: desired EE pose relative to current EE
- `H_OdEn`

Uses fixed internal calibration transforms (`H_GnEn`, `H_GnCn`).

### `calculate_epsilon_quality(...)` / `calculate_squeeze_epsilon_quality(...)`

Wrench-space force-closure quality utilities:

- `calculate_epsilon_quality`: point-contact-based epsilon quality
- `calculate_squeeze_epsilon_quality`: squeezed pad contact-region-based epsilon quality

These are primarily analysis/experimental metrics and are not the main ranking criterion in the default candidate pipeline.

## 6. Output Schemas

### `result = compute_best_patch_pairs(...)`

Important keys:
- `mesh_quad`
- `mesh_patches`
- `num_patches`
- `num_candidates`
- `params`
- `patches`
- `best`
- `top_k`

### `reports = check_gripper_feasibility_* (result)`

Each feasible report may include:
- `pair_index`, `patch_i`, `patch_j`
- `face_i`, `face_j`
- `feasible`
- `feasible_yaw` (`with_yaw` version)
- `dist`, `moment`

## 7. Visualization Helpers

From `visualize_grasping.py`:

- `visualize_merged_patches_plotly(result)`
- `visualize_pairs_centroid_lines(result)`
- `visualize_feasible_pairs_*`
- `visualize_frames(...)`

These are useful for inspecting extracted patches, candidate pairs, feasible contacts, and frame transforms.
