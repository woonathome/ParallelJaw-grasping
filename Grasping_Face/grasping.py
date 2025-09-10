# Compute Patch Pairs, Feasible Grasping Pairs

import json, math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import numpy as np
import trimesh
from trimesh.collision import CollisionManager
import open3d as o3d

@dataclass
class PlanarPatch:
    id: int
    face_indices: List[int]
    normal: List[float]
    b: float
    area: float
    centroid: List[float]


@dataclass
class PatchPairParams:
    min_opening: float = 5.0
    max_opening: float = 140.0
    angle_tolerance_deg: float = 15.0
    # parallel_weight: float = 0.2 # TODO: 추후 다른 기준 추가 시 조정
    # overlap_weight:   float = 0.3 # TODO: 추후 다른 기준 추가 시 조정
    distance_weight:  float = 1.0 # TODO: 추후 다른 기준 추가 시 조정
    samples_per_face= 3          # face 샘플링 개수


@dataclass
class PatchPairCandidate:
    patch_i: int
    patch_j: int
    normal: List[float]
    width: float
    score: float
    terms: Dict[str, float]


@dataclass
class PadParams:
    pad_w: float = 34.0 # 34
    pad_h: float = 21.0 # 21
    pad_d: float = 7.0 # 7


# --------------------------
# Mesh helpers
# --------------------------
def load_uniform_mesh_with_open3d(path, target_triangles=500, min_area=5): # TODO: min_area 기준 업데이트 필요
    # Trimesh로 읽고 Open3D Mesh로 변환
    mesh_true = trimesh.load(path, force='mesh')
    if isinstance(mesh_true, trimesh.Scene):
        mesh_true = mesh_true.dump().sum()

    o3 = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh_true.vertices),
        o3d.utility.Vector3iVector(mesh_true.faces)
    )
    o3.remove_degenerate_triangles()
    o3.remove_duplicated_vertices()
    o3.remove_non_manifold_edges()

    # Quadric Decimation
    o3_simpl = o3.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    v = np.asarray(o3_simpl.vertices)
    f = np.asarray(o3_simpl.triangles) 
    mesh_quad = trimesh.Trimesh(vertices=v, faces=f, process=True)
    # print(mesh_quad.is_watertight)

    # 3) 삼각형 면적 직접 계산 → 작은 face 제거
    V = np.asarray(o3_simpl.vertices)
    F = np.asarray(o3_simpl.triangles, dtype=np.int64)
    v0, v1, v2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    mask = areas >= float(min_area)
    F = F[mask]

    # 4) Trimesh로 반환
    remeshed = trimesh.Trimesh(vertices=V, faces=F, process=True)
    return remeshed, mesh_quad


# --------------------------
# Geometry helpers
# --------------------------
def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def plane_from_points(points: np.ndarray) -> Tuple[np.ndarray, float]:
    centroid = (points.max(axis=0) + points.min(axis=0)) / 2
    X = points - centroid
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1]
    n = unit(n)
    b = float(n @ centroid)
    return n, b

def point_line_distance(c_a:np.ndarray, n_a:np.ndarray, c_b:np.ndarray):
    """
    c_a 시작점, n_a 방향, c_b 거리측정 목표지점
    """
    n_u = unit(n_a)
    r = c_b - c_a
    t = float(np.dot(r, n_u)) # 직선 파라미터
    dist = float(np.linalg.norm(r - t * n_u))
    return dist

def basis_from_normal(n: np.ndarray):
    n = unit(n)
    a = np.array([1.0,0,0]) if abs(n[0]) < 0.9 else np.array([0,1.0,0])
    u = unit(np.cross(n, a))
    v = np.cross(n, u)
    return u, v, n

def point_in_tri_2d(p, a, b, c): # 삼각형 내부 검사
    # barycentric in 2D
    v0 = c - a; v1 = b - a; v2 = p - a
    d00 = np.dot(v0, v0); d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1); d20 = np.dot(v2, v0); d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:  # degenerate
        return False
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (u >= 0) and (v >= 0) and (w >= 0)

def rotate_byaxis(axis, ang_deg):
    axis = unit(axis)
    th = np.deg2rad(ang_deg)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], float)
    I = np.eye(3)
    return I + np.sin(th)*K + (1-np.cos(th))*(K@K)

def sample_points_on_patch(mesh: trimesh.Trimesh, face_idx, samples_per_face):
    faces = mesh.faces[np.asarray(list(face_idx), dtype=int)]
    V = mesh.vertices
    P = []
    for f in faces:
        a,b,c = V[f[0]], V[f[1]], V[f[2]]
        for _ in range(samples_per_face):
            r1 = np.random.rand(); r2 = np.random.rand()
            # uniform on triangle
            if r1 + r2 > 1.0:
                r1, r2 = 1.0 - r1, 1.0 - r2
            p = a + r1*(b-a) + r2*(c-a)
            P.append(p)
    return np.asarray(P, float)

def projection_overlap_ratio(mesh: trimesh.Trimesh,
                              patch_i: dict, patch_j: dict,
                              samples_per_face) -> float:

    n_i = unit(np.asarray(patch_i["normal"], float))
    n_j = unit(np.asarray(patch_j["normal"], float))
    b_j = float(patch_j["b"])

    # i에서 샘플 추출
    pts = sample_points_on_patch(mesh, patch_i["face_indices"], samples_per_face)
    if len(pts) == 0:
        return 0.0

    # 투영: p' = p + t*(-n_i),  where  n_j·(p + t*(-n_i)) = b_j
    denom = -(n_j @ n_i)  # n_j·(-n_i)
    # if abs(denom) < 1e-9:
    #     return 0.0  # 직교투영 불가능(평행)

    t = (b_j - pts @ n_j) / denom
    proj = pts + (t.reshape(-1,1) * (-n_i))

    # patch_j 내부판정: UV 평면으로 투영 후, 각 face 삼각형에 대해 point-in-triangle
    u, v, _ = basis_from_normal(n_j)
    # 기준점은 patch_j의 임의 한 점(centroid)
    c0 = np.asarray(patch_j["centroid"], float)
    # 2D 좌표
    rel = proj - c0
    P2 = np.column_stack([rel @ u, rel @ v])

    V = mesh.vertices
    faces_j = mesh.faces[np.asarray(list(patch_j["face_indices"]), dtype=int)]
    hit = np.zeros(len(P2), dtype=bool)

    # 각 얼굴(삼각형)도 2D로
    for f in faces_j:
        A = V[f[0]] - c0; B = V[f[1]] - c0; C = V[f[2]] - c0
        A2 = np.array([A @ u, A @ v]); B2 = np.array([B @ u, B @ v]); C2 = np.array([C @ u, C @ v])
        # 아직 미히트인 점만 검사
        idx = np.where(~hit)[0]
        if len(idx) == 0: break
        for k in idx:
            if point_in_tri_2d(P2[k], A2, B2, C2):
                hit[k] = True

    valid = len(P2)
    inside = int(hit.sum())
    return (inside / valid) if valid > 0 else 0.0

def make_pad_box_at_patch(patch, pad_w, pad_h, pad_d, clearance_out,
                          lift_mm: float = 2.0):
    """
    패치 평면의 바깥쪽(+n)으로 'pad_d + clearance_out' 만큼 돌출된 OBB 생성
    + centroid에서 바깥쪽으로 lift_mm 만큼 추가 이격하여 검사 (normal이 정확히 표면에 있지 않음)
    - extents: [pad_w, pad_h, pad_d + clearance_out]
    - center : 패치 중심 + n * (extents_z/2)
    - orientation: columns = [u, v, n]
    """
    n = unit(np.asarray(patch["normal"], float))
    u, v, n = basis_from_normal(n)
    c = np.asarray(patch["centroid"], float)

    ext_z = float(pad_d + clearance_out)
    box = trimesh.creation.box(extents=[float(pad_w) / 2, float(pad_h) / 2, ext_z])

    T = np.eye(4)
    T[:3, :3] = np.column_stack([u, v, n])
    T[:3,  3] = c + n * (lift_mm + 0.5*ext_z)
    # T[:3,  3] = c + n*0.5*ext_z
    # + n * lift_mm
    box.apply_transform(T)
    return box

def make_rot_pad_box_at_patch(patch, pad_w, pad_h, pad_d, clearance_out,
                              lift_mm: float = 2.0, yaw_deg: float = 0.0):
    """
    패치 평면의 바깥쪽(+n)으로 'pad_d + clearance_out' 만큼 돌출된 OBB 생성
    - extents = [pad_w, pad_h, pad_d + clearance_out]
    - 로컬 축: [u, v, n] (n=바깥쪽). yaw는 로컬 n축 기준 회전만 적용
    - center = c + n_world * (lift_mm + 0.5*ext_z)
    """
    n = unit(np.asarray(patch["normal"], float))
    u, v, n = basis_from_normal(n)  # right-handed
    c = np.asarray(patch["centroid"], float)

    ext_z = float(pad_d + clearance_out)
    box = trimesh.creation.box(extents=[float(pad_w) / 2, float(pad_h) / 2, ext_z])

    # 로컬 yaw 회전: R_world = B @ Rz(yaw)
    B = np.column_stack([u, v, n])
    cy, sy = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]], float)
    R_world = B @ Rz
    n_world = R_world[:, 2]
    T = np.eye(4)
    T[:3, :3] = R_world
    T[:3,  3] = c + n_world * (lift_mm + 0.5*ext_z)
    box.apply_transform(T)
    return box

# --------------------------
# Planar patch extraction
# --------------------------
def extract_planar_patches(
    mesh: trimesh.Trimesh,
    angle_deg: float = 15.0,
    coplanar_tol: float = 1e-3,
    min_patch_area: float = 20.0, # TODO: min_patch_area 기준 업데이트 필요
) -> List[PlanarPatch]:
    face_normals = mesh.face_normals
    faces = mesh.faces
    face_adjacency = mesh.face_adjacency
    F = len(faces)

    neighbors = [[] for _ in range(F)]
    for f0, f1 in face_adjacency:
        neighbors[f0].append(f1)
        neighbors[f1].append(f0)

    used = np.zeros(F, dtype=bool)
    patches: List[PlanarPatch] = []

    cos_th = math.cos(math.radians(angle_deg))

    tri_verts = mesh.vertices[faces]
    tri_areas = trimesh.triangles.area(tri_verts)
    # tri_centroids = (tri_verts.max(axis=1) + tri_verts.min(axis=1)) / 2

    pid = 0
    for f in range(F):
        if used[f]:
            continue

        region = [f]
        used[f] = True
        seed_points = tri_verts[f].reshape(-1, 3)
        n_seed, b_seed = plane_from_points(seed_points)

        queue = [f]
        while queue:
            cur = queue.pop()
            for nb in neighbors[cur]:
                if used[nb]:
                    continue
                n_nb = face_normals[nb]
                if abs(float(n_seed @ n_nb)) < cos_th:
                    continue
                verts_nb = tri_verts[nb].reshape(-1, 3)
                d = np.abs((verts_nb @ n_seed) - b_seed)
                if np.max(d) > coplanar_tol:
                    continue
                used[nb] = True
                region.append(nb)
                queue.append(nb)

        verts_region = tri_verts[region].reshape(-1, 3)
        n_reg, b_reg = plane_from_points(verts_region)

        area_reg = float(tri_areas[region].sum())
        if area_reg < min_patch_area:     # 면적이 너무 작은 patch 필터링
            continue
    
        centroid_reg = (verts_region.max(axis=0) + verts_region.min(axis=0)) / 2
        # centroid_reg = verts_region.mean(axis=0)

        patches.append(
            PlanarPatch(
                id=pid,
                face_indices=region,
                normal=n_reg.tolist(),
                b=float(b_reg),
                area=area_reg,
                centroid=centroid_reg.tolist(),
            )
        )
        pid += 1

    return patches

def orient_patch_normals(mesh_quad: trimesh.Trimesh, patches, inward=False):
    """
    패치 법선을 mesh 기준으로 일관되게 정렬.
    mesh.contains로 실제 안/밖을 체크
    inward=False -> 모두 '바깥쪽'으로
    """
    # 오프셋 길이: 모델 크기 대비 아주 작게
    bbox_diag = float(np.linalg.norm(mesh_quad.bounds[1] - mesh_quad.bounds[0]))
    eps = 1e-2 * bbox_diag
    # eps = 1

    for p in patches:
        n = np.asarray(p.normal if hasattr(p, "normal") else p["normal"], float)
        n = unit(n)
        c = np.asarray(p.centroid if hasattr(p, "centroid") else p["centroid"], float)

        flip = False

        outside = not mesh_quad.contains([c + eps * n])[0]
        if inward:
            flip = outside   # 밖이라면 뒤집어 안쪽 향하게
        else:
            flip = not outside  # 안이면 뒤집어 바깥 향하게

        if flip:
            n = -n
        # 반영
        if hasattr(p, "normal"):
            p.normal = n.tolist()
        else:
            p["normal"] = n.tolist()

    return patches


# --------------------------
# Patch Pair scoring
# --------------------------
def score_patch_pair(patch_a: PlanarPatch,
                     patch_b: PlanarPatch,
                     params: PatchPairParams,
                     mesh_for_overlap: trimesh.Trimesh) -> PatchPairCandidate:
    n_a = unit(np.array(patch_a.normal))
    n_b = unit(np.array(patch_b.normal))

    # # 평행/반대 점수: normal 각도
    # dot_ab = float(n_a @ n_b)
    # s_parallel = max(0.0, -dot_ab)
    # ang = math.degrees(math.acos(max(-1.0, min(1.0, dot_ab))))
    # if ang < (180.0 - params.angle_tolerance_deg):
    #     s_parallel *= (ang / (180.0 - params.angle_tolerance_deg))
    # if dot_ab > 0.0:
    #     s_parallel = 0.0

    # 투영-포함 점수: i-j 상호 투영 비율의 곱 (패치 2개의 평행성 검사 1)
    overlap_a = projection_overlap_ratio(mesh_for_overlap,
                                           asdict(patch_a) if hasattr(patch_a, "id") else patch_a,
                                           asdict(patch_b) if hasattr(patch_b, "id") else patch_b,
                                           samples_per_face=params.samples_per_face)
    overlap_b = projection_overlap_ratio(mesh_for_overlap,
                                           asdict(patch_b) if hasattr(patch_b, "id") else patch_b,
                                           asdict(patch_a) if hasattr(patch_a, "id") else patch_a,
                                           samples_per_face=params.samples_per_face)
    s_overlap = overlap_a * overlap_b

    # TODO: width patch 간 최소 거리로 업뎃 필요 (현재: centroid 간 거리)
    c_a = np.asarray(patch_a.centroid)
    c_b = np.asarray(patch_b.centroid)
    width = np.linalg.norm(c_a - c_b)

    # TODO: c-n 최소 거리 점수(모멘트): c-n 직선과 c간의 거리 점수(패치 2개의 평행성 검사 2)
    bbox_diag = float(np.linalg.norm(mesh_for_overlap.bounds[1] - mesh_for_overlap.bounds[0]))
    dist_a = point_line_distance(c_a, n_a, c_b)
    dist_b = point_line_distance(c_b, n_b, c_a)
    s_distance = 1 - (dist_a + dist_b) / bbox_diag

    # 최종 점수(가중합)
    score = params.distance_weight * s_distance

    return PatchPairCandidate(
        patch_i=patch_a.id,
        patch_j=patch_b.id,
        normal=n_a.tolist(),
        width=width,
        score=score,
        terms={
            "overlap": s_overlap,
            "distance": s_distance
               },
    )


# --------------------------
# Main API
# --------------------------

def compute_best_patch_pairs(
    mesh_path: str,
    mesh_max_triangles: int = 500,         # 원본 mesh 삼각형 개수
    angle_deg: float = 7.0,                # 패치 병합 허용 각도 (↑면 패치 수 ↓)
    coplanar_tol: float = 1e-3,            # 공면성 허용 오차 (↑면 패치 수 ↓)
    min_opening: float = 10.0,              # 그리퍼 최소 개구(mm)
    max_opening: float = 140.0,              # 그리퍼 최대 개구(mm)
    angle_tolerance_deg: float = 10.0,     # 패치 페어 정반대 허용 각도
    top_k: int = 1,                        # 상위 후보 수
) -> Dict[str, Any]:
    
    remesh, mesh_quad = load_uniform_mesh_with_open3d(mesh_path, target_triangles=mesh_max_triangles)
    # remesh(면적 기준 필터링 이후, watertight X), mesh_quad: (면적 기준 필터링 이전, watertight O)

    patches = extract_planar_patches(remesh, angle_deg=angle_deg, coplanar_tol=coplanar_tol)
    patches = orient_patch_normals(mesh_quad, patches, inward=False)  # false: 모두 바깥쪽으로 정렬

    params = PatchPairParams(
        min_opening=min_opening,
        max_opening=max_opening,
        angle_tolerance_deg=angle_tolerance_deg,
    )

    cands: List[PatchPairCandidate] = []
    nvecs = [unit(np.array(p.normal)) for p in patches] # unit normals 
    # 현재 패치와 normals가 angle_tolerance_deg 이하인 patch만 남김 (각도 180도 +- angle_tolerance_deg 범위)
    for i in range(len(patches) - 1):
        ni = nvecs[i]

        best_cand = None
        best_score = -1.0

        for j in range(i + 1, len(patches)):
            # i-j 법선 각도
            dot = float(ni @ nvecs[j])
            ang = math.degrees(math.acos(max(-1.0, min(1.0, dot))))

            # 180° - tol 보다 작으면 충분히 반대가 아님 -> 스킵
            if ang < 180.0 - params.angle_tolerance_deg:
                continue

            cand = score_patch_pair(patches[i], patches[j], params, remesh)
            
            # 기존 폭/면적/점수 필터
            if cand.width < params.min_opening or cand.width > params.max_opening:
                continue
            if cand.terms['overlap'] <= 0: # 상호 겹치는 부분이 없는 pair는 넘김
                continue
            if cand.score <= 0.0:
                continue
            # print(i,j)


            # i에 대해 최고 점수만 유지
            if cand.score > best_score:
                best_score = cand.score
                best_cand = cand

        if best_cand is not None:
            cands.append(best_cand)

    cands.sort(key=lambda c: c.score, reverse=True)

    return {
        "mesh_quad": mesh_quad,
        "mesh_patches": remesh,
        "num_patches": len(patches),
        "num_candidates": len(cands),
        "params": asdict(params),
        "patches": [asdict(p) for p in patches],
        "best": asdict(cands[0]) if cands else None,
        "top_k": [asdict(c) for c in cands[:top_k]] if cands else [],
        # "top_k": [asdict(c) for c in cands[:]] if cands else [], # 모든 pair cands
    }


def check_gripper_feasibility(
    result: dict,
    pad_w: float = PadParams.pad_w,     # 폭 (u 방향)
    pad_h: float = PadParams.pad_h,     # 높이 (v 방향)
    pad_d: float = PadParams.pad_d,      # 두께 (n 방향)
    clearance_out: float = 20.0,   # 패드 바깥쪽 여유(접근 거리)
    use_mesh: str = "mesh_quad"    # 충돌검사에 사용할 메쉬 키: "mesh_quad" 권장
):
    """
    result['top_k']의 각 pair에 대해 pad approach OBB와 메쉬 충돌 검사.
    충돌 없음: feasible=True
    반환: [{pair_index, patch_i, patch_j, feasible}] 리스트
    """
    mesh = result[use_mesh]
    patches = {p["id"]: p for p in result["patches"]}
    cands = result.get("top_k", [])

    # COM (fallback: bbox center)
    try:
        com = mesh.center_mass
    except Exception:
        bmin, bmax = mesh.bounds
        com = 0.5 * (bmin + bmax)

    # 충돌 매니저 준비(메쉬 1회 등록)
    cm = CollisionManager()
    cm.add_object("part", mesh)

    reports = []
    for k, cand in enumerate(cands):
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in patches or pid_j not in patches:
            reports.append(dict(pair_index=k, patch_i=pid_i, patch_j=pid_j,
                                feasible=False))
            continue
        p_i, p_j = patches[pid_i], patches[pid_j]

        # --- 1) pad OBB 생성 & 충돌 검사 ---
        # 양쪽 패드 OBB 생성
        box_i = make_pad_box_at_patch(p_i, pad_w, pad_h, pad_d, clearance_out)
        box_j = make_pad_box_at_patch(p_j, pad_w, pad_h, pad_d, clearance_out)
        # 충돌 검사 (각각 독립적으로 검사)
        collide_i = cm.in_collision_single(box_i)
        collide_j = cm.in_collision_single(box_j)
        feasible = (not collide_i) and (not collide_j)

        # --- 2) 모멘트 계산 (COM 기준) ---
        # 모멘트(평행 그리퍼 조임 방향: -n)
        ci = np.asarray(p_i["centroid"], float);  cj = np.asarray(p_j["centroid"], float)
        Fi = unit(-np.asarray(p_i["normal"], float))
        Fj = unit(-np.asarray(p_j["normal"], float))
        tau = np.cross(ci - com, Fi) + np.cross(cj - com, Fj)
        moment = float(np.linalg.norm(tau))

        report = dict(
            pair_index=k,
            patch_i=pid_i, patch_j=pid_j,
            feasible=feasible,
            moment=moment
        )
        reports.append(report)

    report_sorted = sorted( # Feasible, moment 정렬
        reports,
        key=lambda r: (not r.get("feasible", False), r.get("moment", float("inf")))
    )
    return report_sorted

# --- yaw만 스윕하여 feasibility 검사(+ 모멘트 계산/정렬) ---
def check_gripper_feasibility_with_yaw(
    result: dict,
    pad_w: float = PadParams.pad_w,     # 폭 (u 방향)
    pad_h: float = PadParams.pad_h,     # 높이 (v 방향)
    pad_d: float = PadParams.pad_d,      # 두께 (n 방향)
    clearance_out: float = 20.0,
    yaw_grid_deg = [0, 90, 30, 60],
    use_mesh: str = "mesh_quad"
):
    mesh    = result[use_mesh]
    patches = {p["id"]: p for p in result["patches"]}
    pairs   = result.get("top_k", [])

    # COM
    try:    com = mesh.center_mass
    except: com = 0.5 * (mesh.bounds[0] + mesh.bounds[1])

    cm = trimesh.collision.CollisionManager()
    cm.add_object("part", mesh)

    reports=[]
    for k, cand in enumerate(pairs):
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in patches or pid_j not in patches:
            reports.append(dict(pair_index=k, patch_i=pid_i, patch_j=pid_j,
                                feasible=False))
            continue
        p_i, p_j = patches[pid_i], patches[pid_j]

        feasible=False; feasible_yaw=None
        for yaw in yaw_grid_deg:
            box_i = make_rot_pad_box_at_patch(p_i, pad_w,pad_h,pad_d,clearance_out, yaw_deg= yaw)
            box_j = make_rot_pad_box_at_patch(p_j, pad_w,pad_h,pad_d,clearance_out, yaw_deg=-yaw)
            if cm.in_collision_single(box_i): continue
            if cm.in_collision_single(box_j): continue
            feasible=True; feasible_yaw=yaw
            break

        # 모멘트(평행 그리퍼 조임 방향: -n)
        ci = np.asarray(p_i["centroid"], float);  cj = np.asarray(p_j["centroid"], float)
        Fi = unit(-np.asarray(p_i["normal"], float))
        Fj = unit(-np.asarray(p_j["normal"], float))
        tau = np.cross(ci - com, Fi) + np.cross(cj - com, Fj)
        moment = float(np.linalg.norm(tau))

        report = dict(
            pair_index=k,
            patch_i=pid_i, patch_j=pid_j,
            feasible=feasible,
            feasible_yaw=feasible_yaw,
            moment=moment
        )
        reports.append(report)

    # feasible 우선, 모멘트 오름차순
    reports_sorted = sorted(reports, key=lambda r: (not r.get("feasible", False), r.get("moment", float("inf"))))
    return reports_sorted