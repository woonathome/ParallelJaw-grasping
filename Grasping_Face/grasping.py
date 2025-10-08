# Compute Patch Pairs, Feasible Grasping Pairs

import json, math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
import numpy as np
import trimesh
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
    overlap_weight:   float = 1 # TODO: 추후 다른 기준 추가 시 조정
    distance_weight:  float = 1 # TODO: 추후 다른 기준 추가 시 조정
    inertia_weight:   float = 1 # TODO: 추후 다른 기준 추가 시 조정
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
    mesh_filter = trimesh.Trimesh(vertices=V, faces=F, process=True)
    return mesh_filter, mesh_quad

def split_long_edges(mesh: trimesh.Trimesh, max_iter: int = 100):
    """
    삼각형별 긴 edge를 기준으로 반복 subdivide
    - mesh: trimesh.Trimesh
    - max_iter: 무한 루프 방지를 위한 최대 반복 횟수
    """
    bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    max_len = bbox_diag * 0.3 # max_len: 허용하는 최대 edge 길이

    V = mesh.vertices.copy()
    F = mesh.faces.copy()

    for it in range(max_iter):
        new_faces = []
        added = False

        for f in F:
            tri = V[f]
            # edge 길이 계산
            edges = [(0,1), (1,2), (2,0)]
            lens = [np.linalg.norm(tri[i] - tri[j]) for i,j in edges]
            max_idx = int(np.argmax(lens))
            i,j = edges[max_idx]
            if lens[max_idx] > max_len:
                added = True
                # 중점 추가
                mid = 0.5 * (tri[i] + tri[j])
                V = np.vstack([V, mid])
                mid_idx = len(V) - 1

                k = 3 - i - j  # 나머지 점 인덱스
                # 삼각형을 두 개로 분할
                new_faces.append([f[i], mid_idx, f[k]])
                new_faces.append([mid_idx, f[j], f[k]])
            else:
                new_faces.append(f)

        F = np.array(new_faces, dtype=np.int64)

        if not added:  # 더 이상 분할 필요 없음
            break

    return trimesh.Trimesh(vertices=V, faces=F, process=True)

def merge_small_faces(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    짧은 edge를 기준으로 반복 병합 (Vertex Clustering 활용)
    - mesh: trimesh.Trimesh
    - min_len: 허용하는 최소 edge 길이. 이 길이보다 가까운 vertex들은 병합
    """

    bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    min_len = bbox_diag * 0.02 # max_len: 허용하는 최대 edge 길이

    # 1. Trimesh를 Open3D Mesh로 변환
    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces)
    )

    # 2. Vertex Clustering을 이용한 메쉬 간소화
    # voxel_size는 병합될 vertex간의 최대 거리 = min_len과 동일
    merged_mesh_o3d = o3d_mesh.simplify_vertex_clustering(
        voxel_size=min_len,
        contraction=o3d.geometry.SimplificationContraction.Average
    )

    # 3. 불필요한 데이터 정리
    merged_mesh_o3d.remove_degenerate_triangles()
    merged_mesh_o3d.remove_unreferenced_vertices()

    # 4. 다시 Trimesh로 변환하여 반환
    new_vertices = np.asarray(merged_mesh_o3d.vertices)
    new_faces = np.asarray(merged_mesh_o3d.triangles)
    
    return trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=True)


# --------------------------
# Geometry helpers
# --------------------------
def Rotx(a):
    c,s=np.cos(np.deg2rad(a)),np.sin(np.deg2rad(a))
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def Roty(a):
    c,s=np.cos(np.deg2rad(a)),np.sin(np.deg2rad(a))
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])
def Rotz(a):
    c,s=np.cos(np.deg2rad(a)),np.sin(np.deg2rad(a))
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def to44(R,t):
    H=np.eye(4); H[:3,:3]=R
    H[:3,3]=np.asarray(t).reshape(3)
    return H

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

def rotate_vector_about_axis(v, axis, ang_deg):
    """
    v: 회전시킬 벡터
    axis: 회전축 (정규화)
    ang_deg: 회전 각도 (deg)
    """
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    theta = np.deg2rad(ang_deg)
    v = np.asarray(v, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    return v * c + np.cross(axis, v) * s + axis * (np.dot(axis, v)) * (1 - c)

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
                              lift_mm: float = 5.0, yaw_deg: float = 0.0):
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
    # box = trimesh.creation.box(extents=[float(pad_w) / 2, float(pad_h) / 2, ext_z])
    box = trimesh.creation.box(extents=[float(pad_w), float(pad_h), ext_z])

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
    # coplanar_tol: float = 1e-3,
) -> List[PlanarPatch]:
    face_normals = mesh.face_normals
    faces = mesh.faces
    face_adjacency = mesh.face_adjacency
    F = len(faces)

    min_patch_area = mesh.area * 0.001 
    # max_patch_area = mesh.area * 0.0001

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
                # verts_nb = tri_verts[nb].reshape(-1, 3)
                # d = np.abs((verts_nb @ n_seed) - b_seed)
                # if np.max(d) > coplanar_tol:
                #     continue
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


def orient_patch_normals(mesh_quad: trimesh.Trimesh, patches):
    """
    패치 법선을 mesh 기준으로 일관되게 정렬.
    mesh.contains로 실제 안/밖을 체크
    """
    # 오프셋 길이: 모델 크기 대비 아주 작게
    bbox_diag = float(np.linalg.norm(mesh_quad.bounds[1] - mesh_quad.bounds[0]))

    for p in patches:
        n = np.asarray(p.normal if hasattr(p, "normal") else p["normal"], float)
        n = unit(n)
        c = np.asarray(p.centroid if hasattr(p, "centroid") else p["centroid"], float)
        eps = bbox_diag * 0.01

        flip = False

        for scale in range(1, 11):
            contain1 = mesh_quad.contains([c + eps * n])[0] # normal 방향으로 이동
            contain2 = mesh_quad.contains([c - eps * n])[0] # normal 반대로 이동

            if contain1 != contain2: # 둘이 다를 경우
                flip = contain1  # 안이면 뒤집어 바깥 향하게
                break
            else: # 둘이 같을 경우
                eps = eps * scale # 탐색 길이 늘리기
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

    # width patch 간 최소 거리로 업뎃 필요 (현재: centroid 간 거리)
    c_a = np.asarray(patch_a.centroid)
    c_b = np.asarray(patch_b.centroid)
    width = np.linalg.norm(c_a - c_b)

    # TODO: c-n 최소 거리 점수: c-n 직선과 c간의 거리 점수(패치 2개의 평행성 검사 2)
    bbox_diag = float(np.linalg.norm(mesh_for_overlap.bounds[1] - mesh_for_overlap.bounds[0]))
    dist_a = point_line_distance(c_a, n_a, c_b)
    dist_b = point_line_distance(c_b, n_b, c_a)
    s_distance = 1 - (dist_a + dist_b) / bbox_diag

    # TODO: 회전 관성 점수: ca-cb 직선과 mesh COM 간 거리 점수
    com_mesh = mesh_for_overlap.center_mass
    direction = (c_a - c_b) / width
    dist_inertia = point_line_distance(c_b, direction, com_mesh)
    s_inertia = 1 - dist_inertia / bbox_diag

    # 최종 점수(가중합)
    score = params.distance_weight * s_distance + params.overlap_weight * s_overlap + params.inertia_weight * s_inertia

    return PatchPairCandidate(
        patch_i=patch_a.id,
        patch_j=patch_b.id,
        normal=n_a.tolist(),
        width=width,
        score=score,
        terms={
            "overlap": s_overlap,
            "distance": s_distance,
            # "inertia": s_inertia
              },
    )


# --------------------------
# Main API
# --------------------------
def compute_best_patch_pairs(
    mesh_path: str,
    mesh_max_triangles: int = 500,         # 원본 mesh 삼각형 개수
    angle_deg: float = 7.0,                # 패치 병합 허용 각도 (↑면 패치 수 ↓)
    # coplanar_tol: float = 1e-3,            # 공면성 허용 오차 (↑면 패치 수 ↓)
    min_opening: float = 10.0,              # 그리퍼 최소 개구(mm)
    max_opening: float = 140.0,              # 그리퍼 최대 개구(mm)
    angle_tolerance_deg: float = 10.0,     # 패치 페어 정반대 허용 각도
    top_k: int = 1,                        # 상위 후보 수
) -> Dict[str, Any]:
    
    mesh_filter, mesh_quad = load_uniform_mesh_with_open3d(mesh_path, target_triangles=mesh_max_triangles)
    # mesh_filter: (면적 기준 필터링 이후, watertight X), mesh_quad: (면적 기준 필터링 이전, watertight O)
    remesh = split_long_edges(mesh_filter)
    remesh = merge_small_faces(remesh)

    # bbox_diag = float(np.linalg.norm(remesh.bounds[1] - remesh.bounds[0]))
    # coplanar_tol = bbox_diag * 0.05
    patches = extract_planar_patches(remesh, angle_deg=angle_deg)
    # coplanar_tol=coplanar_tol)
    patches = orient_patch_normals(mesh_quad, patches)  # 모두 바깥쪽으로 정렬

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
            # i-j 법선 각도: 180° - tol 보다 작으면 충분히 반대가 아님 -> 스킵
            dot = float(ni @ nvecs[j])
            ang = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
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


# --- yaw 스윕하여 feasibility 검사(+ 모멘트 계산/정렬) ---
def check_gripper_feasibility_faces_with_yaw(
    result: dict,
    pad_w: float = PadParams.pad_w,
    pad_h: float = PadParams.pad_h,
    pad_d: float = PadParams.pad_d,
    clearance_out: float = 20.0,
    # yaw_grid_deg = [0, 90, 30, -30, 60, -60],
    yaw_grid_deg = [0, 90],
    use_mesh: str = "mesh_patches",
    check_mesh: str = "mesh_quad"
):
    mesh_ch = result[check_mesh] # 간섭 검사용: 면적 필터 안 된 메시 
    mesh    = result[use_mesh]
    patches = {p["id"]: p for p in result["patches"]}
    pairs   = result.get("top_k", [])

    try:    com = mesh_ch.center_mass
    except: com = 0.5 * (mesh_ch.bounds[0] + mesh_ch.bounds[1])

    cm = trimesh.collision.CollisionManager()
    cm.add_object("part", mesh_ch) # 간섭 검사용: 면적 필터 안 된 메시 

    reports = []
    for k, cand in enumerate(pairs):
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in patches or pid_j not in patches:
            reports.append(dict(pair_index=k, feasible=False))
            continue
        p_i, p_j = patches[pid_i], patches[pid_j]
        n_i = unit(np.asarray(p_i["normal"], float))
        n_j = unit(np.asarray(p_j["normal"], float))

        # pair별로 가능한 모든 후보를 찾기
        reports_for_this_pair = []
        mesh_F, mesh_V = mesh.faces, mesh.vertices
        for f_i in p_i["face_indices"]:
            ci = mesh_V[mesh_F[f_i]].mean(axis=0)

            best_j   = None
            best_cj  = None
            best_scr = -1.0
            for f_j in p_j["face_indices"]:
                cj = mesh_V[mesh_F[f_j]].mean(axis=0)
                d  = cj - ci
                nd = np.linalg.norm(d)
                if nd < 1e-12:
                    continue
                d_hat = d / nd
                scr = abs(float(d_hat @ (-n_i))) * abs(float((-d_hat) @ n_j))
                if scr > best_scr:
                    best_scr, best_j, best_cj = scr, f_j, cj
            if best_j is None:
                continue
            # 엇갈린 face pair 건너뛰기
            alignment_dist = point_line_distance(ci, -n_i, best_cj)
            if alignment_dist > max(pad_w, pad_h) / 2:
                continue

            for yaw in yaw_grid_deg:
                box_i = make_rot_pad_box_at_patch(
                    {"centroid": ci, "normal": n_i},
                    pad_w, pad_h, pad_d, clearance_out, yaw_deg=yaw
                )
                box_j = make_rot_pad_box_at_patch(
                    {"centroid": best_cj, "normal": n_j},
                    pad_w, pad_h, pad_d, clearance_out, yaw_deg=-yaw
                )
                # 충돌이 발생하면 이 yaw는 건너뛰고 다음 yaw를 검사.
                if cm.in_collision_single(box_i): continue
                if cm.in_collision_single(box_j): continue

                # 충돌이 없으면, 이 yaw는 가능한 후보. yaw 값에 대한 리포트를 생성.
                closing_dir_base = unit(n_i - n_j)
                u, v, n_basis = basis_from_normal(closing_dir_base)
                B = np.column_stack([u, v, n_basis])
                Rz_mat = Rotz(yaw)
                R_world = B @ Rz_mat
                pad_w_dir = R_world[:, 0]

                midpoint = 0.5 * (ci + best_cj)
                # current_dist_inertia = point_line_distance(midpoint, pad_w_dir, com)
                current_dist = np.linalg.norm(midpoint - com)
                
                Fi = -n_i; Fj = -n_j
                tau = np.cross(ci - com, Fi) + np.cross(best_cj - com, Fj)
                current_moment = float(np.linalg.norm(tau))
                
                # 리포트를 생성하고 리스트에 추가.
                cand_report = dict(
                    pair_index=k,
                    patch_i=pid_i, patch_j=pid_j,
                    face_i=f_i, face_j=best_j,
                    feasible=True,
                    feasible_yaw=yaw, # 현재 yaw 값을 저장
                    moment=current_moment,
                    dist=current_dist
                )
                reports_for_this_pair.append(cand_report)

        if not reports_for_this_pair:
            reports.append(dict(pair_index=k, patch_i=pid_i, patch_j=pid_j, feasible=False))
        else:
            reports.extend(reports_for_this_pair)

    reports_sorted = sorted(
        reports, key=lambda r: (not r.get("feasible", False),
                                r.get("dist", float("inf")),
                                r.get("moment", float("inf")))
    )
    return reports_sorted


# --------------------------
# Object Pose helpers
# --------------------------
def origin_offset_from_stroke(stroke):
    _STROKE_TO_DIST = {
        0:82, 9:81, 18:80, 35:80, 67:77, 96:71, 122:63, 145:52
    }
    _strokes = np.array(list(_STROKE_TO_DIST.keys()), dtype=float)
    _dists   = np.array(list(_STROKE_TO_DIST.values()), dtype=float)
    return float(np.interp(float(stroke), _strokes, _dists))

# object 좌표계에서 본 gripper pose (H_GO)
def build_gripper_pose_obj(p_i, p_j, yaw_deg=0.0):
    """
    객체 좌표계 상 두 패치 pair로부터 그리퍼 좌표계 pose 생성.
    - origin: 두 centroid의 중점 (패드 중앙)
    - x축: i -j 평균  (closing 방향) 
    반환: (4x4 pose 행렬, stroke=패드 거리)
    """
    ci = np.asarray(p_i["centroid"], float)
    cj = np.asarray(p_j["centroid"], float)
    ni = np.asarray(p_i["normal"], float)
    nj = np.asarray(p_j["normal"], float)

    # origin = 중점
    o_pad = 0.5 * (ci + cj)

    # closing 방향 (x축)
    x_axis = unit(ni - nj) # make_box_rot 함수에서 pad_i normal yaw_deg 회전
    u, v, n = basis_from_normal(x_axis)
    B = np.column_stack([u, v, n])
    Rz = Rotz(yaw_deg)
    R_world = B @ Rz
    z_axis =   R_world[:, 0]  # pad w 방향
    y_axis = - R_world[:, 1]  # pad h 방향
    x_axis =   R_world[:, 2]  # closing 방향

    if z_axis[-1] >= 0:
        # R_world = np.linalg.inv(R_world)
        z_axis = - z_axis  # pad w 방향
        y_axis = - y_axis  # pad h 방향
        x_axis = np.cross(y_axis, z_axis)  # closing 방향

    # 직교 보정 (x,y,z 순서)
    R = np.column_stack([x_axis, y_axis, z_axis])

    # 그리퍼 stroke에 따른 그리퍼 원점 업데이트 (z축 반대방향으로 stroke offset 만큼 이동)
    stroke = np.linalg.norm(cj - ci)
    d = origin_offset_from_stroke(stroke)
    o = o_pad + (- z_axis * d)
    H_OG = to44(R, o)

    return H_OG, stroke

def build_gripper_pose_obj_OPE(p_i, p_j, yaw_deg=0.0, H_OC=np.eye(4)):
    """
    객체 좌표계 상 두 패치 pair로부터 그리퍼 좌표계 pose 생성.
    - origin: 두 centroid의 중점 (패드 중앙)
    - x축: i -j 평균  (closing 방향) 
    반환: (4x4 pose 행렬, stroke=패드 거리)
    """
    ci = np.asarray(p_i["centroid"], float)
    cj = np.asarray(p_j["centroid"], float)
    ni = np.asarray(p_i["normal"], float)
    nj = np.asarray(p_j["normal"], float)

    o_pad = 0.5 * (ci + cj)
    
    # 그리퍼 좌표계
    x_axis_base = unit(ni - nj) 
    u, v, n = basis_from_normal(x_axis_base)
    B = np.column_stack([u, v, n])
    Rz = Rotz(yaw_deg)
    R_world = B @ Rz
    z_axis =   R_world[:, 0]
    y_axis = - R_world[:, 1]
    x_axis =   R_world[:, 2]

    # 2. 방향 결정 로직 수정
    # OPE 결과(H_OC)를 이용해 그리퍼 z축(접근 방향)을 카메라 좌표계 기준으로 변환
    z_axis_in_camera = H_OC[:3, :3] @ z_axis
    # print(z_axis_in_camera)

    # 그리퍼의 접근 방향이 카메라 좌표계의 위쪽(-Z)을 향하면, z 축을 뒤집어 아래(+Z)를 향하도록 함
    # 카메라 좌표계: Z 정면, Y 아래, X 오른쪽
    if z_axis_in_camera[2] < 0: 
        z_axis = - z_axis  # pad w 방향
        y_axis = - y_axis  # pad h 방향
        x_axis = np.cross(y_axis, z_axis)  # closing 방향
        
    # 직교 보정 (x,y,z 순서)
    R = np.column_stack([x_axis, y_axis, z_axis])
    
    stroke = np.linalg.norm(cj - ci)
    d = origin_offset_from_stroke(stroke)
    o = o_pad + (- z_axis * d)
    H_OG = to44(R, o)

    return H_OG, stroke

def ee_delta_pose_des(H_OC: np.ndarray, H_OG: np.ndarray):
    """
    n: now(캡쳐 위치), d: destination(인식된 위치)
    입력:
      H_OC : C좌표계 기준 O좌표계 원점 포즈
      H_OG : O좌표계 기준 G좌표계 원점 포즈
    출력:
      H_EdEn : now EE 기준 des EE 포즈 (EE 움직일 상대 좌표)
    """
    # # 좌표계 시각화
    # H_E = np.eye(4,4) # TODO: 로봇 컨트롤러 신호 받아 변환행렬 만들기 (현재는 EE 좌표계 기준이라 개발 필요 X)

    # 고정변환
    H_GnEn = to44(Rotx(180) @ Rotz(90), [0,0,-135])   # EE -> Grip (TODO 위 반영 업뎃)
    H_GnCn = to44(np.eye(3), [10,60,15])           # Cam -> Grip [0,48,6]
    H_CnGn = np.linalg.inv(H_GnCn)
    H_CnEn = H_GnEn @ H_CnGn                        # EE -> Cam
    H_EG = np.linalg.inv(H_GnEn)

    H_OdCn = H_OC
    H_OdEn = H_CnEn @ H_OdCn
    # H_GdOd = np.linalg.inv(H_OG)
    H_GdEn = H_OdEn @ H_OG
    H_EdEn = H_GdEn @ H_EG
    return H_EdEn, H_OdEn