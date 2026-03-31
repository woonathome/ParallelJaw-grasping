# Compute Patch Pairs, Feasible Grasping Pairs

import json, math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import trimesh
import open3d as o3d
from scipy.spatial import ConvexHull

@dataclass
class PlanarPatch:
    id: int
    face_indices: List[int]
    normal: List[float]
    b: float
    area: float
    centroid: List[float]
    centroid_check: List[float]

@dataclass
class PatchPairParams:
    min_opening: float = 1.0
    max_opening: float = 140.0
    angle_tolerance_deg: float = 15.0
    overlap_weight:   float = 1 # TODO: 추후 다른 기준 추가 시 조정
    distance_weight:  float = 0.2 # TODO: 추후 다른 기준 추가 시 조정
    inertia_weight:   float = 0.4 # TODO: 추후 다른 기준 추가 시 조정
    area_weight:      float = 0.4 # TODO: 추후 다른 기준 추가 시 조정
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
class PadParams: # Robotiq AG145 기준
    pad_w: float = 34.0 # 34
    pad_h: float = 21.0 # 21
    pad_d: float = 7.0 # 7
    mu: float = 0.8


# --------------------------
# Mesh helpers
# --------------------------
def load_uniform_mesh_with_open3d(path, target_triangles=500, min_area=0.001): # TODO: min_area 기준 업데이트 필요
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

    # 3) 삼각형 면적 직접 계산 → 작은 face 제거
    V = np.asarray(o3_simpl.vertices)
    F = np.asarray(o3_simpl.triangles, dtype=np.int64)
    v0, v1, v2 = V[F[:,0]], V[F[:,1]], V[F[:,2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    # TODO: 
    # mask = areas >= float(min_area)
    mask = areas >= 0.0

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
    # bbox_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    # max_len = bbox_diag * 0.3
    # max_len = np.sqrt(PadParams.pad_h ** 2 + PadParams.pad_w ** 2) / 2
    max_len = min(PadParams.pad_h, PadParams.pad_w) # max_len: 허용하는 최대 edge 길이

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

# def point_line_distance(c_a:np.ndarray, n_a:np.ndarray, c_b:np.ndarray):
    # """
    # c_a 시작점, n_a 방향, c_b 거리측정 목표지점
    # """
    # n_u = unit(n_a)
    # r = c_b - c_a
    # t = float(np.dot(r, n_u)) # 직선 파라미터
    # dist = float(np.linalg.norm(r - t * n_u))
    # return dist

def point_line_distance(c_a:np.ndarray, n_a:np.ndarray, c_b:np.ndarray):
    """
    c_a: 반직선(ray)의 시작점
    n_a: 반직선의 방향 벡터
    c_b: 거리 측정 목표 지점
    """
    n_u = unit(n_a)
    # 반직선 시작점에서 목표점까지의 벡터
    r = c_b - c_a
    t = float(np.dot(r, n_u))
    if t >= 0.0:
        dist = float(np.linalg.norm(r - t * n_u))
    else:
        dist = float(np.linalg.norm(r)) # == np.linalg.norm(c_b - c_a)
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
                          lift_mm: float = 3.0):
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

def make_pad_cylinder_at_patch(patch, pad_w, pad_h, pad_d, clearance_out,
                               lift_mm: float = 2.0):
    """
    패치 중심에서 normal 방향으로 회전하는 패드+클리어런스의 swept volume을 근사하는 원기둥 메쉬를 생성

    Args:
        patch: 패치 정보 (dict or object, 'centroid', 'normal' 필요)
        pad_w, pad_h, pad_d: 패드 크기
        clearance_out: 패드 바깥쪽 간격
        lift_mm: 패치 표면에서 띄울 수치

    Returns:
        trimesh.Trimesh: 원기둥 메쉬 객체.
    """
    n = unit(np.asarray(patch["normal"], float))
    c = np.asarray(patch["centroid"], float)

    height = float(pad_d + clearance_out)
    # radius = 0.5 * np.sqrt(pad_w**2 + pad_h**2)
    radius = 0.5 * min(pad_w, pad_h)

    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=16)

    # 1. Z축을 normal 벡터 n으로 회전
    transform_rot = trimesh.geometry.align_vectors([0, 0, 1], n)
    # 2. 원기둥의 중심을 패치 중심에서 lift_mm + height/2 만큼 이동
    center_translation = c + n * (lift_mm + 0.5 * height)
    transform_trans = trimesh.transformations.translation_matrix(center_translation)
    # 변환 행렬 결합 (회전 후 이동)
    transform_matrix = transform_trans @ transform_rot
    # 원기둥에 변환 적용
    cylinder.apply_transform(transform_matrix)

    return cylinder

# --------------------------
# Planar patch extraction
# --------------------------
def extract_planar_patches(
    mesh: trimesh.Trimesh,
    angle_deg: float = 15.0,
) -> List[PlanarPatch]:
    face_normals = mesh.face_normals
    faces = mesh.faces
    face_adjacency = mesh.face_adjacency
    F = len(faces)

    # min_patch_area = mesh.area * 0.0005
    min_patch_area = mesh.area * 0
    # max_patch_area = mesh.area * 0.0005

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
                used[nb] = True
                region.append(nb)
                queue.append(nb)

        verts_region = tri_verts[region].reshape(-1, 3)
        n_reg, b_reg = plane_from_points(verts_region)

        area_reg = float(tri_areas[region].sum())
        if area_reg < min_patch_area:     # 면적이 너무 작은 patch 필터링
            continue
    
        centroid_reg = (verts_region.max(axis=0) + verts_region.min(axis=0)) / 2
        centroid_check = verts_region.mean(axis=0) # Normal 방향 확인용

        patches.append(
            PlanarPatch(
                id=pid,
                face_indices=region,
                normal=n_reg.tolist(),
                b=float(b_reg),
                area=area_reg,
                centroid=centroid_reg.tolist(),
                centroid_check=centroid_check.tolist(),
            )
        )
        pid += 1

    return patches


def orient_patch_normals(mesh_quad: trimesh.Trimesh, mesh_patches: trimesh.Trimesh, patches):
    """
    패치 법선을 mesh 기준으로 일관되게 정렬.
    mesh.contains로 실제 안/밖을 체크
    """
    # 오프셋 길이: 모델 크기 대비 아주 작게
    bbox_diag = float(np.linalg.norm(mesh_quad.bounds[1] - mesh_quad.bounds[0]))
    mesh_F, mesh_V = mesh_patches.faces, mesh_patches.vertices

    for p in patches:
        n = np.asarray(p.normal if hasattr(p, "normal") else p["normal"], float)
        n = unit(n)

        # Patch 내 첫번째 Face에서 normal로 나가면서 검사
        f = p.face_indices[0]
        stp = mesh_V[mesh_F[f]].mean(axis=0) # 기준 변경: patch 무게중심 -> 단일 face 중심
        eps = bbox_diag * 0.001
        flip = False
        for scale in range(1, 100):
            contain1 = mesh_quad.contains([stp + eps * n])[0] # normal 방향으로 이동
            contain2 = mesh_quad.contains([stp - eps * n])[0] # normal 반대로 이동

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
                     mesh_for_overlap: trimesh.Trimesh,
                     com_mesh: np.ndarray,
                     bbox_diag: np.ndarray) -> PatchPairCandidate:
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
    # bbox_diag = float(np.linalg.norm(mesh_for_overlap.bounds[1] - mesh_for_overlap.bounds[0]))
    dist_a = point_line_distance(c_a, -n_a, c_b)
    dist_b = point_line_distance(c_b, -n_b, c_a)
    s_distance = 1 - (dist_a + dist_b) / bbox_diag

    # TODO: 회전 관성 점수: ca-cb 직선과 mesh COM 간 거리 점수
    # com_mesh = mesh_for_overlap.center_mass
    direction_a, direction_b = (c_b - c_a) / width, (c_a - c_b) / width
    dist_COM_a, dist_COM_b = point_line_distance(c_a, direction_a, com_mesh), point_line_distance(c_b, direction_b, com_mesh)
    dist_inertia = max(dist_COM_a, dist_COM_b)
    s_inertia = 1 - dist_inertia / bbox_diag

    # Patch Area 간 차이에 따른 지표(비슷할수록 큼)
    if patch_a.area > 0 and patch_b.area > 0:
        s_area = min(patch_a.area / patch_b.area, patch_b.area / patch_a.area)
    else:
        s_area = 0.0


    # 최종 점수(가중합)
    score = (params.distance_weight * s_distance +
            #  params.overlap_weight * s_overlap +
             params.inertia_weight * s_inertia +
             params.area_weight * s_area)

    return PatchPairCandidate(
        patch_i=patch_a.id,
        patch_j=patch_b.id,
        normal=n_a.tolist(),
        width=width,
        score=score,
        terms={
            "distance": s_distance,
            "inertia": s_inertia,
            "area": s_area,
            "overlap": s_overlap # score 합산 미반영
            },
    )


# --------------------------
# Main API
# --------------------------
def compute_best_patch_pairs(
    mesh_path: str,
    mesh_max_triangles: int = 1000,         # 원본 mesh 삼각형 개수
    angle_deg: float = 7.0,                # 패치 병합 허용 각도 (↑면 패치 수 ↓)
    min_opening: float = 1.0,              # 그리퍼 최소 개구(mm)
    max_opening: float = 140.0,              # 그리퍼 최대 개구(mm)
    angle_tolerance_deg: float = 10.0,     # 패치 페어 정반대 허용 각도
    top_k: int = 100,                        # 상위 후보 수
) -> Dict[str, Any]:
    
    mesh_filter, mesh_quad = load_uniform_mesh_with_open3d(mesh_path, target_triangles=mesh_max_triangles)

    mesh_COM = mesh_quad.center_mass
    mesh_bound = float(np.linalg.norm(mesh_quad.bounds[1] - mesh_quad.bounds[0]))
    # mesh_filter: (면적 기준 필터링 이후, watertight X), mesh_quad: (면적 기준 필터링 이전, watertight O)
    remesh = split_long_edges(mesh_filter)
    # remesh = merge_small_faces(remesh) # TODO: 형상 왜곡이 너무 심해서 제거

    patches = extract_planar_patches(remesh, angle_deg=angle_deg)
    patches = orient_patch_normals(mesh_quad, remesh, patches)  # 모두 바깥쪽으로 정렬

    params = PatchPairParams(
        min_opening=min_opening,
        max_opening=max_opening,
        angle_tolerance_deg=angle_tolerance_deg,
    )

    cands: List[PatchPairCandidate] = []

    # patch pair 탐색 및 scoring
    for i, j in itertools.combinations(range(len(patches)), 2):
        patch_i = patches[i]
        patch_j = patches[j]
        ni = np.array(patch_i.normal)
        nj = np.array(patch_j.normal)
        ci = np.array(patch_i.centroid)
        cj = np.array(patch_j.centroid)

        # 평행성 검사: patch i-j 법선 각도가 180° - tol 보다 작으면 충분히 반대가 아님
        dot_normal = float(ni @ nj)
        ang_normal = math.degrees(math.acos(max(-1.0, min(1.0, dot_normal))))
        if ang_normal < 180.0 - params.angle_tolerance_deg: 
            continue

        # 마주봄 검사: patch ci -> cj와 ni가 90도 이상, 
        vec_ij, vec_ji = unit(cj - ci), unit(ci - cj)
        dot_i, dot_j = float(vec_ij @ ni), float(vec_ji @ nj) 
        ang_i = math.degrees(math.acos(max(-1.0, min(1.0, dot_i))))
        ang_j = math.degrees(math.acos(max(-1.0, min(1.0, dot_j))))
        if ang_i < 90.0 or ang_j < 90.0:
            continue

        # 검사 통과된 후보들에 대해 score 계산
        cand = score_patch_pair(patch_i, patch_j, params, remesh, mesh_COM, mesh_bound)
        
        # 폭/면적/점수 필터
        if cand.width < params.min_opening or cand.width > params.max_opening:
            continue
        if cand.terms['overlap'] <= 0: # 상호 겹치는 부분이 없는 pair는 넘김
            continue
        if cand.score <= 0.0:
            continue
        
        cands.append(cand)

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
    clearance_out: float = 10.0,
    # yaw_grid_deg = [0, 90, 45, -45],
    yaw_grid_deg = [0, 90],
    use_mesh: str = "mesh_patches",
    check_mesh: str = "mesh_quad"
):
    mesh    = result[use_mesh]
    mesh_ch = result[check_mesh] # 간섭 검사용: 면적 필터 안 된 메시 
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
            alignment_dist_i = point_line_distance(ci, -n_i, best_cj)
            alignment_dist_j = point_line_distance(best_cj, -n_j, ci)
            dist_criteria = min(pad_w, pad_h) / 5 # 엇갈림 기준
            # dist_criteria = 5 # 엇갈림 기준, mm
            if alignment_dist_i > dist_criteria or alignment_dist_j > dist_criteria:
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
                # 충돌이 발생하면 이 yaw는 건너뛰고 다음 yaw를 검사
                if cm.in_collision_single(box_i): continue
                if cm.in_collision_single(box_j): continue
                
                # 충돌이 없는 경우, 리포트 생성
                cand_report = dict(
                    pair_index=k, patch_i=pid_i, patch_j=pid_j,
                    face_i=f_i, face_j=best_j, feasible=True, feasible_yaw=yaw
                )

                # # Metric 0: Epsilon Quality (TODO: 실행시간 얼마나 달라지는지 확인)
                # current_epsilon = calculate_squeeze_epsilon_quality(
                #     mesh, p_i['face_indices'], p_j['face_indices'],
                #     ci, n_i, yaw, best_cj, n_j, -yaw, com )

                # Metric 1: Face Pair to COM 거리   
                midpoint = 0.5 * (ci + best_cj)
                current_dist = np.linalg.norm(midpoint - com)
                
                # Metric 2: COM에 가해지는 토크 합산 (Force Closure)   
                Fi = -n_i; Fj = -n_j
                tau = np.cross(ci - com, Fi) + np.cross(best_cj - com, Fj)
                current_moment = float(np.linalg.norm(tau))
                
                # 리포트에 메트릭 추가
                # cand_report['epsilon'] = current_epsilon 
                cand_report['moment'] = current_moment
                cand_report['dist'] = current_dist
                reports_for_this_pair.append(cand_report)

        if not reports_for_this_pair:
            reports.append(dict(pair_index=k, patch_i=pid_i, patch_j=pid_j, feasible=False))
        else:
            reports.extend(reports_for_this_pair)

    # epsilon 높은 / dist 낮은 / moment 낮은 순서로 정렬 (오름차순, 동점자 처리)
    reports_sorted = sorted(
        [r for r in reports if r.get("feasible")], # 유효한 리포트만 정렬
        key=lambda r: (
            # r.get("epsilon", -1.0),
            -r.get("dist", float("inf")),
            -r.get("moment", float("inf"))
            ),
        reverse=True
    )

    return reports_sorted


def check_gripper_feasibility_faces_with_rotation(
    result: dict,
    pad_w: float = PadParams.pad_w,
    pad_h: float = PadParams.pad_h,
    pad_d: float = PadParams.pad_d,
    clearance_out: float = 10.0,
    use_mesh: str = "mesh_patches",
    check_mesh: str = "mesh_quad"
):
    mesh    = result[use_mesh]
    mesh_ch = result[check_mesh] # 간섭 검사용: 면적 필터 안 된 메시 
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
            alignment_dist_i = point_line_distance(ci, -n_i, best_cj)
            alignment_dist_j = point_line_distance(best_cj, -n_j, ci)
            dist_criteria = min(pad_w, pad_h) / 5 # 엇갈림 기준
            if alignment_dist_i > dist_criteria or alignment_dist_j > dist_criteria:
                continue

            cyl_i = make_pad_cylinder_at_patch({"centroid": ci,      "normal": n_i}, pad_w / 2, pad_h / 2, pad_d, clearance_out)
            cyl_j = make_pad_cylinder_at_patch({"centroid": best_cj, "normal": n_j}, pad_w / 2, pad_h / 2, pad_d, clearance_out)

            if not cm.in_collision_single(cyl_i) and not cm.in_collision_single(cyl_j):
                midpoint = 0.5 * (ci + best_cj)
                current_dist = np.linalg.norm(midpoint - com) # 두 점으로 잡았을 때 중간점 <-> mesh COM 거리
                Fi = -n_i # 힘 방향 (normal의 반대) 
                Fj = -n_j
                tau = np.cross(ci - com, Fi) + np.cross(best_cj - com, Fj)
                current_moment = float(np.linalg.norm(tau)) # 두 점으로 잡았을 때 걸리는 모멘트

                report = dict(
                    pair_index=k,
                    patch_i=pid_i,
                    patch_j=pid_j,
                    face_i=f_i,
                    face_j=best_j,
                    dist=current_dist,
                    moment=current_moment,
                    feasible=True # 원기둥 검사 통과 의미
                )
                reports.append(report)

    # 1. feasible=True 먼저
    # 2. dist 낮은 순서 (오름차순, 동점자 처리)
    # 3. moment 낮은 순서 (오름차순, 동점자 처리)
    reports_sorted = sorted(
        [r for r in reports if r.get("feasible")], # 유효한 리포트만 정렬
        key=lambda r: (
            r.get("dist", float("inf")),
            r.get("moment", float("inf"))
            ),
        reverse=False
    )

    return reports_sorted

# --------------------------
# Performance-oriented overrides (v2)
# --------------------------
def split_long_edges(
    mesh: trimesh.Trimesh,
    max_iter: int = 100,
    max_faces_after_split: int = 20000,
    max_len: Optional[float] = None,
):
    """
    Split long edges iteratively with a face-count guard to avoid mesh explosion.
    """
    if max_len is None:
        max_len = float(min(PadParams.pad_h, PadParams.pad_w))

    V = mesh.vertices.copy()
    F = mesh.faces.copy()
    if len(F) == 0:
        return mesh.copy()

    for _ in range(max_iter):
        new_faces: List[List[int]] = []
        new_vertices: List[np.ndarray] = []
        added = False

        for f in F:
            tri = V[f]
            edges = ((0, 1), (1, 2), (2, 0))
            lens = [np.linalg.norm(tri[i] - tri[j]) for i, j in edges]
            max_idx = int(np.argmax(lens))
            i, j = edges[max_idx]
            if lens[max_idx] > max_len and len(new_faces) < (2 * max_faces_after_split):
                added = True
                mid = 0.5 * (tri[i] + tri[j])
                mid_idx = len(V) + len(new_vertices)
                new_vertices.append(mid)
                k = 3 - i - j
                new_faces.append([int(f[i]), mid_idx, int(f[k])])
                new_faces.append([mid_idx, int(f[j]), int(f[k])])
            else:
                new_faces.append([int(f[0]), int(f[1]), int(f[2])])

        if new_vertices:
            V = np.vstack([V, np.asarray(new_vertices, dtype=float)])
        F = np.asarray(new_faces, dtype=np.int64)

        if not added:
            break
        if len(F) >= max_faces_after_split:
            break

    return trimesh.Trimesh(vertices=V, faces=F, process=True)


def _point_in_tri_2d_batch(P: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if len(P) == 0:
        return np.zeros(0, dtype=bool)
    v0 = c - a
    v1 = b - a
    v2 = P - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < eps:
        return np.zeros(len(P), dtype=bool)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (u >= -eps) & (v >= -eps) & (w >= -eps)


def _prepare_patch_overlap_cache(
    mesh: trimesh.Trimesh,
    patches: List[Any],
    samples_per_face: int,
) -> Dict[int, Dict[str, np.ndarray]]:
    V = mesh.vertices
    F = mesh.faces
    cache: Dict[int, Dict[str, np.ndarray]] = {}

    for p in patches:
        pid = int(p.id if hasattr(p, "id") else p["id"])
        face_indices = np.asarray(
            p.face_indices if hasattr(p, "face_indices") else p["face_indices"],
            dtype=np.int64,
        )
        if len(face_indices) == 0:
            continue

        n = unit(np.asarray(p.normal if hasattr(p, "normal") else p["normal"], float))
        b = float(p.b if hasattr(p, "b") else p["b"])
        c0 = np.asarray(p.centroid if hasattr(p, "centroid") else p["centroid"], float)
        u, v, _ = basis_from_normal(n)

        faces = F[face_indices]
        tri = V[faces]
        rel = tri - c0.reshape(1, 1, 3)
        tri2d = np.empty((len(faces), 3, 2), dtype=float)
        tri2d[:, :, 0] = rel @ u
        tri2d[:, :, 1] = rel @ v
        tri_bbox_min = tri2d.min(axis=1)
        tri_bbox_max = tri2d.max(axis=1)
        patch_bbox_min = tri_bbox_min.min(axis=0)
        patch_bbox_max = tri_bbox_max.max(axis=0)

        pts = sample_points_on_patch(mesh, face_indices, samples_per_face=samples_per_face)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)

        cache[pid] = {
            "normal": n,
            "b": np.asarray(b, dtype=float),
            "centroid": c0,
            "u": u,
            "v": v,
            "samples": pts,
            "tri2d": tri2d,
            "tri_bbox_min": tri_bbox_min,
            "tri_bbox_max": tri_bbox_max,
            "patch_bbox_min": patch_bbox_min,
            "patch_bbox_max": patch_bbox_max,
        }

    return cache


def _projection_overlap_ratio_cached(src_cache: Dict[str, np.ndarray], dst_cache: Dict[str, np.ndarray]) -> float:
    pts = src_cache["samples"]
    if len(pts) == 0:
        return 0.0

    n_i = src_cache["normal"]
    n_j = dst_cache["normal"]
    denom = -float(n_j @ n_i)
    if abs(denom) < 1e-12:
        return 0.0

    b_j = float(dst_cache["b"])
    t = (b_j - pts @ n_j) / denom
    proj = pts + (t.reshape(-1, 1) * (-n_i))
    rel = proj - dst_cache["centroid"].reshape(1, 3)
    P2 = np.column_stack([rel @ dst_cache["u"], rel @ dst_cache["v"]])
    if len(P2) == 0:
        return 0.0

    patch_box_mask = np.all(
        (P2 >= dst_cache["patch_bbox_min"].reshape(1, 2))
        & (P2 <= dst_cache["patch_bbox_max"].reshape(1, 2)),
        axis=1,
    )
    if not np.any(patch_box_mask):
        return 0.0

    candidate_idx = np.where(patch_box_mask)[0]
    P_local = P2[candidate_idx]
    hit = np.zeros(len(P_local), dtype=bool)

    tri2d = dst_cache["tri2d"]
    tri_bbox_min = dst_cache["tri_bbox_min"]
    tri_bbox_max = dst_cache["tri_bbox_max"]

    for tri_id in range(len(tri2d)):
        remaining = np.where(~hit)[0]
        if len(remaining) == 0:
            break

        Q = P_local[remaining]
        bbox_mask = np.all(
            (Q >= tri_bbox_min[tri_id].reshape(1, 2))
            & (Q <= tri_bbox_max[tri_id].reshape(1, 2)),
            axis=1,
        )
        if not np.any(bbox_mask):
            continue

        rem_sel = remaining[np.where(bbox_mask)[0]]
        inside = _point_in_tri_2d_batch(
            P_local[rem_sel],
            tri2d[tri_id, 0],
            tri2d[tri_id, 1],
            tri2d[tri_id, 2],
        )
        if np.any(inside):
            hit[rem_sel[inside]] = True

    return float(hit.sum()) / float(len(P2))


def projection_overlap_ratio(
    mesh: trimesh.Trimesh,
    patch_i: dict,
    patch_j: dict,
    samples_per_face,
) -> float:
    """
    Compatibility wrapper. For repeated evaluations, use the cache-based path.
    """
    p_i = dict(patch_i)
    p_j = dict(patch_j)
    p_i.setdefault("id", -1)
    p_j.setdefault("id", -2)
    cache = _prepare_patch_overlap_cache(mesh, [p_i, p_j], samples_per_face)
    if -1 not in cache or -2 not in cache:
        return 0.0
    return _projection_overlap_ratio_cached(cache[-1], cache[-2])


def score_patch_pair(
    patch_a: PlanarPatch,
    patch_b: PlanarPatch,
    params: PatchPairParams,
    mesh_for_overlap: trimesh.Trimesh,
    com_mesh: np.ndarray,
    bbox_diag: float,
    overlap_cache: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    precomputed_terms: Optional[Dict[str, float]] = None,
) -> PatchPairCandidate:
    n_a = unit(np.asarray(patch_a.normal, float))
    n_b = unit(np.asarray(patch_b.normal, float))

    c_a = np.asarray(patch_a.centroid, float)
    c_b = np.asarray(patch_b.centroid, float)

    if precomputed_terms is not None:
        width = float(precomputed_terms["width"])
        s_distance = float(precomputed_terms["distance"])
        s_inertia = float(precomputed_terms["inertia"])
        s_area = float(precomputed_terms["area"])
        score = float(precomputed_terms["score"])
    else:
        width = float(np.linalg.norm(c_a - c_b))
        if width <= 1e-12:
            width = 1e-12
        dist_a = point_line_distance(c_a, -n_a, c_b)
        dist_b = point_line_distance(c_b, -n_b, c_a)
        s_distance = 1.0 - (dist_a + dist_b) / max(float(bbox_diag), 1e-9)
        direction_a = (c_b - c_a) / width
        direction_b = -direction_a
        dist_COM_a = point_line_distance(c_a, direction_a, com_mesh)
        dist_COM_b = point_line_distance(c_b, direction_b, com_mesh)
        s_inertia = 1.0 - max(dist_COM_a, dist_COM_b) / max(float(bbox_diag), 1e-9)
        if patch_a.area > 0.0 and patch_b.area > 0.0:
            s_area = min(patch_a.area / patch_b.area, patch_b.area / patch_a.area)
        else:
            s_area = 0.0
        score = (
            params.distance_weight * s_distance
            + params.inertia_weight * s_inertia
            + params.area_weight * s_area
        )

    if overlap_cache is not None:
        src_a = overlap_cache.get(int(patch_a.id))
        src_b = overlap_cache.get(int(patch_b.id))
        if src_a is None or src_b is None:
            s_overlap = 0.0
        else:
            overlap_a = _projection_overlap_ratio_cached(src_a, src_b)
            overlap_b = _projection_overlap_ratio_cached(src_b, src_a)
            s_overlap = overlap_a * overlap_b
    else:
        overlap_a = projection_overlap_ratio(mesh_for_overlap, asdict(patch_a), asdict(patch_b), params.samples_per_face)
        overlap_b = projection_overlap_ratio(mesh_for_overlap, asdict(patch_b), asdict(patch_a), params.samples_per_face)
        s_overlap = overlap_a * overlap_b

    return PatchPairCandidate(
        patch_i=patch_a.id,
        patch_j=patch_b.id,
        normal=n_a.tolist(),
        width=width,
        score=score,
        terms={
            "distance": float(s_distance),
            "inertia": float(s_inertia),
            "area": float(s_area),
            "overlap": float(s_overlap),
        },
    )


def compute_best_patch_pairs(
    mesh_path: str,
    mesh_max_triangles: int = 1000,
    angle_deg: float = 7.0,
    min_opening: float = 1.0,
    max_opening: float = 140.0,
    angle_tolerance_deg: float = 10.0,
    top_k: int = 100,
    prefilter_multiplier: int = 8,
    min_prefilter_pool: int = 256,
    min_area_ratio_prefilter: float = 0.03,
    max_faces_after_split: int = 20000,
) -> Dict[str, Any]:
    mesh_filter, mesh_quad = load_uniform_mesh_with_open3d(mesh_path, target_triangles=mesh_max_triangles)

    mesh_COM = mesh_quad.center_mass
    mesh_bound = float(np.linalg.norm(mesh_quad.bounds[1] - mesh_quad.bounds[0]))
    remesh = split_long_edges(mesh_filter, max_faces_after_split=max_faces_after_split)

    patches = extract_planar_patches(remesh, angle_deg=angle_deg)
    patches = orient_patch_normals(mesh_quad, remesh, patches)

    params = PatchPairParams(
        min_opening=min_opening,
        max_opening=max_opening,
        angle_tolerance_deg=angle_tolerance_deg,
    )

    cands: List[PatchPairCandidate] = []
    if len(patches) < 2:
        return {
            "mesh_quad": mesh_quad,
            "mesh_patches": remesh,
            "num_patches": len(patches),
            "num_candidates": 0,
            "params": asdict(params),
            "patches": [asdict(p) for p in patches],
            "best": None,
            "top_k": [],
        }

    normals = np.asarray([unit(np.asarray(p.normal, float)) for p in patches], dtype=float)
    centroids = np.asarray([np.asarray(p.centroid, float) for p in patches], dtype=float)
    areas = np.asarray([float(p.area) for p in patches], dtype=float)
    n_patch = len(patches)

    dvec = centroids[None, :, :] - centroids[:, None, :]
    width = np.linalg.norm(dvec, axis=2)
    safe_width = np.where(width > 1e-12, width, 1.0)
    unit_d = dvec / safe_width[:, :, None]

    upper_mask = np.triu(np.ones((n_patch, n_patch), dtype=bool), k=1)
    cos_tol = math.cos(math.radians(params.angle_tolerance_deg))
    opposite_mask = (normals @ normals.T) <= (-cos_tol)

    dot_i = np.einsum("ijk,ik->ij", unit_d, normals)
    dot_j = np.einsum("ijk,jk->ij", unit_d, normals)
    facing_mask = (dot_i <= 0.0) & (dot_j >= 0.0)
    width_mask = (width >= params.min_opening) & (width <= params.max_opening)

    area_denom_i = np.maximum(areas[:, None], 1e-12)
    area_denom_j = np.maximum(areas[None, :], 1e-12)
    area_ratio = np.minimum(areas[:, None] / area_denom_j, areas[None, :] / area_denom_i)
    area_mask = area_ratio >= float(min_area_ratio_prefilter)

    ni = normals[:, None, :]
    nj = normals[None, :, :]
    t_a = np.sum(dvec * (-ni), axis=2)
    dist_a = np.where(
        t_a >= 0.0,
        np.linalg.norm(dvec - t_a[:, :, None] * (-ni), axis=2),
        width,
    )

    neg_dvec = -dvec
    t_b = np.sum(neg_dvec * (-nj), axis=2)
    dist_b = np.where(
        t_b >= 0.0,
        np.linalg.norm(neg_dvec - t_b[:, :, None] * (-nj), axis=2),
        width,
    )
    s_distance = 1.0 - (dist_a + dist_b) / max(mesh_bound, 1e-9)

    com_ref = np.asarray(mesh_COM, float).reshape(1, 1, 3)
    ca_to_com = com_ref - centroids[:, None, :]
    cb_to_com = com_ref - centroids[None, :, :]

    dir_a = unit_d
    dir_b = -unit_d

    t_com_a = np.sum(ca_to_com * dir_a, axis=2)
    dist_com_a = np.where(
        t_com_a >= 0.0,
        np.linalg.norm(ca_to_com - t_com_a[:, :, None] * dir_a, axis=2),
        np.linalg.norm(ca_to_com, axis=2),
    )
    t_com_b = np.sum(cb_to_com * dir_b, axis=2)
    dist_com_b = np.where(
        t_com_b >= 0.0,
        np.linalg.norm(cb_to_com - t_com_b[:, :, None] * dir_b, axis=2),
        np.linalg.norm(cb_to_com, axis=2),
    )

    s_inertia = 1.0 - np.maximum(dist_com_a, dist_com_b) / max(mesh_bound, 1e-9)
    quick_score = (
        params.distance_weight * s_distance
        + params.inertia_weight * s_inertia
        + params.area_weight * area_ratio
    )

    valid_mask = upper_mask & opposite_mask & facing_mask & width_mask & area_mask & (quick_score > 0.0)
    pair_idx = np.argwhere(valid_mask)

    if len(pair_idx) == 0:
        return {
            "mesh_quad": mesh_quad,
            "mesh_patches": remesh,
            "num_patches": len(patches),
            "num_candidates": 0,
            "params": asdict(params),
            "patches": [asdict(p) for p in patches],
            "best": None,
            "top_k": [],
        }

    pair_scores = quick_score[pair_idx[:, 0], pair_idx[:, 1]]
    pool_size = max(int(top_k * max(prefilter_multiplier, 1)), int(min_prefilter_pool))
    if len(pair_idx) > pool_size:
        sel = np.argpartition(-pair_scores, pool_size - 1)[:pool_size]
        pair_idx = pair_idx[sel]
        pair_scores = pair_scores[sel]

    order = np.argsort(-pair_scores)
    pair_idx = pair_idx[order]

    overlap_cache = _prepare_patch_overlap_cache(remesh, patches, params.samples_per_face)

    sd_vals = s_distance[pair_idx[:, 0], pair_idx[:, 1]]
    si_vals = s_inertia[pair_idx[:, 0], pair_idx[:, 1]]
    sa_vals = area_ratio[pair_idx[:, 0], pair_idx[:, 1]]
    wd_vals = width[pair_idx[:, 0], pair_idx[:, 1]]
    qs_vals = quick_score[pair_idx[:, 0], pair_idx[:, 1]]

    for k in range(len(pair_idx)):
        i = int(pair_idx[k, 0])
        j = int(pair_idx[k, 1])

        cand = score_patch_pair(
            patches[i],
            patches[j],
            params,
            remesh,
            mesh_COM,
            mesh_bound,
            overlap_cache=overlap_cache,
            precomputed_terms={
                "distance": float(sd_vals[k]),
                "inertia": float(si_vals[k]),
                "area": float(sa_vals[k]),
                "width": float(wd_vals[k]),
                "score": float(qs_vals[k]),
            },
        )

        if cand.terms["overlap"] <= 0.0:
            continue
        if cand.score <= 0.0:
            continue
        cands.append(cand)

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
    }


def _precompute_face_centroids(mesh: trimesh.Trimesh) -> np.ndarray:
    if len(mesh.faces) == 0:
        return np.zeros((0, 3), dtype=float)
    return mesh.vertices[mesh.faces].mean(axis=1)


def _best_face_matches(
    face_centroids: np.ndarray,
    face_i_idx: np.ndarray,
    face_j_idx: np.ndarray,
    n_i: np.ndarray,
    n_j: np.ndarray,
    max_face_trials_per_pair: int,
) -> List[Tuple[int, int, np.ndarray, np.ndarray]]:
    if len(face_i_idx) == 0 or len(face_j_idx) == 0:
        return []

    ci_all = face_centroids[face_i_idx]
    cj_all = face_centroids[face_j_idx]
    d = cj_all[None, :, :] - ci_all[:, None, :]
    nd = np.linalg.norm(d, axis=2)
    valid = nd > 1e-12

    d_hat = np.zeros_like(d)
    d_hat[valid] = d[valid] / nd[valid, None]
    score = np.abs(np.sum(d_hat * (-n_i).reshape(1, 1, 3), axis=2)) * np.abs(
        np.sum((-d_hat) * n_j.reshape(1, 1, 3), axis=2)
    )
    score[~valid] = -1.0

    best_j_local = np.argmax(score, axis=1)
    best_score = score[np.arange(len(face_i_idx)), best_j_local]
    order = np.argsort(-best_score)
    if max_face_trials_per_pair > 0:
        order = order[:max_face_trials_per_pair]

    out: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
    for li in order:
        if best_score[li] <= 0.0:
            continue
        fi = int(face_i_idx[li])
        lj = int(best_j_local[li])
        fj = int(face_j_idx[lj])
        out.append((fi, fj, ci_all[li], cj_all[lj]))
    return out


def _collision_broadphase_reject(
    geom: trimesh.Trimesh,
    mesh_bounds: np.ndarray,
    mesh_center: np.ndarray,
    mesh_radius: float,
) -> bool:
    gmin, gmax = geom.bounds
    if np.any(gmax < mesh_bounds[0]) or np.any(gmin > mesh_bounds[1]):
        return True
    gcenter = 0.5 * (gmin + gmax)
    gradius = 0.5 * float(np.linalg.norm(gmax - gmin))
    return float(np.linalg.norm(gcenter - mesh_center)) > float(mesh_radius + gradius)


def check_gripper_feasibility_faces_with_yaw(
    result: dict,
    pad_w: float = PadParams.pad_w,
    pad_h: float = PadParams.pad_h,
    pad_d: float = PadParams.pad_d,
    clearance_out: float = 10.0,
    yaw_grid_deg = [0, 90],
    use_mesh: str = "mesh_patches",
    check_mesh: str = "mesh_quad",
    max_face_trials_per_pair: int = 40,
    max_feasible_per_pair: int = 4,
    use_broadphase: bool = True,
):
    mesh = result[use_mesh]
    mesh_ch = result[check_mesh]
    patches = {p["id"]: p for p in result["patches"]}
    pairs = result.get("top_k", [])
    face_centroids = _precompute_face_centroids(mesh)
    patch_faces = {pid: np.asarray(p["face_indices"], dtype=np.int64) for pid, p in patches.items()}

    try:
        com = mesh_ch.center_mass
    except Exception:
        com = 0.5 * (mesh_ch.bounds[0] + mesh_ch.bounds[1])

    cm = trimesh.collision.CollisionManager()
    cm.add_object("part", mesh_ch)

    mesh_bounds = mesh_ch.bounds
    mesh_center = 0.5 * (mesh_bounds[0] + mesh_bounds[1])
    mesh_radius = 0.5 * float(np.linalg.norm(mesh_bounds[1] - mesh_bounds[0]))

    reports = []
    dist_criteria = float(min(pad_w, pad_h) / 5.0)

    for k, cand in enumerate(pairs):
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in patches or pid_j not in patches:
            reports.append(dict(pair_index=k, feasible=False))
            continue

        p_i = patches[pid_i]
        p_j = patches[pid_j]
        n_i = unit(np.asarray(p_i["normal"], float))
        n_j = unit(np.asarray(p_j["normal"], float))

        reports_for_this_pair = []
        matches = _best_face_matches(
            face_centroids,
            patch_faces[pid_i],
            patch_faces[pid_j],
            n_i,
            n_j,
            max_face_trials_per_pair=max_face_trials_per_pair,
        )

        for f_i, best_j, ci, best_cj in matches:
            alignment_dist_i = point_line_distance(ci, -n_i, best_cj)
            alignment_dist_j = point_line_distance(best_cj, -n_j, ci)
            if alignment_dist_i > dist_criteria or alignment_dist_j > dist_criteria:
                continue

            for yaw in yaw_grid_deg:
                box_i = make_rot_pad_box_at_patch(
                    {"centroid": ci, "normal": n_i},
                    pad_w,
                    pad_h,
                    pad_d,
                    clearance_out,
                    yaw_deg=yaw,
                )
                box_j = make_rot_pad_box_at_patch(
                    {"centroid": best_cj, "normal": n_j},
                    pad_w,
                    pad_h,
                    pad_d,
                    clearance_out,
                    yaw_deg=-yaw,
                )

                if use_broadphase and _collision_broadphase_reject(box_i, mesh_bounds, mesh_center, mesh_radius):
                    coll_i = False
                else:
                    coll_i = cm.in_collision_single(box_i)
                if coll_i:
                    continue

                if use_broadphase and _collision_broadphase_reject(box_j, mesh_bounds, mesh_center, mesh_radius):
                    coll_j = False
                else:
                    coll_j = cm.in_collision_single(box_j)
                if coll_j:
                    continue

                midpoint = 0.5 * (ci + best_cj)
                current_dist = float(np.linalg.norm(midpoint - com))
                Fi = -n_i
                Fj = -n_j
                tau = np.cross(ci - com, Fi) + np.cross(best_cj - com, Fj)
                current_moment = float(np.linalg.norm(tau))

                reports_for_this_pair.append(
                    dict(
                        pair_index=k,
                        patch_i=pid_i,
                        patch_j=pid_j,
                        face_i=f_i,
                        face_j=best_j,
                        feasible=True,
                        feasible_yaw=yaw,
                        moment=current_moment,
                        dist=current_dist,
                    )
                )

                if len(reports_for_this_pair) >= max_feasible_per_pair:
                    break
            if len(reports_for_this_pair) >= max_feasible_per_pair:
                break

        if not reports_for_this_pair:
            reports.append(dict(pair_index=k, patch_i=pid_i, patch_j=pid_j, feasible=False))
        else:
            reports.extend(reports_for_this_pair)

    reports_sorted = sorted(
        [r for r in reports if r.get("feasible")],
        key=lambda r: (-r.get("dist", float("inf")), -r.get("moment", float("inf"))),
    )
    return reports_sorted


def check_gripper_feasibility_faces_with_rotation(
    result: dict,
    pad_w: float = PadParams.pad_w,
    pad_h: float = PadParams.pad_h,
    pad_d: float = PadParams.pad_d,
    clearance_out: float = 10.0,
    use_mesh: str = "mesh_patches",
    check_mesh: str = "mesh_quad",
    max_face_trials_per_pair: int = 40,
    max_feasible_per_pair: int = 4,
    use_broadphase: bool = True,
):
    mesh = result[use_mesh]
    mesh_ch = result[check_mesh]
    patches = {p["id"]: p for p in result["patches"]}
    pairs = result.get("top_k", [])
    face_centroids = _precompute_face_centroids(mesh)
    patch_faces = {pid: np.asarray(p["face_indices"], dtype=np.int64) for pid, p in patches.items()}

    try:
        com = mesh_ch.center_mass
    except Exception:
        com = 0.5 * (mesh_ch.bounds[0] + mesh_ch.bounds[1])

    cm = trimesh.collision.CollisionManager()
    cm.add_object("part", mesh_ch)

    mesh_bounds = mesh_ch.bounds
    mesh_center = 0.5 * (mesh_bounds[0] + mesh_bounds[1])
    mesh_radius = 0.5 * float(np.linalg.norm(mesh_bounds[1] - mesh_bounds[0]))

    reports = []
    dist_criteria = float(min(pad_w, pad_h) / 5.0)

    for k, cand in enumerate(pairs):
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in patches or pid_j not in patches:
            reports.append(dict(pair_index=k, feasible=False))
            continue

        p_i = patches[pid_i]
        p_j = patches[pid_j]
        n_i = unit(np.asarray(p_i["normal"], float))
        n_j = unit(np.asarray(p_j["normal"], float))

        feasible_for_pair = 0
        matches = _best_face_matches(
            face_centroids,
            patch_faces[pid_i],
            patch_faces[pid_j],
            n_i,
            n_j,
            max_face_trials_per_pair=max_face_trials_per_pair,
        )

        for f_i, best_j, ci, best_cj in matches:
            alignment_dist_i = point_line_distance(ci, -n_i, best_cj)
            alignment_dist_j = point_line_distance(best_cj, -n_j, ci)
            if alignment_dist_i > dist_criteria or alignment_dist_j > dist_criteria:
                continue

            cyl_i = make_pad_cylinder_at_patch({"centroid": ci, "normal": n_i}, pad_w / 2, pad_h / 2, pad_d, clearance_out)
            cyl_j = make_pad_cylinder_at_patch({"centroid": best_cj, "normal": n_j}, pad_w / 2, pad_h / 2, pad_d, clearance_out)

            if use_broadphase and _collision_broadphase_reject(cyl_i, mesh_bounds, mesh_center, mesh_radius):
                coll_i = False
            else:
                coll_i = cm.in_collision_single(cyl_i)
            if coll_i:
                continue

            if use_broadphase and _collision_broadphase_reject(cyl_j, mesh_bounds, mesh_center, mesh_radius):
                coll_j = False
            else:
                coll_j = cm.in_collision_single(cyl_j)
            if coll_j:
                continue

            midpoint = 0.5 * (ci + best_cj)
            current_dist = float(np.linalg.norm(midpoint - com))
            Fi = -n_i
            Fj = -n_j
            tau = np.cross(ci - com, Fi) + np.cross(best_cj - com, Fj)
            current_moment = float(np.linalg.norm(tau))

            reports.append(
                dict(
                    pair_index=k,
                    patch_i=pid_i,
                    patch_j=pid_j,
                    face_i=f_i,
                    face_j=best_j,
                    dist=current_dist,
                    moment=current_moment,
                    feasible=True,
                )
            )

            feasible_for_pair += 1
            if feasible_for_pair >= max_feasible_per_pair:
                break

    reports_sorted = sorted(
        [r for r in reports if r.get("feasible")],
        key=lambda r: (r.get("dist", float("inf")), r.get("moment", float("inf"))),
    )
    return reports_sorted


# --------------------------
# Grasping metric: Epsilon-Quality metric 
# --------------------------

# --------------------------
# TODO: 점 접촉 Epsilon-Quality metric: 제거 예정 
# --------------------------
def get_friction_cone_vectors(normal: np.ndarray, mu: float = PadParams.mu, k: int = 6) -> List[np.ndarray]:
    """
    주어진 법선 벡터(inward normal)와 마찰 계수(mu)에 대해,
    마찰 콘(friction cone)을 근사하는 k개의 단위 힘 벡터(primitive force vectors)를 생성
    
    Args:
        normal (np.ndarray): 3D inward normal vector (그리퍼가 물체를 누르는 방향).
        mu (float): 마찰 계수 (e.g., 0.5). mu = tan(alpha).
        k (int): 근사 벡터의 개수 (e.g., 4 또는 8).

    Returns:
        List[np.ndarray]: k개의 단위 힘 벡터 리스트.
    """
    u, v, n = basis_from_normal(normal)
    
    vectors = []
    for i in range(k):
        angle = 2.0 * np.pi * i / k
        
        # 1. 접선 벡터(tangent vector)를 생성합니다.
        # 이 벡터는 마찰력의 방향을 나타냅니다.
        tangent_vec = u * np.cos(angle) + v * np.sin(angle)
        
        # 2. 법선 벡터와 마찰 벡터를 결합합니다.
        # f = n + mu * tangent_vec
        # 이 힘 벡터 f는 마찰 콘의 경계(edge)를 따라 형성됩니다.
        f = n + mu * tangent_vec
        
        # 3. 단위 벡터로 정규화하여 'primitive force'를 만듭니다.
        vectors.append(unit(f))
        
    return vectors

def calculate_epsilon_quality(
    c_i: np.ndarray, 
    n_i: np.ndarray, 
    c_j: np.ndarray, 
    n_j: np.ndarray, 
    com: np.ndarray, 
    mu: float = PadParams.mu, 
    k: int = 16
) -> float:
    """
    두 접촉점(i, j)에 대한 6D Grasp Wrench Space (GWS)의 
    epsilon-quality (force closure) 계산
    Args:
        c_i, c_j (np.ndarray): 3D 접촉점 (e.g., face centroids).
        n_i, n_j (np.ndarray): 3D *outward* surface normals.
        com (np.ndarray): 객체의 3D Center of Mass.
        mu (float): 마찰 계수 (coefficient of friction).
        k (int): 마찰 콘 근사를 위한 벡터 수.

    Returns:
        float: Epsilon-quality (epsilon). 0.0 이면 force closure 실패.
    """
    
    # 1. 법선 벡터를 'inward normal' (그리퍼가 누르는 방향)으로 뒤집습니다.
    inward_n_i = -unit(np.asarray(n_i))
    inward_n_j = -unit(np.asarray(n_j))
    
    # 2. 각 접촉점에서 마찰 콘을 근사하는 k개의 단위 힘 벡터(primitive forces)를 얻습니다.
    cone_vecs_i = get_friction_cone_vectors(inward_n_i, mu, k)
    cone_vecs_j = get_friction_cone_vectors(inward_n_j, mu, k)
    
    # 3. 각 힘 벡터가 COM(Center of Mass) 기준으로 생성하는 Wrench(f, tau)를 계산합니다.
    r_i = np.asarray(c_i) - np.asarray(com) # COM에서 접촉점 i까지의 벡터
    r_j = np.asarray(c_j) - np.asarray(com) # COM에서 접촉점 j까지의 벡터
    
    wrenches = []
    # 접촉점 i에 대한 렌치
    for f_i in cone_vecs_i:
        tau_i = np.cross(r_i, f_i)
        wrenches.append(np.concatenate([f_i, tau_i]))
        
    # 접촉점 j에 대한 렌치
    for f_j in cone_vecs_j:
        tau_j = np.cross(r_j, f_j)
        wrenches.append(np.concatenate([f_j, tau_j]))
        
    wrenches = np.array(wrenches)
    
    # 4. 모든 렌치 벡터를 사용하여 6D Convex Hull (GWS)을 계산합니다.
    if wrenches.shape[0] <= 6:
        return 0.0  # 6D Hull을 만들기에 점이 부족합니다.

    try:
        hull = ConvexHull(wrenches, qhull_options="QJ")
    except Exception as e:
        # print(f"Convex Hull 6D Error: {e}")
        return 0.0  # 점들이 6D 공간을 채우지 못하고 저차원 평면에 존재 (e.g., 2D 물체)

    # 5. Epsilon-quality 계산: 원점(0,0,0,0,0,0)에서 GWS의 가장 가까운 경계면(hyperplane)까지의 거리
    min_dist = float('inf')
    
    # hull.equations는 [A, b] 형태이며, Ax + b >= 0 이 Hull의 내부를 정의합니다.
    # (scipy는 법선(A)이 Hull 외부를 가리키므로, 원점이 내부(Ax+b >= 0)에 있으려면 b >= 0 이어야 합니다)
    for eq in hull.equations:
        A, b = eq[:-1], eq[-1]
        
        # 원점이 Hull 내부에 있는지 확인 (b < 0 이면 원점이 외부에 있음)
        if b < -1e-9: # 수치적 안정성을 위해 작은 음수 허용
            return 0.0  # Force closure 실패 (Epsilon = 0)
            
        # 원점에서 평면까지의 거리: |A*x + b| / ||A||, (x=0 이므로 |b| / ||A||)
        dist = np.abs(b) / (np.linalg.norm(A) + 1e-9)
        
        if dist < min_dist:
            min_dist = dist
            
    # 원점이 내부에 있고(b >= 0), 가장 가까운 평면까지의 거리가 epsilon 값입니다.
    return min_dist
# --------------------------
# TODO: 점 접촉 Epsilon-Quality metric: 제거 예정 
# --------------------------


# Friction Cone Wrench 계산
def add_wrenches(points, normal, com, ch_len, mu=PadParams.mu, k=8):
    # Tangent, Bitangent
    t, b, _ = basis_from_normal(normal)
    angle_step = 2 * np.pi / k
    wrenches = []
    for pt in points:
        r = pt - com # 모멘트 암
        for i in range(k):
            theta = i * angle_step
            # Force direction (Normal + Friction)
            f = normal + mu * np.cos(theta) * t + mu * np.sin(theta) * b
            f = f / np.linalg.norm(f) # 스케일링
            # Torque
            tau = np.cross(r, f)
            # Wrench
            # w = np.concatenate([f, tau / ch_len]) # 토크를 Characteristic Length로 힘/토크 스케일 보정
            w = np.concatenate([f, tau])
            wrenches.append(w)
    
    return np.array(wrenches)

# Pad 위치에 따른 Contact Points 찾기
def get_points_in_squeezed_pad(mesh, patch_info, 
                               pad_w, pad_h, pad_d, 
                               yaw_deg=0.0, 
                               squeeze_depth=0.1, 
                            #    sample_count=100 # sample_surface 사용시
    ):
    """
    패드를 물체 안쪽으로 squeeze_depth만큼 밀어넣었을 때,
    패드 영역 안에 포함되는 물체 표면의 점들을 반환
    """
    # 1. 물체 표면에서 점 샘플링
    # samples, _ = trimesh.sample.sample_surface(mesh, sample_count) # submesh 내부 점 샘플링
    samples = mesh.vertices # submesh의 삼각형 점 사용

    if len(samples) == 0:
        return np.array([]), None   

    # 2. 패드 박스(OBB) 생성
    pad_box = trimesh.creation.box(extents=[pad_w, pad_h, pad_d])
    
    # 패치 정보
    c = np.asarray(patch_info["centroid"])
    n = unit(np.asarray(patch_info["normal"]))
    
    # 3. 패드 회전 및 이동 (Pose 변환)
    # R_align = trimesh.geometry.align_vectors([0, 0, 1], n) # Normal 정렬 회전 (Z축 -> Normal)
    u, v, n_vec = basis_from_normal(n)
    R_align = np.eye(4)
    R_align[:3, :3] = np.column_stack([u, v, n_vec]) # [u, v, n] 순서로 회전 행렬 구성

    rad = np.deg2rad(yaw_deg) # Yaw 회전 (Normal 축 기준)
    R_yaw = trimesh.transformations.rotation_matrix(rad, [0, 0, 1])

    # 3-3. 이동 (Translation): 패드 접촉 면이 물체 표면에서 squeeze_depth 만큼 들어가게 함
    offset_dist = (pad_d / 2.0) - squeeze_depth # 패드 중심점 오프셋
    t_vec = c + n * offset_dist # 
    
    T = np.eye(4)
    T[:3, 3] = t_vec
    
    # 최종 변환 행렬: T @ R_align @ R_yaw
    final_pose = T @ R_align @ R_yaw
    pad_box.apply_transform(final_pose)
    
    # 4. 포함 여부 검사 (Points inside Box): samples 점들이 pad_box 안에 있는지 확인
    is_inside = pad_box.contains(samples)
    contact_points = samples[is_inside]
    
    return contact_points, pad_box # 시각화를 위해 pad_box도 반환

def calculate_squeeze_epsilon_quality(mesh, f_i_idx, f_j_idx, c_i, n_i, yaw_i, c_j, n_j, yaw_j, com,
                                      pad_w=PadParams.pad_w,
                                      pad_h=PadParams.pad_h,
                                      pad_d=PadParams.pad_d,
                                      squeeze_depth=0.1, # mm 단위 접촉 면 깊이
    # 해당 패치 근처의 face들만 서브셋으로 뽑아서 샘플링하면 더 빠름 (여기서는 전체 메쉬 사용 예시)
    # 최적화를 위해 mesh.submesh([patch_face_indices]) 사용 권장
                                      ):
    """
    양쪽 패드를 물체 쪽으로 squeeze_depth 만큼 침투시켰을 때
    겹치는 영역의 점들을 사용하여 GWS 및 Epsilon Quality를 계산
    """

    # 1. 접촉점 추출 (Points containment)
    # Patch I에 대한 서브메쉬
    mesh_i = mesh.submesh([f_i_idx], append=True)
    pts_i, _ = get_points_in_squeezed_pad(mesh_i, {"centroid": c_i, "normal": n_i}, 
                                          pad_w, pad_h, pad_d, yaw_i, squeeze_depth)    
    # Patch J에 대한 서브메쉬
    mesh_j = mesh.submesh([f_j_idx], append=True)
    pts_j, _ = get_points_in_squeezed_pad(mesh_j, {"centroid": c_j, "normal": n_j}, 
                                          pad_w, pad_h, pad_d, yaw_j, squeeze_depth)

    # 접촉점 검사
    if len(pts_i) == 0 or len(pts_j) == 0: # 접촉점이 없는 경우 Grasping 없음
        return 0.0

    # 힘은 물체를 미는 방향이므로 -normal
    ch_len = 0.5 * float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])) # RWS Characteristic Length = Radius 스케일링

    wrenches_i = add_wrenches(pts_i, -n_i, com, ch_len)
    wrenches_j = add_wrenches(pts_j, -n_j, com, ch_len)
    wrenches = np.concatenate([wrenches_i, wrenches_j], axis=0)
    
    # 2. Convex Hull (GWS)
    try:
        if len(wrenches) < 4: return 0.0 # Contact Point 갯수가 너무 적으면 계산 안 됨
        hull = ConvexHull(wrenches, qhull_options='QJ') # QJ: Joggled input (에러 방지)
    except Exception as e:
        # print(f"ConvexHull Error: {e}")
        return 0.0

    # 3. Epsilon Quality (원점에서 가장 가까운 면까지 거리)
    min_dist = float('inf')
    for eq in hull.equations:
        # eq: [nx, ny, nz, nw, nt, nr, offset]
        # distance from origin = abs(offset) / norm(normal)
        # 원점이 내부: normal @ origin + offset <= 0 (offset <= 0)
        dist = -eq[-1] 
        if dist < min_dist:
            min_dist = dist
            
    if min_dist < 0:
        return 0.0

    return min_dist


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
    # 그리퍼의 접근 방향이 카메라 좌표계의 위쪽(-Z)을 향하면, z 축을 뒤집어 아래(+Z)를 향하도록 함
    # 카메라 좌표계: Z 정면, Y 아래, X 오른쪽
    z_axis_in_camera = H_OC[:3, :3] @ z_axis
    if z_axis_in_camera[2] < 0: 
        z_axis = - z_axis  # pad w 방향
        y_axis = - y_axis  # pad h 방향
        x_axis = np.cross(y_axis, z_axis)  # closing 방향
    # 로봇 6축의 불필요한 회전 방지를 위한 코드
    x_axis_in_camera = H_OC[:3, :3] @ x_axis
    y_axis_in_camera = H_OC[:3, :3] @ y_axis
    if x_axis_in_camera[0] < 0 and y_axis_in_camera[1] < 0: 
        x_axis = - x_axis
        y_axis = - y_axis

    # 직교 보정 (x,y,z 순서)
    R = np.column_stack([x_axis, y_axis, z_axis])
    
    # 그리퍼 스트로크에 따른 위치 반영
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
    # 좌표계 시각화
    # H_E = np.eye(4,4) # TODO: 로봇 컨트롤러 신호 받아 변환행렬 만들기 (현재는 EE 좌표계 기준이라 개발 필요 X)

    # 고정변환
    H_GnEn = to44(Rotx(180) @ Rotz(90), [0,0,-135])   # EE -> Grip [0,0,-135] <- 실측
    H_GnCn = to44(np.eye(3), [30,58,10])           # Cam -> Grip [18,45,10] <- 실측, [30,58,10] <- 실험
    H_CnGn = np.linalg.inv(H_GnCn)
    H_CnEn = H_GnEn @ H_CnGn                        # EE -> Cam
    H_EG = np.linalg.inv(H_GnEn)

    H_OdCn = H_OC
    H_OdEn = H_CnEn @ H_OdCn
    # H_GdOd = np.linalg.inv(H_OG)
    H_GdEn = H_OdEn @ H_OG
    H_EdEn = H_GdEn @ H_EG
    return H_EdEn, H_OdEn

