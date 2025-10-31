import numpy as np
import plotly.graph_objects as go
import trimesh
from dataclasses import dataclass
from scipy.spatial import ConvexHull
from plotly.subplots import make_subplots

# -------------------------------
# helpers
# -------------------------------
@dataclass
class PadParams:
    pad_w: float = 34.0 # 34
    pad_h: float = 21.0 # 21
    pad_d: float = 7.0 # 7

def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def mesh3d_from_trimesh(m, face_indices=None, color="lightgray", opacity=0.3, name="mesh"):
    faces = m.faces if face_indices is None else m.faces[np.asarray(list(face_indices), dtype=int)]
    x, y, z = m.vertices.T
    i, j, k = faces.T
    return go.Mesh3d(
        x=x, y=y, z=z, i=i, j=j, k=k,
        color=color, opacity=opacity, flatshading=True,
        lighting=dict(ambient=0.45, diffuse=0.65, specular=0.2, roughness=0.9),
        lightposition=dict(x=0, y=0, z=1),
        name=name, showlegend=True
    )

def boundary_edges_trace(mesh, face_idx, color="#166534", width=1, name="boundary"):
    """patch 경계선(외곽 edge만) Scatter3d로 표현"""
    F = mesh.faces[np.asarray(list(face_idx), dtype=int)]
    # edge -> count
    edges = {}
    for a,b,c in F:
        for u,v in ((a,b),(b,c),(c,a)):
            e = (u,v) if u < v else (v,u)
            edges[e] = edges.get(e, 0) + 1
    # 경계(edge count==1)만 라인화
    lines = []
    for (u,v), c in edges.items():
        if c == 1:
            p1, p2 = mesh.vertices[u], mesh.vertices[v]
            lines.extend([p1, p2, [None, None, None]])  # 분리용 None
    if not lines:
        return None
    L = np.array(lines, dtype=float)
    return go.Scatter3d(
        x=L[:,0], y=L[:,1], z=L[:,2],
        mode="lines", line=dict(color=color, width=width),
        name=name, showlegend=False
    )

def hsv(i, n, s=0.6, v=0.95):
    import colorsys
    h = (i % n) / max(1, n)
    r,g,b = colorsys.hsv_to_rgb(h, s, v)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

def cones_for_normals(
    patches,
    mesh_bounds,
    normal_scale: float = 20.0,   # ↑ 키울수록 cone 커짐
    min_len_ratio: float = 0.1, # bbox 대각선 대비 최소 벡터 길이
    color: str = "#000000",
    name: str = "normals",
):
    """
    - 각 패치 길이 nlen = max(bbox_min_len, 0.5*sqrt(area))
    - sizeref = median(nlen) / normal_scale  (outlier 영향 완화)
    """
    (bmin, bmax) = mesh_bounds
    extent = float(np.linalg.norm(np.asarray(bmax) - np.asarray(bmin))) or 1.0
    bbox_min_len = max(min_len_ratio * extent, 1e-6)

    cx, cy, cz, ux, uy, uz, lens = [], [], [], [], [], [], []
    for p in patches:
        c = np.asarray(p["centroid"], float)
        n = np.asarray(p["normal"], float)
        n = n / (np.linalg.norm(n) + 1e-12)
        a = float(p.get("area", 0.0))
        nlen = max(bbox_min_len, 0.5 * np.sqrt(max(a, 0.0)))  # 글로벌+로컬
        lens.append(nlen)

        cx.append(c[0]); cy.append(c[1]); cz.append(c[2])
        ux.append(n[0]*nlen); uy.append(n[1]*nlen); uz.append(n[2]*nlen)

    med = float(np.median(lens)) if lens else bbox_min_len
    sizeref = max(med / max(normal_scale, 1e-6), 1e-9)

    return go.Cone(
        x=cx, y=cy, z=cz, u=ux, v=uy, w=uz,
        sizemode="raw", sizeref=sizeref,
        anchor="tail", colorscale=[[0, color], [1, color]],
        showscale=False, name=name, opacity=1.0,
    )

def basis_from_normal(n):
    n=unit(n); a=np.array([1,0,0]) if abs(n[0])<0.9 else np.array([0,1,0])
    u=unit(np.cross(n,a)); v=np.cross(n,u)
    return u,v,n

def rotate_byaxis(axis, ang_deg):
    axis = unit(axis)
    th = np.deg2rad(ang_deg)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], float)
    I = np.eye(3)
    return I + np.sin(th)*K + (1-np.cos(th))*(K@K)

def make_pad_box_at_patch(patch, pad_w, pad_h, pad_d):
    """
    패치 평면의 바깥쪽(+n)으로 'pad_d' 만큼 돌출된 OBB 생성
    + centroid에서 바깥쪽으로 lift_mm 만큼 추가 이격
    - extents: [pad_w, pad_h, pad_d]
    - center : 패치 중심 + n * (extents_z/2)
    - orientation: columns = [u, v, n]
    """
    n = unit(np.asarray(patch["normal"], float))
    u, v, n = basis_from_normal(n)
    c = np.asarray(patch["centroid"], float)

    ext_z = float(pad_d)
    box = trimesh.creation.box(extents=[float(pad_w), float(pad_h), float(pad_d)])

    T = np.eye(4)
    T[:3, :3] = np.column_stack([u, v, n])
    T[:3,  3] = c + n*0.5*ext_z
    box.apply_transform(T)

    return box

def make_pad_box_at_patch_with_yaw(patch, pad_w, pad_h, pad_d, yaw_deg):
    """
    패치 평면의 바깥쪽(+n)으로 'pad_d' 만큼 돌출된 OBB 생성
    + centroid에서 바깥쪽으로 lift_mm 만큼 추가 이격
    - extents: [pad_w, pad_h, pad_d]
    - center : 패치 중심 + n * (extents_z/2)
    - orientation: columns = [u, v, n]
    """
    n = unit(np.asarray(patch["normal"], float))
    u, v, n = basis_from_normal(n)
    c = np.asarray(patch["centroid"], float)

    ext_z = float(pad_d)
    box = trimesh.creation.box(extents=[float(pad_w), float(pad_h), float(pad_d)])

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
    T[:3,  3] = c + n_world*0.5*ext_z
    box.apply_transform(T)

    return box

def make_pad_cylinder_at_patch(patch, pad_w, pad_h, pad_d,
                               lift_mm: float = 2.0):
    """
    패치 중심에서 normal 방향으로 회전하는 패드+클리어런스의 swept volume을 근사하는 원기둥 메쉬를 생성
    Returns:
        trimesh.Trimesh: 원기둥 메쉬 객체.
    """
    n = unit(np.asarray(patch["normal"], float))
    c = np.asarray(patch["centroid"], float)

    height = float(pad_d)
    radius = 0.5 * np.sqrt(pad_w**2 + pad_h**2)

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


def frame_traces(H, name, scale=20.0, legendgroup=None, showlegend=False):
    """
    4x4 pose 행렬 H 기준 좌표축(x=red, y=green, z=blue) 시각화
    """
    o = H[:3, 3]
    x_axis = o + H[:3, 0] * scale
    y_axis = o + H[:3, 1] * scale
    z_axis = o + H[:3, 2] * scale
    traces = []
    traces.append(go.Scatter3d(x=[o[0],x_axis[0]], y=[o[1],x_axis[1]], z=[o[2],x_axis[2]],
                               mode="lines", line=dict(color="red", width=6),
                               name=f"{name}-x", legendgroup=legendgroup, showlegend=showlegend))
    traces.append(go.Scatter3d(x=[o[0],y_axis[0]], y=[o[1],y_axis[1]], z=[o[2],y_axis[2]],
                               mode="lines", line=dict(color="green", width=6),
                               name=f"{name}-y", legendgroup=legendgroup, showlegend=showlegend))
    traces.append(go.Scatter3d(x=[o[0],z_axis[0]], y=[o[1],z_axis[1]], z=[o[2],z_axis[2]],
                               mode="lines", line=dict(color="blue", width=6),
                               name=f"{name}-z", legendgroup=legendgroup, showlegend=showlegend))
    return traces

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

    # origin = 중점d
    o_pad = 0.5 * (ci + cj)

    # closing 방향 (x축)
    x_axis = unit(ni - nj) # make_box_rot 함수에서 pad_i normal yaw_deg 회전
    u, v, n = basis_from_normal(x_axis)
    B = np.column_stack([u, v, n])
    Rz = Rotz(yaw_deg)
    R_world = B @ Rz
    z_axis = R_world[:, 0]  # pad w 방향
    y_axis = - R_world[:, 1]  # pad h 방향
    x_axis = R_world[:, 2]  # closing 방향

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
    H_GO = to44(R, o)

    return H_GO, stroke

# -------------------------------
# 1) 추출된 패치 시각화
# -------------------------------
def visualize_merged_patches_plotly(result,
                                    show_silhouette=True,
                                    show=False):
    """
    result: grasping.compute_best_patch_pairs(...) 출력 dict
      - mesh_quad: 원본 실루엣용 Trimesh
      - mesh_patches: 패치가 추출된(간소화) Trimesh
      - patches: [{'id','face_indices','centroid','normal','b','area'}, ...]
    """
    mesh_sil = result["mesh_quad"]
    mesh_p   = result["mesh_patches"]
    patches  = result["patches"]

    # 범위/스케일
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh_sil.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin, ymax-ymin, zmax-zmin)

    # 색/스타일
    patch_fill_color  = "#a7f3d0"  # 옅은 초록
    edge_color        = "#166534"  # 진한 초록(테두리)
    normal_color      = "#666666"  # 검정

    traces = []
    if show_silhouette:
        traces.append(mesh3d_from_trimesh(mesh_sil, None, color="#cfcfcf", opacity=0.12, name="silhouette"))

    # 모든 패치: 동일 색 + 경계선 + 검정 normal
    for p in patches:
        fidx = set(p["face_indices"])
        traces.append(mesh3d_from_trimesh(mesh_p, fidx, color=patch_fill_color, opacity=0.6, name=f"patch {p['id']}"))
        edge = boundary_edges_trace(mesh_p, fidx, color=edge_color, width=2, name=f"edge {p['id']}")
        if edge is not None:
            traces.append(edge)

    # normals(검정)
    traces.append(cones_for_normals(patches, mesh_bounds=mesh_sil.bounds, normal_scale=50.0,
                                     min_len_ratio=0.1, color=normal_color, name="normals"))
    
    # COM 넣기
    com = mesh_sil.center_mass
    traces.append(go.Scatter3d(x=[com[0]],y=[com[1]],z=[com[2]],
                               mode='markers', marker=dict(size=10,color='red',symbol='diamond',opacity=0.9),
                               name='Center of Mass'))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Patches (count={len(patches)})",
        margin=dict(l=0, r=0, t=48, b=0),
        showlegend=True,
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig

# -------------------------------
# 2) patch pair(top_k) 시각화
# -------------------------------
def visualize_pairs_centroid_lines(result,
                                   show_silhouette=True,
                                   patch_opacity=0.3,
                                   show=False):
    """
    result['top_k']의 모든 pair를 표시:
      - pair마다 두 패치(같은 색, 옅게)
      - 두 패치의 centroid를 잇는 선
    """
    mesh_sil = result["mesh_quad"]
    mesh_p = result["mesh_patches"]
    patches = result["patches"]
    cands = result.get("top_k", [])
    if not cands:
        raise ValueError("result['top_k'] is empty.")

    # id -> patch
    pmap = {p["id"]: p for p in patches}

    # 범위
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh_p.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin,ymax-ymin,zmax-zmin)

    traces = []
    if show_silhouette:
        traces.append(mesh3d_from_trimesh(mesh_sil, None, color="#cfcfcf", opacity=0.12, name="silhouette"))

    for k, cand in enumerate(cands):
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in pmap or pid_j not in pmap:
            continue
        pi, pj = pmap[pid_i], pmap[pid_j]
        fi, fj = set(pi["face_indices"]), set(pj["face_indices"])
        col = hsv(k, len(cands), s=0.55, v=0.95)          # pair 공통색
        col_edge = hsv(k, len(cands), s=0.85, v=0.7)      # 경계 진한색

        # patches (같은 색, 옅게)
        traces.append(mesh3d_from_trimesh(mesh_p, fi, color=col, opacity=patch_opacity, name=f"pair{k}: patch {pid_i}"))
        traces.append(mesh3d_from_trimesh(mesh_p, fj, color=col, opacity=patch_opacity, name=f"pair{k}: patch {pid_j}"))

        # 경계선
        e1 = boundary_edges_trace(mesh_p, fi, color=col_edge, width=3, name=f"pair{k}: edge i")
        e2 = boundary_edges_trace(mesh_p, fj, color=col_edge, width=3, name=f"pair{k}: edge j")
        if e1 is not None: traces.append(e1)
        if e2 is not None: traces.append(e2)

        # centroid line (COM of patch surfaces ≈ face centroid 평균 사용)
        ci = np.asarray(pi["centroid"], float)
        cj = np.asarray(pj["centroid"], float)
        traces.append(go.Scatter3d(
            x=[ci[0], cj[0]], y=[ci[1], cj[1]], z=[ci[2], cj[2]],
            mode="lines+markers",
            line=dict(width=6, color=col_edge),
            marker=dict(size=3, color=col_edge),
            name=f"pair{k}: centroid line"
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Top-k Pairs (k={len(cands)}) — patches (light) + centroid lines",
        margin=dict(l=0, r=0, t=48, b=0),
        showlegend=True,
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig

# -------------------------------
# 3) feasible patch pair 시각화
# -------------------------------
# pad mesh 있는 버전
def visualize_feasible_pairs_pads(result, reports,
                                  show_silhouette=True,
                                  pad_w=PadParams.pad_w,
                                  pad_h=PadParams.pad_h,
                                  pad_d=PadParams.pad_d,
                                  show=False):
    mesh = result.get("mesh_quad")
    mesh_patches = result.get("mesh_patches")
    patches = {p["id"]: p for p in result["patches"]}
    pairs = result.get("top_k", [])

    ok_idx = [r["pair_index"] for r in reports if r.get("feasible")]
    # print(ok_idx)

    if not ok_idx:
        raise ValueError("feasible pair가 없습니다.")

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin,ymax-ymin,zmax-zmin)

    traces=[]
    if show_silhouette:
        traces.append(mesh3d_from_trimesh(mesh, color="#cfcfcf", opacity=0.5, name="mesh"))

    for k, idx in enumerate(ok_idx):
        if idx >= len(pairs): continue
        cand = pairs[idx]
        yaw_deg = reports[k]['feasible_yaw']
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in patches or pid_j not in patches: continue
        pi, pj = patches[pid_i], patches[pid_j]
        col = hsv(k, len(ok_idx), s=0.55, v=0.95)

        # pad boxes + rot
        fi, fj = reports[k]["face_i"], reports[k]["face_j"]
        ni = unit(np.asarray(pi["normal"], float))
        nj = unit(np.asarray(pj["normal"], float))
        ci = mesh_patches.vertices[mesh_patches.faces[fi]].mean(axis=0)
        cj = mesh_patches.vertices[mesh_patches.faces[fj]].mean(axis=0)
        box_i = make_pad_box_at_patch_with_yaw({"centroid": ci, "normal": ni}, pad_w, pad_h, pad_d, yaw_deg= yaw_deg)
        box_j = make_pad_box_at_patch_with_yaw({"centroid": cj, "normal": nj}, pad_w, pad_h, pad_d, yaw_deg=-yaw_deg)

        # legendgroup으로 두 pad를 하나의 토글 그룹에 묶기
        lg = f"pair {idx} - faces {fi, fj} - yaw {yaw_deg}"
        # proxy(범례 핸들) – 클릭 시 그룹 전체 토글
        traces.append(go.Scatter3d(
            x=[ci[0]], y=[ci[1]], z=[ci[2]],
            mode="markers", marker=dict(size=1, opacity=0.0),
            name=lg, legendgroup=lg, showlegend=True))
        # 그리퍼 패드 시각화
        # pad_i
        x,y,z = box_i.vertices.T; I,J,K = box_i.faces.T
        traces.append(go.Mesh3d(x=x,y=y,z=z,i=I,j=J,k=K,
                                color=col, opacity=0.2,
                                name=f"{lg} - pad_i", legendgroup=lg, showlegend=False))
        # pad_j
        x,y,z = box_j.vertices.T; I,J,K = box_j.faces.T
        traces.append(go.Mesh3d(x=x,y=y,z=z,i=I,j=J,k=K,
                                color=col, opacity=0.2,
                                name=f"{lg} - pad_j", legendgroup=lg, showlegend=False))
        # 중심선
        traces.append(go.Scatter3d(
            x=[ci[0], cj[0]], y=[ci[1], cj[1]], z=[ci[2], cj[2]],
            mode="lines", line=dict(color=col, width=5),
            name=f"{lg} - centerline", legendgroup=lg, showlegend=False))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Feasible Pairs — Pads only (count={len(ok_idx)})",
        margin=dict(l=0,r=0,t=48,b=0),
        showlegend=True,
        legend=dict(groupclick="togglegroup"),  # ← 그룹 단위 토글
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig

# pad mesh 없는 버전
def visualize_feasible_pairs_with_yaw(result, reports,
                             show_silhouette=True,
                             pad_w=PadParams.pad_w,
                             pad_h=PadParams.pad_h,
                             pad_d=PadParams.pad_d,
                             show=False):
    mesh = result.get("mesh_quad")
    mesh_patches = result.get("mesh_patches")
    patches = {p["id"]: p for p in result["patches"]}
    pairs = result.get("top_k", [])

    ok_idx = [r["pair_index"] for r in reports if r.get("feasible")]
    # print(ok_idx)

    if not ok_idx:
        raise ValueError("feasible pair가 없습니다.")

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin,ymax-ymin,zmax-zmin)

    traces=[]
    if show_silhouette:
        traces.append(mesh3d_from_trimesh(mesh, color="#cfcfcf", opacity=0.5, name="mesh"))

    for k, idx in enumerate(ok_idx):
        if idx >= len(pairs): continue
        cand = pairs[idx]
        yaw_deg = reports[k]['feasible_yaw']
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in patches or pid_j not in patches: continue
        pi, pj = patches[pid_i], patches[pid_j]
        col = hsv(k, len(ok_idx), s=0.55, v=0.95)

        # pad boxes + rot
        fi, fj = reports[k]["face_i"], reports[k]["face_j"]
        ci = mesh_patches.vertices[mesh_patches.faces[fi]].mean(axis=0)
        cj = mesh_patches.vertices[mesh_patches.faces[fj]].mean(axis=0)

        # legendgroup으로 두 pad를 하나의 토글 그룹에 묶기
        lg = f"pair {idx} - faces {fi, fj} - yaw {yaw_deg}"
        # proxy(범례 핸들) – 클릭 시 그룹 전체 토글
        traces.append(go.Scatter3d(
            x=[ci[0]], y=[ci[1]], z=[ci[2]],
            mode="markers", marker=dict(size=1, opacity=0.0),
            name=lg, legendgroup=lg, showlegend=True))

        # 중심선
        traces.append(go.Scatter3d(
            x=[ci[0], cj[0]], y=[ci[1], cj[1]], z=[ci[2], cj[2]],
            mode="lines", line=dict(color=col, width=5),
            name=f"{lg} - centerline", legendgroup=lg, showlegend=False))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Feasible Pairs — Pads only (count={len(ok_idx)})",
        margin=dict(l=0,r=0,t=48,b=0),
        showlegend=True,
        legend=dict(groupclick="togglegroup"),  # ← 그룹 단위 토글
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig

def visualize_feasible_pairs_with_cylinder(result, reports,
    show_silhouette=True,
    pad_w=PadParams.pad_w,
    pad_h=PadParams.pad_h,
    pad_d=PadParams.pad_d,
    lift_mm: float = 2.0,       # 원기둥 생성에 필요
    show=False
):

    mesh = result.get("mesh_quad")
    mesh_patches = result.get("mesh_patches")
    patches = {p["id"]: p for p in result["patches"]}
    pairs = result.get("top_k", [])

    ok_idx = [r["pair_index"] for r in reports if r.get("feasible")]
    # print(ok_idx)

    if not ok_idx:
        raise ValueError("feasible pair가 없습니다.")

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin,ymax-ymin,zmax-zmin)

    traces=[]
    if show_silhouette:
        traces.append(mesh3d_from_trimesh(mesh, color="#cfcfcf", opacity=0.5, name="mesh"))

    for k, idx in enumerate(ok_idx):
        if idx >= len(pairs): continue
        cand = pairs[idx]
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in patches or pid_j not in patches: continue
        pi, pj = patches[pid_i], patches[pid_j]
        col = hsv(k, len(ok_idx), s=0.55, v=0.95)

        # pad boxes + rot
        fi, fj = reports[k]["face_i"], reports[k]["face_j"]
        ni = unit(np.asarray(pi["normal"], float))
        nj = unit(np.asarray(pj["normal"], float))
        ci = mesh_patches.vertices[mesh_patches.faces[fi]].mean(axis=0)
        cj = mesh_patches.vertices[mesh_patches.faces[fj]].mean(axis=0)

        # legendgroup 생성
        lg = f"Pair {idx} | Faces ({fi}, {fj})"

        # --- 원기둥 생성 및 추가 ---
        # cyl_i = make_pad_cylinder_at_patch({"centroid": ci, "normal": ni}, pad_w, pad_h, pad_d, lift_mm)
        # cyl_j = make_pad_cylinder_at_patch({"centroid": cj, "normal": nj}, pad_w, pad_h, pad_d, lift_mm)

        # # 원기둥 i trace 추가
        # x,y,z = cyl_i.vertices.T; I,J,K = cyl_i.faces.T
        # traces.append(go.Mesh3d(x=x,y=y,z=z,i=I,j=J,k=K,
        #                         color=col, opacity=0.2,
        #                         name=f"{lg} - cyl_i", legendgroup=lg, showlegend=False))
        # # 원기둥 j trace 추가
        # x,y,z = cyl_j.vertices.T; I,J,K = cyl_j.faces.T
        # traces.append(go.Mesh3d(x=x,y=y,z=z,i=I,j=J,k=K,
        #                         color=col, opacity=0.2,
        #                         name=f"{lg} - cyl_j", legendgroup=lg, showlegend=False))

        # 중심선
        traces.append(go.Scatter3d(
            x=[ci[0], cj[0]], y=[ci[1], cj[1]], z=[ci[2], cj[2]],
            mode="lines", line=dict(color=col, width=5), # 중심선은 원래 색상
            name=f"{lg}", legendgroup=lg, showlegend=True
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Feasible Swept Volumes (Cylinders) (count={len(ok_idx)})", # 제목 변경
        margin=dict(l=0, r=0, t=48, b=0),
        showlegend=True,
        legend=dict(groupclick="togglegroup"),
        scene=dict(
            xaxis=dict(range=[cx - max_range, cx + max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy - max_range, cy + max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz - max_range, cz + max_range], showgrid=False, zeroline=False),
            aspectmode="cube" # aspectmode 변경 (데이터 비율에 맞게)
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig

# pad mesh 없는 버전
def visualize_feasible_pairs(result, reports,
                             show_silhouette=True,
                             pad_w=PadParams.pad_w,
                             pad_h=PadParams.pad_h,
                             pad_d=PadParams.pad_d,
                             show=False):
    mesh = result.get("mesh_quad")
    mesh_patches = result.get("mesh_patches")
    patches = {p["id"]: p for p in result["patches"]}
    pairs = result.get("top_k", [])

    ok_idx = [r["pair_index"] for r in reports if r.get("feasible")]
    # print(ok_idx)

    if not ok_idx:
        raise ValueError("feasible pair가 없습니다.")

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin,ymax-ymin,zmax-zmin)

    traces=[]
    if show_silhouette:
        traces.append(mesh3d_from_trimesh(mesh, color="#cfcfcf", opacity=0.5, name="mesh"))

    for k, idx in enumerate(ok_idx):
        if idx >= len(pairs): continue
        cand = pairs[idx]
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in patches or pid_j not in patches: continue
        pi, pj = patches[pid_i], patches[pid_j]
        col = hsv(k, len(ok_idx), s=0.55, v=0.95)

        # pad boxes + rot
        fi, fj = reports[k]["face_i"], reports[k]["face_j"]
        ci = mesh_patches.vertices[mesh_patches.faces[fi]].mean(axis=0)
        cj = mesh_patches.vertices[mesh_patches.faces[fj]].mean(axis=0)

        # legendgroup으로 두 pad를 하나의 토글 그룹에 묶기
        lg = f"pair {idx} - faces {fi, fj}"
        # proxy(범례 핸들) – 클릭 시 그룹 전체 토글
        traces.append(go.Scatter3d(
            x=[ci[0]], y=[ci[1]], z=[ci[2]],
            mode="markers", marker=dict(size=1, opacity=0.0),
            name=lg, legendgroup=lg, showlegend=True))

        # 중심선
        traces.append(go.Scatter3d(
            x=[ci[0], cj[0]], y=[ci[1], cj[1]], z=[ci[2], cj[2]],
            mode="lines", line=dict(color=col, width=5),
            name=f"{lg} - centerline", legendgroup=lg, showlegend=False))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Feasible Pairs — Pads only (count={len(ok_idx)})",
        margin=dict(l=0,r=0,t=48,b=0),
        showlegend=True,
        legend=dict(groupclick="togglegroup"),  # ← 그룹 단위 토글
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig


# -------------------------------
# 4) feasible patch pair + 그리퍼 좌표계
# -------------------------------
def visualize_feasible_pairs_pads_gripper(result, reports,
                                        show_silhouette=True,
                                        pad_w=PadParams.pad_w,
                                        pad_h=PadParams.pad_h,
                                        pad_d=PadParams.pad_d,
                                        show=False):
    mesh = result.get("mesh_quad")
    mesh_patches = result.get("mesh_patches")
    patches = {p["id"]: p for p in result["patches"]}
    pairs = result.get("top_k", [])

    ok_idx = [r["pair_index"] for r in reports if r.get("feasible")]
    if not ok_idx:
        raise ValueError("feasible pair가 없습니다.")

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin,ymax-ymin,zmax-zmin)

    traces=[]
    if show_silhouette:
        traces.append(mesh3d_from_trimesh(mesh, color="#cfcfcf", opacity=0.8, name="mesh"))

    for k, idx in enumerate(ok_idx):
        if idx >= len(pairs): continue
        cand = pairs[idx]
        yaw_deg = reports[k].get('feasible_yaw', 0.0)
        pid_i, pid_j = cand["patch_i"], cand["patch_j"]
        if pid_i not in patches or pid_j not in patches: continue
        pi, pj = patches[pid_i], patches[pid_j]
        col = hsv(k, len(ok_idx), s=0.55, v=0.95)

        # pad boxes + rot
        fi, fj = reports[k]["face_i"], reports[k]["face_j"]
        ni = unit(np.asarray(pi["normal"], float))
        nj = unit(np.asarray(pj["normal"], float))
        ci = mesh_patches.vertices[mesh_patches.faces[fi]].mean(axis=0)
        cj = mesh_patches.vertices[mesh_patches.faces[fj]].mean(axis=0)
        box_i = make_pad_box_at_patch_with_yaw({"centroid": ci, "normal": ni}, pad_w, pad_h, pad_d, yaw_deg= yaw_deg)
        box_j = make_pad_box_at_patch_with_yaw({"centroid": cj, "normal": nj}, pad_w, pad_h, pad_d, yaw_deg=-yaw_deg)

        # legendgroup 묶기
        lg = f"pair {idx}"
        # ci = np.asarray(pi["centroid"], float); cj = np.asarray(pj["centroid"], float)
        traces.append(go.Scatter3d(
            x=[ci[0]], y=[ci[1]], z=[ci[2]],
            mode="markers", marker=dict(size=1, opacity=0.0),
            name=lg, legendgroup=lg, showlegend=True
        ))
        # 그리퍼 패드 시각화
        # pad_i
        x,y,z = box_i.vertices.T; I,J,K = box_i.faces.T
        traces.append(go.Mesh3d(x=x,y=y,z=z,i=I,j=J,k=K,
                                color=col, opacity=0.2,
                                name=f"{lg} - pad_i", legendgroup=lg, showlegend=False))
        # pad_j
        x,y,z = box_j.vertices.T; I,J,K = box_j.faces.T
        traces.append(go.Mesh3d(x=x,y=y,z=z,i=I,j=J,k=K,
                                color=col, opacity=0.2,
                                name=f"{lg} - pad_j", legendgroup=lg, showlegend=False))

        # gripper frame
        H_grip, stroke = build_gripper_pose_obj({"centroid": ci, "normal": ni}, {"centroid": cj, "normal": nj}, yaw_deg)
        
        traces += frame_traces(H_grip, lg, scale=15.0, legendgroup=lg, showlegend=False)

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Feasible Pairs — Pads + Gripper Frames (count={len(ok_idx)})",
        margin=dict(l=0,r=0,t=48,b=0),
        showlegend=True,
        legend=dict(groupclick="togglegroup"),
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig


# -------------------------------
# 기타 임시 함수
# -------------------------------
def visualize_mesh_with_edges(mesh, color="#166534", edge_color="#166534", opacity=0.5,
                              edge_width=4, show=True):
    """
    주어진 trimesh.Trimesh를 단순히 시각화.
    - mesh: trimesh.Trimesh
    - color: 표면 색
    - edge_color: 경계선 색
    - opacity: 표면 투명도
    - edge_width: 경계선 두께
    """
    # bounding box로 범위 계산
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin, ymax-ymin, zmax-zmin)

    traces = []

    # 표면 (Mesh3d)
    x, y, z = mesh.vertices.T
    I, J, K = mesh.faces.T
    traces.append(go.Mesh3d(
        x=x, y=y, z=z,
        i=I, j=J, k=K,
        color=color, opacity=opacity,
        name="mesh"
    ))

    # edge (Wireframe)
    edges = mesh.edges_unique
    verts = mesh.vertices
    xe, ye, ze = [], [], []
    for (v0, v1) in edges:
        xe += [verts[v0][0], verts[v1][0], None]
        ye += [verts[v0][1], verts[v1][1], None]
        ze += [verts[v0][2], verts[v1][2], None]
    traces.append(go.Scatter3d(
        x=xe, y=ye, z=ze,
        mode="lines",
        line=dict(color=edge_color, width=edge_width),
        name="edges"
    ))

    # figure
    fig = go.Figure(traces)
    fig.update_layout(
        title="Mesh with Edges",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=True,
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig


def visualize_report_faces(result, report, 
                           show_silhouette=True, show=True):
    """
    단일 report에 대한 face 시각화
    - mesh_quad: 반투명 회색
    - patch i, j 의 모든 face: 단일 색상
    - face_i, face_j: 강조 색상
    """
    mesh_sil = result["mesh_quad"]
    mesh_p   = result["mesh_patches"]
    patches  = {p["id"]: p for p in result["patches"]}

    pid_i, pid_j = report["patch_i"], report["patch_j"]
    fi_sel, fj_sel = int(report["face_i"]), int(report["face_j"])

    pi, pj = patches[pid_i], patches[pid_j]

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh_sil.bounds
    cx, cy, cz   = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range    = max(xmax-xmin, ymax-ymin, zmax-zmin)

    traces = []
    if show_silhouette:
        traces.append(mesh3d_from_trimesh(mesh_sil, None,
                                          color="#cfcfcf", opacity=0.15,
                                          name="silhouette"))

    # Patch i (연두색)
    fidx_i = set(pi["face_indices"])
    traces.append(mesh3d_from_trimesh(mesh_p, fidx_i, color="#99ff99", opacity=0.5,
                                      name=f"patch {pid_i} faces"))
    edge_i = boundary_edges_trace(mesh_p, fidx_i, color="#008800", width=2,
                                  name=f"patch {pid_i} edges")
    if edge_i is not None:
        traces.append(edge_i)

    # Patch j (하늘색)
    fidx_j = set(pj["face_indices"])
    traces.append(mesh3d_from_trimesh(mesh_p, fidx_j, color="#99ccff", opacity=0.5,
                                      name=f"patch {pid_j} faces"))
    edge_j = boundary_edges_trace(mesh_p, fidx_j, color="#004488", width=2,
                                  name=f"patch {pid_j} edges")
    if edge_j is not None:
        traces.append(edge_j)

    # 강조 face_i (빨강)
    traces.append(mesh3d_from_trimesh(mesh_p, {fi_sel}, color="#ff0000", opacity=1.0,
                                      name=f"face {fi_sel} (patch {pid_i})"))

    # 강조 face_j (파랑)
    traces.append(mesh3d_from_trimesh(mesh_p, {fj_sel}, color="#0000ff", opacity=1.0,
                                      name=f"face {fj_sel} (patch {pid_j})"))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Report pair {report['pair_index']} (patch {pid_i}, {pid_j})",
        margin=dict(l=0,r=0,t=48,b=0),
        showlegend=True,
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")
    return fig


def visualize_patch_pair_faces(result, report, show=True):
    """
    result: grasping.compute_best_patch_pairs(...) 출력 dict
    report: reports[k] (단일 pair)
    
    patch_i, patch_j 에 포함된 모든 face를 각각 따로 토글 가능하도록 시각화
    """
    mesh_sil = result["mesh_quad"]
    mesh_p = result["mesh_patches"]
    patches = {p["id"]: p for p in result["patches"]}
    
    pid_i, pid_j = report["patch_i"], report["patch_j"]
    pi, pj = patches[pid_i], patches[pid_j]

    verts = np.asarray(mesh_p.vertices)
    faces = np.asarray(mesh_p.faces)

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh_p.bounds
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    max_range  = max(xmax-xmin, ymax-ymin, zmax-zmin)

    traces = []

    traces.append(mesh3d_from_trimesh(mesh_sil, None,
                                        color="#cfcfcf", opacity=0.15,
                                        name="silhouette"))

    # patch i faces (빨강 계열)
    for fidx in pi["face_indices"]:
        tri = faces[fidx]
        x,y,z = verts[tri].T
        traces.append(go.Mesh3d(
            x=x, y=y, z=z,
            i=[0], j=[1], k=[2],
            color="red",
            opacity=0.6,
            name=f"patch {pid_i} face {fidx}",
            legendgroup=f"patch {pid_i} face {fidx}",
            showlegend=True
        ))

    # patch j faces (파랑 계열)
    for fidx in pj["face_indices"]:
        tri = faces[fidx]
        x,y,z = verts[tri].T
        traces.append(go.Mesh3d(
            x=x, y=y, z=z,
            i=[0], j=[1], k=[2],
            color="blue",
            opacity=0.6,
            name=f"patch {pid_j} face {fidx}",
            legendgroup=f"patch {pid_j} face {fidx}",
            showlegend=True
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Patch pair {pid_i} & {pid_j} (faces)",
        margin=dict(l=0,r=0,t=48,b=0),
        showlegend=True,
        legend=dict(groupclick="togglegroup"),  # 그룹 단위 토글
        scene=dict(
            xaxis=dict(range=[cx-max_range, cx+max_range], showgrid=False, zeroline=False),
            yaxis=dict(range=[cy-max_range, cy+max_range], showgrid=False, zeroline=False),
            zaxis=dict(range=[cz-max_range, cz+max_range], showgrid=False, zeroline=False),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    return fig


def visualize_frames(H_dict, scale=30.0, show=True, H_OdEn = None, result = None, save = False, save_path = None):
    """
    여러 좌표계 프레임을 시각화
    H_dict: {이름: 4x4 행렬}
    H_OdEn: 엔드이펙터 기준 오브젝트 HM

    """
    traces = []

    mesh_sil = result["mesh_quad"]
    # H_OdEn 으로 mesh 회전 후 fig에 추가
    mesh_tf = mesh_sil.copy()
    mesh_tf.apply_transform(H_OdEn)
    x, y, z = mesh_tf.vertices.T
    i, j, k = mesh_tf.faces.T
    traces.append(go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color="#cfcfcf", opacity=0.3,
        name="object_mesh", legendgroup="object_mesh", showlegend=True
    ))

    for i, (name, H) in enumerate(H_dict.items()):
        traces += frame_traces(H, name, scale=scale, legendgroup=name, showlegend=True)

    fig = go.Figure(traces)
    fig.update_layout(
        title="Coordinate Frames Visualization",
        margin=dict(l=0,r=0,t=30,b=0),
        showlegend=True,
        legend=dict(groupclick="togglegroup"),  # 그룹 단위 토글
        scene=dict(
            xaxis=dict(showgrid=True, zeroline=True, range=[-1000, 1000]),
            yaxis=dict(showgrid=True, zeroline=True, range=[-1000, 1000]),
            zaxis=dict(showgrid=True, zeroline=True, range=[-1000, 1000]),
            aspectmode="cube"
        )
    )
    if show:
        fig.show(renderer="browser")

    if save:
        fig.write_html(save_path)
    return fig


